import argparse
import math
import os
import pickle
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# This file lives at <repo>/EpiScan/EpiScan/EpiScan/commands/testset_cv.py.
# The importable package root is the directory whose child is `EpiScan/`
# containing commands/ — i.e. <repo>/EpiScan/EpiScan. Add it to sys.path so
# `import EpiScan` resolves to the real package (with commands/).
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from EpiScan.commands.utils import log
from EpiScan.models.contact_sep import ContactCNN
from EpiScan.models.embedding import FullyConnectedEmbed
from EpiScan.models.interaction_sep import ModelInteraction


def build_model(use_cuda):
    embedding_model = FullyConnectedEmbed(6165, 46, 0.5)
    embedding_modelAg = FullyConnectedEmbed(46, 46, 0.5)
    contact_model = ContactCNN(46, 23, 7)
    modelCon = ModelInteraction(embedding_model, embedding_modelAg, contact_model, use_cuda)
    return modelCon


def load_checkpoint(modelCon, path, use_cuda):
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict):
        modelCon.load_state_dict(loaded)
    else:
        modelCon = loaded
    # Critical: disable dropout for inference (matches train eval path)
    modelCon.eval()
    # The use_cuda flag is baked into the saved object at training time;
    # sync it to the actual runtime device so map_predict moves tensors correctly.
    modelCon.use_cuda = use_cuda
    return modelCon


def predict_one(model, n0, n1, embeddings, encoding_dict, con_cdr_dict, catsite, use_cuda):
    """Single-pair inference, mirrors predict_cmap_interaction (fixed eval)."""
    z_a = embeddings[n0]
    z_b = embeddings[n1]

    # Clamp CDR indices to the actual antibody embedding length
    ab_len = z_b.shape[1]
    index_cdrlist = [a for a, b in enumerate(con_cdr_dict[n1]) if b == 1]
    index_cdrlist = [a for a in index_cdrlist if a < ab_len]

    meta_a = torch.tensor(encoding_dict[n0]).unsqueeze(0)
    meta_a = meta_a.to(torch.float)
    # Align antigen meta-feature length to embedding length
    if meta_a.shape[1] < z_a.shape[1]:
        pad = torch.zeros((1, z_a.shape[1] - meta_a.shape[1], meta_a.shape[2]))
        meta_a = torch.cat([meta_a, pad], 1)
    elif meta_a.shape[1] > z_a.shape[1]:
        meta_a = meta_a[:, : z_a.shape[1], :]
    z_a = torch.cat([z_a[:, :, :], meta_a], 2)

    if use_cuda:
        z_a = z_a.cuda()
        z_b = z_b.cuda()
    with torch.no_grad():
        cm, ph = model.map_predict(z_a, z_b, catsite, index_cdrlist)
        prob = torch.mean(cm[:, :, :, :], 3).squeeze()
    prob = prob.cpu().float()
    return prob


def eval_fold(probs, labels_str):
    """Same metric logic as interaction_eval: ACC micro, others per-sample macro, thr 0.5."""
    esp = 1e-6
    guess_cutoff = 0.5
    correct_list, pr_list, re_list, f1_list, mcc_list, auc_list, aupr_list, len_list = [], [], [], [], [], [], [], []

    from sklearn.metrics import average_precision_score, roc_auc_score

    for prob, lab_str in zip(probs, labels_str):
        b_ii = len(prob)
        lab = np.array(list(map(int, list(str(lab_str)))))[:b_ii]
        p_guess = (guess_cutoff < prob.numpy()).astype(float)
        lab = lab.astype(float)

        correct_list.append(np.sum(p_guess == lab).item())
        len_list.append(b_ii)

        tp = np.sum(lab * p_guess).item()
        tn = np.sum((1 - lab) * (1 - p_guess)).item()
        fp = np.sum((1 - lab) * p_guess).item()
        fn = np.sum(lab * (1 - p_guess)).item()
        pr = tp / (tp + fp + esp)
        re = tp / (tp + fn + esp)
        f1 = 2 * pr * re / (pr + re + esp)
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + esp
        mcc = (tp * tn - fp * fn) / denom
        pr_list.append(pr)
        re_list.append(re)
        f1_list.append(f1)
        mcc_list.append(mcc)

        auc_list.append(roc_auc_score(lab, prob.numpy()) if lab.sum() > 0 and (1 - lab).sum() > 0 else 0.5)
        aupr_list.append(average_precision_score(lab, prob.numpy()) if lab.sum() > 0 else 0.5)

    acc = np.mean(correct_list) / np.mean(len_list)
    return {
        "acc": acc,
        "auroc": np.mean(auc_list),
        "auprc": np.mean(aupr_list),
        "precision": np.mean(pr_list),
        "recall": np.mean(re_list),
        "f1": np.mean(f1_list),
        "mcc": np.mean(mcc_list),
    }


def main(args):
    device = args.device
    use_cuda = (device > -1) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)

    log(f"Using CUDA device {device}" if use_cuda else "Using CPU", print_also=True)

    # ---- Load test TSV (same columns as train: prot1, prot2, label, catsite)
    test_df = pd.read_csv(args.test, sep="\t", header=None)
    test_df.columns = ["prot1", "prot2", "label", "catnum"]
    log(f"Loaded {len(test_df)} test pairs from {args.test}", print_also=True)

    # ---- Load only the proteins needed for the test set
    all_proteins = set(test_df["prot1"]).union(test_df["prot2"])
    h5fi = h5py.File(args.embedding, "r")
    embeddings = {}
    for prot_name in tqdm(all_proteins, desc="Loading embeddings"):
        embeddings[prot_name] = torch.from_numpy(h5fi[prot_name][:, :]).float()
    h5fi.close()
    log(f"Loaded {len(embeddings)} protein embeddings", print_also=True)

    with open(args.pdb_dict, "rb") as fh:
        encoding_dict = pickle.load(fh)
    with open(args.cdr_dict, "rb") as fh:
        con_cdr_dict = pickle.load(fh)

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for ckpt in args.checkpoints:
        modelCon = build_model(use_cuda)
        modelCon = load_checkpoint(modelCon, ckpt, use_cuda)
        if use_cuda:
            modelCon.cuda()
        name = os.path.splitext(os.path.basename(ckpt))[0].replace("_best_final", "")

        probs = []
        for i in tqdm(range(len(test_df)), desc=f"Infer {name}"):
            n0 = test_df["prot1"][i]
            n1 = test_df["prot2"][i]
            catsite = int(test_df["catnum"][i])
            prob = predict_one(modelCon, n0, n1, embeddings, encoding_dict, con_cdr_dict, catsite, use_cuda)
            probs.append(prob)

        metrics = eval_fold(probs, test_df["label"].values)

        # Save raw probs for this fold
        np.savez(os.path.join(args.outdir, f"test_probs_{name}.npz"), **{f"s{i}": p.numpy() for i, p in enumerate(probs)})

        row = {"fold": name}
        row.update({k: round(v, 6) for k, v in metrics.items()})
        rows.append(row)
        log(f"{name}: ACC={metrics['acc']:.4f} AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
            f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f} F1={metrics['f1']:.4f} MCC={metrics['mcc']:.4f}",
            print_also=True)
        del modelCon
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "testset_metrics.csv"), index=False)
    log(f"Saved metrics to {args.outdir}/testset_metrics.csv", print_also=True)
    print("\n" + df.to_string(index=False))


def add_args(parser):
    parser.add_argument("--test", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--pdb-dict", required=True)
    parser.add_argument("--cdr-dict", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--outdir", default="output/testset_cv")
    parser.add_argument("--device", type=int, default=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
