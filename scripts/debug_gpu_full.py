"""GPU 上遍历训练样本，train 模式，捕获完整 traceback 找出失败原因"""
import sys, os, pickle, h5py, traceback
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())
from EpiScan.models.embedding import FullyConnectedEmbed
from EpiScan.models.contact_sep import ContactCNN
from EpiScan.models.interaction_sep import ModelInteraction
from EpiScan.models.deep_ppi import DeepPPI
from EpiScan.selfLoss.BdiceLoss import BinaryDiceLoss, dice_coeff
from EpiScan.selfLoss.Bfocalloss import BinaryFocalLoss

torch.cuda.set_device(0)
df = pd.read_csv('dataProcess/custom/custom_train.tsv', sep='\t', header=None)
with open('dataProcess/custom/custom_pdb_dict_AgAb.pickle', 'rb') as f:
    encoding_dict = pickle.load(f)
with open('dataProcess/custom/custom_cdr_dict.pickle', 'rb') as f:
    con_cdr_dict = pickle.load(f)
h5fi = h5py.File('dataProcess/custom/custom_all_proteins.h5', 'r')

embedding_model = FullyConnectedEmbed(6165, 46, 0.5).cuda()
embedding_modelAg = FullyConnectedEmbed(46, 46, 0.5).cuda()
contact_model = ContactCNN(46, 23, 7).cuda()
model = ModelInteraction(embedding_model, embedding_modelAg, contact_model, True)
model = model.cuda()  # 整体 cuda，让 theta/lambda_/gamma 也上 GPU
modelSeq = DeepPPI(50, 5).cuda()
model.train(); modelSeq.train()  # train 模式，与真实训练一致

diceloss = BinaryDiceLoss()
focalloss = BinaryFocalLoss()

first_fail = True
for idx in range(len(df)):
    n0 = df[0][idx]; n1 = df[1][idx]; catsite = int(df[3][idx])
    z_a = torch.from_numpy(h5fi[n0][:, :])
    z_b = torch.from_numpy(h5fi[n1][:, :])
    ab_len = z_b.shape[1]
    index_cdrlist = [a for a, b in enumerate(con_cdr_dict[n1]) if b == 1]
    index_cdrlist = [a for a in index_cdrlist if a < ab_len]
    meta_a = torch.tensor(encoding_dict[n0]).unsqueeze(0).float()
    if meta_a.shape[1] < z_a.shape[1]:
        pad = torch.zeros((1, z_a.shape[1] - meta_a.shape[1], meta_a.shape[2])).float()
        meta_a = torch.cat([meta_a, pad], 1)
    elif meta_a.shape[1] > z_a.shape[1]:
        meta_a = meta_a[:, :z_a.shape[1], :]
    z_a = torch.cat([z_a, meta_a], 2).cuda()
    z_b = z_b.cuda()

    y_str = str(df[2][idx])
    y_arr = np.array(list(map(int, list(y_str))))
    y_i = torch.from_numpy(y_arr).float().cuda()

    try:
        cm, ph = model.map_predict(z_a, z_b, catsite, index_cdrlist)
        prob_tmp = torch.mean(cm[:,:,:,:],3).squeeze()
        phat_c = torch.clamp(prob_tmp.float(), 1e-7, 1 - 1e-7)
        nan = torch.isnan(phat_c).any().item()
        rng = (phat_c.min().item(), phat_c.max().item())
        bce = F.binary_cross_entropy(phat_c, y_i[:len(phat_c)])
        print(f"  样本{idx} ({n0}+{n1}) OK: prob范围={rng}, NaN={nan}, BCE={bce.item():.4f}")
    except Exception as e:
        if first_fail:
            print(f"❌ 第一个失败样本 {idx} ({n0}+{n1}): {e}")
            traceback.print_exc()
            first_fail = False
        else:
            print(f"❌ 样本{idx} ({n0}+{n1}): {str(e)[:60]}")

print("\n测试完成")
