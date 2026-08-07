"""完整复现 interaction_grad 逻辑，检查 ph 是否有 NaN"""
import sys, os, pickle, h5py
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

# 检查 H5 嵌入是否有 NaN
h5fi = h5py.File('dataProcess/custom/custom_all_proteins.h5', 'r')
keys = list(h5fi.keys())
print("检查 H5 嵌入 NaN:")
nan_count = 0
for k in keys[:50]:
    arr = np.array(h5fi[k][:])
    if np.isnan(arr).any():
        nan_count += 1
        print(f"  ⚠️ {k} 含 NaN!")
print(f"前50个key中 {nan_count} 个含NaN")

# 更全面: 检查训练/测试涉及的所有蛋白
df = pd.read_csv('dataProcess/custom/custom_train.tsv', sep='\t', header=None)
all_prots = set(df[0]).union(set(df[1]))
nan_prots = [p for p in all_prots if np.isnan(np.array(h5fi[p][:])).any()]
print(f"\n全部 {len(all_prots)} 个蛋白中, {len(nan_prots)} 个 H5 嵌入含 NaN")
if nan_prots:
    print(f"  示例: {nan_prots[:10]}")

# 检查 pickle 特征 NaN
with open('dataProcess/custom/custom_pdb_dict_AgAb.pickle', 'rb') as f:
    encoding_dict = pickle.load(f)
nan_pkl = [k for k in all_prots if k in encoding_dict and np.isnan(encoding_dict[k]).any()]
print(f"pickle 特征含 NaN 的蛋白: {len(nan_pkl)}")
if nan_pkl:
    print(f"  示例: {nan_pkl[:10]}")

# 加载模型 CPU
embedding_model = FullyConnectedEmbed(6165, 46, 0.5)
embedding_modelAg = FullyConnectedEmbed(46, 46, 0.5)
contact_model = ContactCNN(46, 23, 7)
model = ModelInteraction(embedding_model, embedding_modelAg, contact_model, False)
modelSeq = DeepPPI(50, 5)
model.eval(); modelSeq.eval()

print("\n========== 完整前向 + 损失 ==========")
with open('dataProcess/custom/custom_cdr_dict.pickle', 'rb') as f:
    con_cdr_dict = pickle.load(f)

for idx in range(2):
    n0 = df[0][idx]; n1 = df[1][idx]; catsite = int(df[3][idx])
    z_a = torch.from_numpy(h5fi[n0][:, :])
    z_b = torch.from_numpy(h5fi[n1][:, :])
    ab_len = z_b.shape[1]
    index_cdrlist = [a for a, b in enumerate(con_cdr_dict[n1]) if b == 1]
    index_cdrlist = [a for a in index_cdrlist if a < ab_len]
    meta_a = torch.tensor(encoding_dict[n0]).unsqueeze(0).float()
    if meta_a.shape[1] < z_a.shape[1]:
        pad = torch.zeros((1, z_a.shape[1] - meta_a.shape[1], meta_a.shape[2]))
        meta_a = torch.cat([meta_a, pad], 1)
    elif meta_a.shape[1] > z_a.shape[1]:
        meta_a = meta_a[:, :z_a.shape[1], :]
    z_a = torch.cat([z_a, meta_a], 2)

    cm, ph_scalar = model.map_predict(z_a, z_b, catsite, index_cdrlist)
    prob_tmp = torch.mean(cm[:,:,:,:],3).squeeze()
    padnum = 50 - (z_a.shape[1] % 50)
    p0Conseq = torch.cat((z_a, z_a[:,-padnum:,:]), 1)
    phatnew,_ = modelSeq(p0Conseq, prob_tmp)

    print(f"\n--- 样本{idx} ({n0}+{n1}) ---")
    print(f"  prob_tmp shape: {prob_tmp.shape}, min={prob_tmp.min().item():.6f}, max={prob_tmp.max().item():.6f}, NaN={torch.isnan(prob_tmp).any().item()}")
    print(f"  ph_scalar: {ph_scalar.item():.6f}")

    # 模拟 interaction_grad 的损失计算
    y_str = str(df[2][idx])
    y_arr = np.array(list(map(int, list(y_str))))
    y_i = torch.from_numpy(y_arr).float()
    ph = torch.clamp(prob_tmp.float(), 1e-7, 1 - 1e-7)
    print(f"  ph(clamp后): min={ph.min().item():.8f}, max={ph.max().item():.8f}, NaN={torch.isnan(ph).any().item()}")
    print(f"  y_i: min={y_i.min().item()}, max={y_i.max().item()}, len={len(y_i)}, ph len={len(ph)}")

    bce = F.binary_cross_entropy(ph, y_i[:len(ph)])
    print(f"  BCE = {bce.item():.4f}")

print("\n✅ CPU 完整逻辑通过 (如果没报错)")
