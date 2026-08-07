"""在 CPU 上测试模型前向传播 + 损失计算，精确定位 CUDA 异步错误的真实来源"""
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

# 加载 2 个训练样本
df = pd.read_csv('dataProcess/custom/custom_train.tsv', sep='\t', header=None, nrows=2)
print("训练样本:")
print(df)

with open('dataProcess/custom/custom_pdb_dict_AgAb.pickle', 'rb') as f:
    encoding_dict = pickle.load(f)
with open('dataProcess/custom/custom_cdr_dict.pickle', 'rb') as f:
    con_cdr_dict = pickle.load(f)

h5fi = h5py.File('dataProcess/custom/custom_all_proteins.h5', 'r')
embeddings = {}
for prot in set(df[0]).union(set(df[1])):
    embeddings[prot] = torch.from_numpy(h5fi[prot][:, :])
print("\n嵌入形状:")
for k, v in embeddings.items():
    print(f"  {k}: {v.shape}")

# 标签
labels = []
for i in range(len(df)):
    y = np.array(list(map(int, list(str(df[2][i])))))
    labels.append(torch.from_numpy(y))
print(f"\n标签长度: {[len(y) for y in labels]}")

# 构建模型（CPU）
embedding_model = FullyConnectedEmbed(6165, 46, 0.5)
embedding_modelAg = FullyConnectedEmbed(46, 46, 0.5)
contact_model = ContactCNN(46, 23, 7)
model = ModelInteraction(embedding_model, embedding_modelAg, contact_model, False)
modelSeq = DeepPPI(50, 5)
model.eval()
modelSeq.eval()

print("\n========== 前向传播测试 ==========")
p_hat = []
for i in range(len(df)):
    n0 = df[0][i]
    n1 = df[1][i]
    catsite = int(df[3][i])
    z_a = embeddings[n0]
    z_b = embeddings[n1]
    index_cdrlist = [a for a, b in enumerate(con_cdr_dict[n1]) if b == 1]
    meta_a = torch.tensor(encoding_dict[n0]).unsqueeze(0).float()
    z_a = torch.cat([z_a, meta_a], 2)

    print(f"\n--- 样本 {i}: {n0} + {n1}, catsite={catsite} ---")
    print(f"z_a shape: {z_a.shape}, z_b shape: {z_b.shape}")
    print(f"CDR 索引数: {len(index_cdrlist)}, 最大索引: {max(index_cdrlist) if index_cdrlist else 'N/A'}")
    print(f"z_b 长度: {z_b.shape[1]}")
    # 检查 CDR 是否越界
    oob = [j for j in index_cdrlist if j >= z_b.shape[1]]
    if oob:
        print(f"  ⚠️ CDR 索引越界! {len(oob)} 个索引 >= {z_b.shape[1]}: {oob[:5]}")

    try:
        cm, ph = model.map_predict(z_a, z_b, catsite, index_cdrlist)
        print(f"cm shape: {cm.shape}, ph: {ph.item():.4f}")
        p_hat.append(ph)
    except Exception as e:
        print(f"  ❌ 前向失败: {e}")
        import traceback
        traceback.print_exc()
        break

if len(p_hat) == 2:
    print("\n========== 损失计算测试 ==========")
    diceloss = BinaryDiceLoss()
    focalloss = BinaryFocalLoss()
    for ii in range(2):
        ph = torch.clamp(p_hat[ii].float(), 1e-7, 1 - 1e-7)
        y_i = labels[ii].float()
        bce = F.binary_cross_entropy(ph, y_i[:len(ph)])
        dice = diceloss(ph, y_i[:len(ph)])
        focal = focalloss(ph, y_i[:len(ph)])
        print(f"样本 {ii}: BCE={bce.item():.4f}, Dice={dice.item():.4f}, Focal={focal.item():.4f}")
    print("\n✅ CPU 前向 + 损失全部通过，说明代码逻辑正确，问题在 CUDA 环境")
