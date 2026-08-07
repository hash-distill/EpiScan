"""单独测试一个样本（新进程），定位 device-side assert 的真实来源"""
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

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 100
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
model = model.cuda()
modelSeq = DeepPPI(50, 5).cuda()
model.train(); modelSeq.train()

row = df.iloc[idx]
n0, n1, catsite = row[0], row[1], int(row[3])
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
print(f'样本{idx} ({n0}+{n1}), z_a={tuple(z_a.shape)}, z_b={tuple(z_b.shape)}, catsite={catsite}')
print(f'CDR 索引({len(index_cdrlist)}): {index_cdrlist}')

try:
    cm, ph = model.map_predict(z_a, z_b, catsite, index_cdrlist)
    print(f'✅ forward OK, cm={tuple(cm.shape)}, ph={ph.item():.6f}')
    prob_tmp = torch.mean(cm[:,:,:,:],3).squeeze()
    phat_c = torch.clamp(prob_tmp.float(), 1e-7, 1-1e-7)
    print(f'   prob_tmp: min={prob_tmp.min().item():.6f}, max={prob_tmp.max().item():.6f}, nan={torch.isnan(prob_tmp).any().item()}')
    y_str = str(row[2]); y_arr = np.array(list(map(int, list(y_str))))
    y_i = torch.from_numpy(y_arr).float().cuda()
    bce = F.binary_cross_entropy(phat_c, y_i[:len(phat_c)])
    print(f'   BCE OK: {bce.item():.4f}')
except Exception as e:
    print(f'❌ 失败: {type(e).__name__}: {e}')
    traceback.print_exc()
