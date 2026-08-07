"""在 GPU 上遍历所有训练样本的前向传播，找出触发 gather 越界的样本"""
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

device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.cuda.set_device(device_id)
print(f"Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")

# 加载数据
df = pd.read_csv('dataProcess/custom/custom_train.tsv', sep='\t', header=None)
print(f"总样本: {len(df)}")
with open('dataProcess/custom/custom_pdb_dict_AgAb.pickle', 'rb') as f:
    encoding_dict = pickle.load(f)
with open('dataProcess/custom/custom_cdr_dict.pickle', 'rb') as f:
    con_cdr_dict = pickle.load(f)

h5fi = h5py.File('dataProcess/custom/custom_all_proteins.h5', 'r')

# 构建模型到 GPU
embedding_model = FullyConnectedEmbed(6165, 46, 0.5).cuda()
embedding_modelAg = FullyConnectedEmbed(46, 46, 0.5).cuda()
contact_model = ContactCNN(46, 23, 7).cuda()
model = ModelInteraction(embedding_model, embedding_modelAg, contact_model, True)
modelSeq = DeepPPI(50, 5).cuda()
model.eval()
modelSeq.eval()

# 遍历样本测试
fail = 0
for idx in range(min(len(df), 100)):
    n0 = df[0][idx]
    n1 = df[1][idx]
    catsite = int(df[3][idx])
    z_a = torch.from_numpy(h5fi[n0][:, :]).cuda()
    z_b = torch.from_numpy(h5fi[n1][:, :]).cuda()
    index_cdrlist = [a for a, b in enumerate(con_cdr_dict[n1]) if b == 1]
    meta_a = torch.tensor(encoding_dict[n0]).unsqueeze(0).float().cuda()
    z_a = torch.cat([z_a, meta_a], 2)

    # 检查 CDR 数量 vs VH/VL 长度
    vh_len = catsite if catsite > 0 else -catsite
    vl_len = z_b.shape[1] - abs(catsite)
    cdr_vh = [i for i in index_cdrlist if i < vh_len]
    cdr_vl = [i - vh_len for i in index_cdrlist if i >= vh_len]

    # 检查 gather 是否可能越界：索引数量 > 对应长度
    issue = []
    if len(cdr_vh) > vh_len:
        issue.append(f"CDR_VH数量({len(cdr_vh)}) > VH长度({vh_len})")
    if len(cdr_vl) > vl_len:
        issue.append(f"CDR_VL数量({len(cdr_vl)}) > VL长度({vl_len})")

    if issue:
        fail += 1
        print(f"  ⚠️ 样本{idx} ({n0}+{n1}): {'; '.join(issue)}")

print(f"\n检查完成: {fail}/{min(len(df),100)} 个样本存在 gather 越界风险")

# 额外检查：测试集的样本
print("\n=== 检查测试集 ===")
df2 = pd.read_csv('dataProcess/custom/custom_test.tsv', sep='\t', header=None)
fail2 = 0
for idx in range(len(df2)):
    n1 = df2[1][idx]
    catsite = int(df2[3][idx])
    z_b_len = h5fi[n1].shape[1]
    vh_len = abs(catsite)
    vl_len = z_b_len - abs(catsite)
    index_cdrlist = [a for a, b in enumerate(con_cdr_dict[n1]) if b == 1]
    cdr_vh = [i for i in index_cdrlist if i < vh_len]
    cdr_vl = [i - vh_len for i in index_cdrlist if i >= vh_len]
    issue = []
    if len(cdr_vh) > vh_len:
        issue.append(f"CDR_VH({len(cdr_vh)})>VH({vh_len})")
    if len(cdr_vl) > vl_len:
        issue.append(f"CDR_VL({len(cdr_vl)})>VL({vl_len})")
    if issue:
        fail2 += 1
        print(f"  ⚠️ 测试样本{idx} ({df2[0][idx]}+{n1}): {'; '.join(issue)}")
print(f"测试集: {fail2}/{len(df2)} 个样本有越界风险")
