"""
生成 Custom 数据 5 折交叉验证划分
基于表位比例进行分层采样

输出:
  - dataProcess/custom/custom_fold{0-4}_train.tsv
  - dataProcess/custom/custom_fold{0-4}_val.tsv
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_TSV = os.path.join(BASE, "dataProcess/custom/custom_train.tsv")
TEST_TSV = os.path.join(BASE, "dataProcess/custom/custom_test.tsv")
OUT_DIR = os.path.join(BASE, "dataProcess/custom")

# 读取全量数据（训练+测试合并做交叉验证）
train_df = pd.read_csv(TRAIN_TSV, sep='\t', header=None)
test_df = pd.read_csv(TEST_TSV, sep='\t', header=None)
all_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"总样本数: {len(all_df)}")

# 计算表位比例作为分层依据
def epitope_ratio(label_str):
    if isinstance(label_str, str) and len(label_str) > 0:
        label_arr = np.array(list(map(int, list(label_str))))
        return np.mean(label_arr) if len(label_arr) > 0 else 0
    return 0

all_df['ratio'] = all_df[2].apply(epitope_ratio)
all_df['ratio_bin'] = pd.cut(all_df['ratio'], bins=5, labels=False)

# 5 折分层交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(all_df, all_df['ratio_bin'])):
    train_fold = all_df.iloc[train_idx]
    val_fold = all_df.iloc[val_idx]
    # 保存时只保留前 4 列（去掉 ratio 相关列）
    train_fold[[0, 1, 2, 3]].to_csv(
        os.path.join(OUT_DIR, f'custom_fold{fold}_train.tsv'),
        sep='\t', index=False, header=False
    )
    val_fold[[0, 1, 2, 3]].to_csv(
        os.path.join(OUT_DIR, f'custom_fold{fold}_val.tsv'),
        sep='\t', index=False, header=False
    )
    print(f'Fold {fold}: train={len(train_fold)}, val={len(val_fold)}, '
          f'epitope_ratio={val_fold["ratio"].mean():.4f}')

print("\n✅ 5 折划分完成")
