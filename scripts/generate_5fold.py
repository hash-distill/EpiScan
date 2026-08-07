"""
生成 DB1 5 折交叉验证划分
将 162 个复合物（103 训练 + 29 验证 + 30 测试）合并后按表位比例分层分 5 折

输出:
  - dataProcess/public/fold{0-4}_train.tsv
  - dataProcess/public/fold{0-4}_test.tsv
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(BASE, "dataProcess/public")

# 合并 162 个样本
train_df = pd.read_csv(os.path.join(PUBLIC_DIR, "public_sep_trainAg.tsv"), sep='\t', header=None)
val_df = pd.read_csv(os.path.join(PUBLIC_DIR, "public_sep_valAg.tsv"), sep='\t', header=None)
test_df = pd.read_csv(os.path.join(PUBLIC_DIR, "public_sep_testAg.tsv"), sep='\t', header=None)
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"总样本数: {len(all_df)} (DB1 162 复合物)")

# 按表位比例分层
def epitope_ratio(label_str):
    if isinstance(label_str, str) and len(label_str) > 0:
        label_arr = np.array(list(map(int, list(label_str))))
        return np.mean(label_arr) if len(label_arr) > 0 else 0
    return 0

all_df['ratio'] = all_df[2].apply(epitope_ratio)
all_df['ratio_bin'] = pd.cut(all_df['ratio'], bins=5, labels=False)

# 5 折分层交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(all_df, all_df['ratio_bin'])):
    train_fold = all_df.iloc[train_idx]
    test_fold = all_df.iloc[test_idx]
    train_fold[[0, 1, 2, 3]].to_csv(
        os.path.join(PUBLIC_DIR, f'fold{fold}_train.tsv'),
        sep='\t', index=False, header=False
    )
    test_fold[[0, 1, 2, 3]].to_csv(
        os.path.join(PUBLIC_DIR, f'fold{fold}_test.tsv'),
        sep='\t', index=False, header=False
    )
    print(f'Fold {fold}: train={len(train_fold)}, test={len(test_fold)}, '
          f'epitope_ratio={test_fold["ratio"].mean():.4f}')

print("\n✅ DB1 5 折划分完成")
