"""
生成 EpiScan 训练/测试 TSV 文件 (4列格式: antigen_id, antibody_id, label, catsite)

输入:
  - dataProcess/custom/fasta_label4.5.tsv
  - dataProcess/custom/epitope_ratio_train_pdb_ids4645.txt
  - dataProcess/custom/epitope_ratio_test_pdb_ids4645.txt
  - dataProcess/custom/custom_catsite_dict.pickle

输出:
  - dataProcess/custom/custom_train.tsv
  - dataProcess/custom/custom_test.tsv
"""
import os, pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSV_PATH = os.path.join(BASE_DIR, "dataProcess/custom/fasta_label4.5.tsv")
TRAIN_IDS = os.path.join(BASE_DIR, "dataProcess/custom/epitope_ratio_train_pdb_ids4645.txt")
TEST_IDS = os.path.join(BASE_DIR, "dataProcess/custom/epitope_ratio_test_pdb_ids4645.txt")
CATSITE_PATH = os.path.join(BASE_DIR, "dataProcess/custom/custom_catsite_dict.pickle")
OUT_TRAIN = os.path.join(BASE_DIR, "dataProcess/custom/custom_train.tsv")
OUT_TEST = os.path.join(BASE_DIR, "dataProcess/custom/custom_test.tsv")

# 读取 catsite 字典
with open(CATSITE_PATH, 'rb') as f:
    catsite_dict = pickle.load(f)
print(f"加载 catsite 字典: {len(catsite_dict)} 条")

# 读取原始数据
df = pd.read_csv(TSV_PATH, sep='\t')
print(f"加载标注数据: {len(df)} 条")

# 读取 ID 列表
def load_ids(filepath):
    ids = set()
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                ids.add((parts[0].strip(), parts[1].strip()))
    return ids

train_ids = load_ids(TRAIN_IDS)
test_ids = load_ids(TEST_IDS)
print(f"训练 ID: {len(train_ids)}, 测试 ID: {len(test_ids)}")

# 生成 TSV
def to_tsv(df, id_set, catsite_dict, output_path, split_name=""):
    records = []
    missing_catsite = 0
    missing_pair = 0
    for _, row in df.iterrows():
        pair = (row['antigen_id'], row['antibody_id'])
        if pair in id_set:
            catsite = catsite_dict.get(row['antibody_id'])
            if catsite is None:
                # fallback
                ab_seq = str(row['antibody_seq'])
                catsite = int(len(ab_seq) * 0.6)
                missing_catsite += 1
            # label 补零到 2000 字符（训练脚本的要求）
            label_str = str(row['antigen_label'])
            records.append([
                row['antigen_id'],
                row['antibody_id'],
                label_str,
                catsite
            ])

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"{split_name}: 写入 {len(out_df)} 条记录到 {output_path}"
          f" (缺 catsite: {missing_catsite})")

to_tsv(df, train_ids, catsite_dict, OUT_TRAIN, "训练集")
to_tsv(df, test_ids, catsite_dict, OUT_TEST, "测试集")
print("✅ TSV 生成完成")
