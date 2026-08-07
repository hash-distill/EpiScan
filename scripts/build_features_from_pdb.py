"""
EpiScan Custom 数据特征构建脚本
从 PDB 结构文件提取抗原 66 维特征 + 抗体特征 + CDR 标注 + Catsite

输入:
  - dataProcess/custom/pdb/          : PDB 结构文件目录 (4742 个)
  - dataProcess/custom/fasta_label4.5.tsv  : 标注数据

输出:
  - dataProcess/custom/custom_pdb_dict_AgAb.pickle  : 抗原66维 + 抗体46维特征
  - dataProcess/custom/custom_cdr_dict.pickle        : 抗体 CDR 标注
  - dataProcess/custom/custom_catsite_dict.pickle    : VH/VL 切割位点

运行:
  conda run -n episcan python scripts/build_features_from_pdb.py
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ============================================================
# 配置路径
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDB_DIR = os.path.join(BASE_DIR, "dataProcess/custom/pdb")
TSV_PATH = os.path.join(BASE_DIR, "dataProcess/custom/fasta_label4.5.tsv")
OUTPUT_PICKLE = os.path.join(BASE_DIR, "dataProcess/custom/custom_pdb_dict_AgAb.pickle")
OUTPUT_CDR = os.path.join(BASE_DIR, "dataProcess/custom/custom_cdr_dict.pickle")
OUTPUT_CATSITE = os.path.join(BASE_DIR, "dataProcess/custom/custom_catsite_dict.pickle")

# ============================================================
# 常量
# ============================================================
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}
# 非标准氨基酸映射
EXTRA_MAP = {'MSE':'M','CSO':'C','SEP':'S','TPO':'T','PTR':'Y',
             'HID':'H','HIE':'H','HIP':'H','ASH':'D','GLH':'E',
             'CYX':'C'}


def three_to_one(resname):
    """3 字母氨基酸代码 → 1 字母"""
    if resname in THREE_TO_ONE:
        return THREE_TO_ONE[resname]
    if resname in EXTRA_MAP:
        return EXTRA_MAP[resname]
    return 'X'


# ============================================================
# PDB 解析
# ============================================================
def parse_pdb_structure(pdb_path):
    """解析 PDB 文件，按链提取 CA 原子坐标和残基序列"""
    chains = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                chain_id = line[21].strip()
                if not chain_id:
                    chain_id = " "
                resname = line[17:20].strip()
                resid = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                if chain_id not in chains:
                    chains[chain_id] = {"seq_3": [], "coords": [], "resids": []}
                chains[chain_id]["seq_3"].append(resname)
                chains[chain_id]["coords"].append([x, y, z])
                chains[chain_id]["resids"].append(resid)

    for ch in chains:
        chains[ch]["coords"] = np.array(chains[ch]["coords"], dtype=np.float64)
        chains[ch]["seq_1"] = "".join(three_to_one(r) for r in chains[ch]["seq_3"])
    return chains


def find_chain(chains, target_chain_id):
    """在 chains 字典中查找链 ID（不区分大小写）"""
    if target_chain_id in chains:
        return target_chain_id, chains[target_chain_id]
    # 尝试大小写不敏感
    for cid in chains:
        if cid.upper() == target_chain_id.upper():
            return cid, chains[cid]
    # 尝试去掉末尾数字
    target_base = target_chain_id.rstrip("0123456789")
    for cid in chains:
        if cid.upper() == target_base.upper():
            return cid, chains[cid]
    return None, None


# ============================================================
# 特征计算函数
# ============================================================
def compute_local_frequency_profile(coords, seq, radius=8.0):
    """计算 8Å 半径内氨基酸频率谱 (L x 20)"""
    n_res = len(seq)
    profile = np.zeros((n_res, 20), dtype=np.float64)
    if n_res == 0:
        return profile

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    for i in range(n_res):
        mask = (dist[i] <= radius) & (dist[i] > 0)
        neighbor_idx = np.where(mask)[0]
        for j in neighbor_idx:
            aa = seq[j]
            if aa in AA_TO_INT:
                profile[i, AA_TO_INT[aa]] += 1.0
        row_sum = profile[i].sum()
        if row_sum > 0:
            profile[i] /= row_sum
    return profile


def compute_one_hot(seq):
    """氨基酸 one-hot 编码 (L x 20)"""
    n = len(seq)
    one_hot = np.zeros((n, 20), dtype=np.float64)
    for i, aa in enumerate(seq):
        if aa in AA_TO_INT:
            one_hot[i, AA_TO_INT[aa]] = 1.0
    return one_hot


def compute_sasa_freesasa(pdb_path, chain_id):
    """用 FreeSASA 计算溶剂可及表面积"""
    import freesasa
    try:
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)
        sasa_data = result.residueAreas()
        # FreeSASA v2: {chain_id: {resnum: ResidueArea, ...}, ...}
        # FreeSASA v1: {(chain, resid, inscode): ResidueArea, ...}
        chain_sasa = []
        # Try v2 format first
        if isinstance(list(sasa_data.values())[0], dict):
            # v2: chain -> residue dict
            if chain_id in sasa_data:
                res_areas = sasa_data[chain_id]
                for resnum in sorted(res_areas.keys(), key=lambda x: int(str(x).rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))):
                    res_area = res_areas[resnum]
                    abs_sasa = res_area.totalArea
                    rel_sasa = min(abs_sasa / 150.0, 1.5)
                    chain_sasa.append([abs_sasa, rel_sasa])
        else:
            # v1: (chain, resid, inscode) tuples
            for res_key in sorted(sasa_data.keys(), key=lambda k: (k[0], k[1])):
                if isinstance(res_key, tuple) and len(res_key) == 3:
                    ch = str(res_key[0])
                    if ch == chain_id:
                        res_area = sasa_data[res_key]
                        abs_sasa = res_area.totalArea
                        rel_sasa = min(abs_sasa / 150.0, 1.5)
                        chain_sasa.append([abs_sasa, rel_sasa])
        if chain_sasa:
            return np.array(chain_sasa, dtype=np.float64)
    except Exception as e:
        sys.stderr.write(f"  FreeSASA warning for {os.path.basename(pdb_path)}:{chain_id} - {e}\n")
    return None


# ============================================================
# 构建抗原 66 维特征
# ============================================================
def build_antigen_features(chains, antigen_id, pdb_path):
    """为单个抗原构建 66 维特征 (L, 66)"""
    chain_id = antigen_id.split("_")[-1]
    cid, ch_data = find_chain(chains, chain_id)
    if ch_data is None:
        print(f"  ⚠ 抗原 {antigen_id}: 链 '{chain_id}' 未找到, 可用链: {list(chains.keys())}")
        return None

    seq_len = len(ch_data["seq_1"])
    seq = ch_data["seq_1"]
    coords = ch_data["coords"]

    # 确保序列长度与坐标一致
    min_len = min(seq_len, len(coords))

    # 1) 3D 坐标 [0:3]
    coords_feat = coords[:min_len]

    # 2) One-hot [3:23]
    one_hot = compute_one_hot(seq[:min_len])

    # 3) PSSM [23:44] → 21维（20维 PSSM + 1维 bias 列）
    #    README: "Indices [23-43] represent a conservation profile (The initial column with a bias value of zero)"
    pssm_20 = np.zeros((min_len, 20), dtype=np.float64)
    bias_col = np.zeros((min_len, 1), dtype=np.float64)
    pssm = np.concatenate([pssm_20, bias_col], axis=1)  # (L, 21)

    # 4) 局部氨基酸频率谱 [44:64]
    local_freq = compute_local_frequency_profile(
        coords[:min_len], seq[:min_len], radius=8.0
    )

    # 5) SASA [64:66]
    sasa = compute_sasa_freesasa(pdb_path, cid)
    if sasa is None or len(sasa) < min_len:
        sasa_pad = np.zeros((min_len, 2), dtype=np.float64)
        if sasa is not None:
            n = min(min_len, len(sasa))
            sasa_pad[:n] = sasa[:n]
        sasa = sasa_pad
    else:
        sasa = sasa[:min_len]

    # 拼接 66 维: 3 + 20 + 21 + 20 + 2 = 66
    feature = np.concatenate([
        coords_feat,    # [0:3]     = 3
        one_hot,       # [3:23]    = 20
        pssm,          # [23:44]   = 21  (20 PSSM + 1 bias)
        local_freq,    # [44:64]   = 20
        sasa,          # [64:66]   = 2
    ], axis=1)

    return feature


# ============================================================
# 构建抗体 46 维特征
# ============================================================
def build_antibody_features(chains, antibody_id, pdb_path):
    """为单个抗体构建 46 维特征"""
    chain_id = antibody_id.split("_")[-1]
    cid, ch_data = find_chain(chains, chain_id)
    if ch_data is None:
        print(f"  ⚠ 抗体 {antibody_id}: 链 '{chain_id}' 未找到, 可用链: {list(chains.keys())}")
        return None

    seq_len = len(ch_data["seq_1"])
    seq = ch_data["seq_1"]
    coords = ch_data["coords"]

    min_len = min(seq_len, len(coords))

    # 20维 one-hot
    one_hot = compute_one_hot(seq[:min_len])

    # 20维 局部频率
    local_freq = compute_local_frequency_profile(
        coords[:min_len], seq[:min_len], radius=8.0
    )

    # 3D 坐标中心化
    centered = coords[:min_len] - coords[:min_len].mean(axis=0)

    # 3维 SASA
    sasa = compute_sasa_freesasa(pdb_path, cid)
    if sasa is None or len(sasa) < min_len:
        sasa_pad = np.zeros((min_len, 2), dtype=np.float64)
        if sasa is not None:
            n = min(min_len, len(sasa))
            sasa_pad[:n] = sasa[:n]
        sasa = sasa_pad
    else:
        sasa = sasa[:min_len]

    # 拼接：20 + 20 + 3 + 2 = 45，再加 1 维零补齐到 46
    feat_45 = np.concatenate([one_hot, local_freq, centered, sasa], axis=1)
    pad = np.zeros((min_len, 1), dtype=np.float64)
    feat_46 = np.concatenate([feat_45, pad], axis=1)

    # 保证恰好 46 维
    if feat_46.shape[1] > 46:
        feat_46 = feat_46[:, :46]
    elif feat_46.shape[1] < 46:
        pad2 = np.zeros((min_len, 46 - feat_46.shape[1]))
        feat_46 = np.concatenate([feat_46, pad2], axis=1)

    return feat_46


# ============================================================
# ANARCI 抗体标注
# ============================================================
def annotate_antibody(pdb_path, antibody_id):
    """用 ANARCI 标注抗体 VH/VL 分界和 CDR 位置"""
    from anarci import anarci

    chain_id = antibody_id.split("_")[-1]

    # 从 PDB 读取该链序列
    sequence = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ch = line[21].strip()
                if ch == chain_id:
                    resname = line[17:20].strip()
                    one = three_to_one(resname)
                    if one != 'X' and (not sequence or sequence[-1][0] != line[22:26].strip()):
                        sequence.append((line[22:26].strip(), one))

    if not sequence:
        return None, None

    seq_str = "".join(s[1] for s in sequence)

    try:
        result = anarci([("query", seq_str)], scheme="kabat")
        if result[0] is not None:
            numbered = result[0][0]
            cdr_positions = set()
            vh_end = 0

            for domain_data in numbered:
                domain_info, align_info, _ = domain_data
                domain_type = domain_info[0]  # 'H', 'K', 'L'

                for seq_idx, kabat_num in align_info:
                    # 提取数字部分
                    try:
                        num = int(float(str(kabat_num)))
                    except (ValueError, TypeError):
                        continue

                    # CDR 定义 (Kabat)
                    if domain_type == 'H':
                        if (31 <= num <= 35) or (50 <= num <= 65) or (95 <= num <= 102):
                            cdr_positions.add(seq_idx)
                        vh_end = max(vh_end, seq_idx + 1)
                    elif domain_type in ('K', 'L'):
                        if (24 <= num <= 34) or (50 <= num <= 56) or (89 <= num <= 97):
                            cdr_positions.add(seq_idx)

            if vh_end == 0:
                vh_end = len(seq_str)

            cdr_list = [1 if i in cdr_positions else 0 for i in range(len(seq_str))]
            return vh_end, cdr_list
    except Exception as e:
        print(f"  ⚠ ANARCI error for {antibody_id}: {e}")

    return None, None


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("  EpiScan Custom 数据特征构建 (PDB → Pickle)")
    print("=" * 70)

    # 读取数据
    df = pd.read_csv(TSV_PATH, sep='\t')
    print(f"\n📄 读取标注数据: {len(df)} 条记录")

    all_antigens = df[['antigen_id', 'antigen_seq']].drop_duplicates()
    all_antibodies = df[['antibody_id', 'antibody_seq']].drop_duplicates()
    print(f"   独特抗原: {len(all_antigens)}, 独特抗体: {len(all_antibodies)}")

    # 收集所有 unique PDB 代码
    all_pdb_codes = set()
    for aid in list(all_antigens['antigen_id']) + list(all_antibodies['antibody_id']):
        all_pdb_codes.add(aid.split("_")[0].lower())
    print(f"   需要解析 PDB 文件: {len(all_pdb_codes)}")

    # 初始化字典
    encoding_dict = {}
    cdr_dict = {}
    catsite_dict = {}

    # 统计
    stats = {
        'pdb_found': 0, 'pdb_not_found': 0,
        'antigen_ok': 0, 'antigen_fail': 0,
        'antibody_ok': 0, 'antibody_fail': 0,
        'cdr_anarci': 0, 'cdr_fallback': 0, 'cdr_fail': 0,
        'catsite_ok': 0, 'catsite_fallback': 0,
    }

    # 预计算 (antigen_id, antibody_id) 到 PDB 代码的映射
    antigen_pdb_map = {}
    for aid in all_antigens['antigen_id']:
        antigen_pdb_map[aid] = aid.split("_")[0].lower()
    antibody_pdb_map = {}
    for aid in all_antibodies['antibody_id']:
        antibody_pdb_map[aid] = aid.split("_")[0].lower()

    # 处理所有 unique PDB
    processed_pdb = set()
    for pdb_code in sorted(all_pdb_codes):
        pdb_path = os.path.join(PDB_DIR, f"{pdb_code}.pdb")
        if not os.path.exists(pdb_path):
            stats['pdb_not_found'] += 1
            continue
        stats['pdb_found'] += 1

        # 解析 PDB
        chains = parse_pdb_structure(pdb_path)
        processed_pdb.add(pdb_code)

        # --- 处理该 PDB 内的抗原 ---
        for antigen_id in all_antigens['antigen_id']:
            if antigen_pdb_map.get(antigen_id) != pdb_code:
                continue
            if antigen_id in encoding_dict:
                continue  # 已处理
            feat = build_antigen_features(chains, antigen_id, pdb_path)
            if feat is not None:
                encoding_dict[antigen_id] = feat
                stats['antigen_ok'] += 1
            else:
                stats['antigen_fail'] += 1

        # --- 处理该 PDB 内的抗体 ---
        for antibody_id in all_antibodies['antibody_id']:
            if antibody_pdb_map.get(antibody_id) != pdb_code:
                continue

            # 抗体特征
            if antibody_id not in encoding_dict:
                feat = build_antibody_features(chains, antibody_id, pdb_path)
                if feat is not None:
                    encoding_dict[antibody_id] = feat
                    stats['antibody_ok'] += 1
                else:
                    stats['antibody_fail'] += 1

            # CDR + Catsite (用 ANARCI)
            if antibody_id not in cdr_dict:
                vh_end, cdr = annotate_antibody(pdb_path, antibody_id)
                if vh_end is not None and cdr is not None:
                    cdr_dict[antibody_id] = cdr
                    catsite_dict[antibody_id] = vh_end
                    stats['cdr_anarci'] += 1
                    stats['catsite_ok'] += 1
                else:
                    # 降级: 从 antibody_label 列获取 CDR
                    ab_rows = df[df['antibody_id'] == antibody_id]
                    if len(ab_rows) > 0:
                        label_str = str(ab_rows.iloc[0]['antibody_label'])
                        cdr_fb = [int(c) for c in label_str if c in '01']
                        ab_seq = str(ab_rows.iloc[0]['antibody_seq'])
                        catsite_fb = int(len(ab_seq) * 0.6)
                        # 只取与序列长度一致的部分
                        cdr_fb = cdr_fb[:len(ab_seq)]
                        if len(cdr_fb) < len(ab_seq):
                            cdr_fb += [0] * (len(ab_seq) - len(cdr_fb))
                        cdr_dict[antibody_id] = cdr_fb
                        catsite_dict[antibody_id] = catsite_fb
                        stats['cdr_fallback'] += 1
                        stats['catsite_fallback'] += 1
                        print(f"  ℹ 降级 {antibody_id}: ANARCI 失败, 使用序列标签")
                    else:
                        stats['cdr_fail'] += 1

        if len(processed_pdb) % 500 == 0:
            print(f"  进度: {len(processed_pdb)}/{len(all_pdb_codes)} PDB 已处理"
                  f" (抗原: {stats['antigen_ok']}, 抗体: {stats['antibody_ok']}, CDR: {stats['cdr_anarci']})")

    # --- 补充未从 PDB 获取到的抗原特征 (零填充) ---
    for antigen_id in all_antigens['antigen_id']:
        if antigen_id not in encoding_dict:
            seq_len = len(all_antigens[all_antigens['antigen_id'] == antigen_id].iloc[0]['antigen_seq'])
            encoding_dict[antigen_id] = np.zeros((seq_len, 66), dtype=np.float64)
            stats['antigen_fail'] += 1
            print(f"  ℹ 零填充 {antigen_id} (PDB 特征提取失败)")

    # --- 补充未从 PDB 获取到的抗体特征 (零填充) ---
    for antibody_id in all_antibodies['antibody_id']:
        if antibody_id not in encoding_dict:
            seq_len = len(all_antibodies[all_antibodies['antibody_id'] == antibody_id].iloc[0]['antibody_seq'])
            encoding_dict[antibody_id] = np.zeros((seq_len, 46), dtype=np.float64)
            stats['antibody_fail'] += 1

    # --- 补充未标注 CDR 的抗体 (从序列标签) ---
    for antibody_id in all_antibodies['antibody_id']:
        if antibody_id not in cdr_dict:
            ab_rows = df[df['antibody_id'] == antibody_id]
            if len(ab_rows) > 0:
                label_str = str(ab_rows.iloc[0]['antibody_label'])
                ab_seq = str(ab_rows.iloc[0]['antibody_seq'])
                cdr_fb = [int(c) for c in label_str if c in '01']
                cdr_fb = cdr_fb[:len(ab_seq)]
                if len(cdr_fb) < len(ab_seq):
                    cdr_fb += [0] * (len(ab_seq) - len(cdr_fb))
                cdr_dict[antibody_id] = cdr_fb
                catsite_dict[antibody_id] = int(len(ab_seq) * 0.6)
                stats['cdr_fallback'] += 1
                print(f"  ℹ 补充 CDR {antibody_id} (从 TSV 标签)")

    # ============================================================
    # 报告
    # ============================================================
    print("\n" + "=" * 70)
    print("  处理汇总")
    print("=" * 70)
    print(f"  PDB 解析: 找到 {stats['pdb_found']}, 未找到 {stats['pdb_not_found']}")
    print(f"  抗原特征: 成功 {stats['antigen_ok']}, 失败/零填充 {stats['antigen_fail']}")
    print(f"  抗体特征: 成功 {stats['antibody_ok']}, 失败/零填充 {stats['antibody_fail']}")
    print(f"  CDR 标注: ANARCI {stats['cdr_anarci']}, 降级 {stats['cdr_fallback']}, 失败 {stats['cdr_fail']}")
    print(f"  Catsite: ANARCI {stats['catsite_ok']}, 降级 {stats['catsite_fallback']}")
    print(f"  Encoding dict keys: {len(encoding_dict)}")
    print(f"  CDR dict keys: {len(cdr_dict)}")
    print(f"  Catsite dict keys: {len(catsite_dict)}")

    # 维度检查
    print("\n  维度检查 (抗原示例):")
    ag_examples = list(all_antigens['antigen_id'])[:3]
    for aid in ag_examples:
        if aid in encoding_dict:
            print(f"    {aid}: {encoding_dict[aid].shape}")

    # ============================================================
    # 保存
    # ============================================================
    print(f"\n💾 保存: {OUTPUT_PICKLE}")
    with open(OUTPUT_PICKLE, 'wb') as f:
        pickle.dump(encoding_dict, f)

    print(f"💾 保存: {OUTPUT_CDR}")
    with open(OUTPUT_CDR, 'wb') as f:
        pickle.dump(cdr_dict, f)

    print(f"💾 保存: {OUTPUT_CATSITE}")
    with open(OUTPUT_CATSITE, 'wb') as f:
        pickle.dump(catsite_dict, f)

    # ============================================================
    # 校验
    # ============================================================
    print("\n" + "=" * 70)
    print("  校验")
    print("=" * 70)
    # 检查训练/测试 ID 都在 pickle 中
    for split_name, id_file in [
        ("训练集", BASE_DIR + "/dataProcess/custom/epitope_ratio_train_pdb_ids4645.txt"),
        ("测试集", BASE_DIR + "/dataProcess/custom/epitope_ratio_test_pdb_ids4645.txt"),
    ]:
        missing = []
        with open(id_file) as f:
            for line in f:
                parts = line.strip().split(',')
                ag = parts[0].strip()
                ab = parts[1].strip()
                if ag not in encoding_dict:
                    missing.append(ag)
                if ab not in encoding_dict:
                    missing.append(ab)
                if ab not in cdr_dict:
                    missing.append(f"{ab}_CDR")
                if ab not in catsite_dict:
                    missing.append(f"{ab}_catsite")
        if missing:
            print(f"  ⚠ {split_name}: {len(missing)} 个 ID 缺失")
        else:
            print(f"  ✅ {split_name}: 所有 ID 在 pickle 中")

    print("\n✅ 特征构建完成！")


if __name__ == "__main__":
    main()
