"""Fix SASA values in custom_pdb_dict_AgAb.pickle (FreeSASA v2 API fix)"""
import os, pickle, sys, warnings
import numpy as np

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDB_DIR = os.path.join(BASE, "dataProcess/custom/pdb")
PICKLE_PATH = os.path.join(BASE, "dataProcess/custom/custom_pdb_dict_AgAb.pickle")

# Inline SASA function (FreeSASA v2 compatible)
def compute_sasa(pdb_path, chain_id):
    import freesasa
    try:
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)
        sasa_data = result.residueAreas()
        chain_sasa = []
        # FreeSASA v2: {chain_id: {resnum: ResidueArea}}
        # FreeSASA v1: {(chain, resid, inscode): ResidueArea}
        first_val = list(sasa_data.values())[0]
        if isinstance(first_val, dict):
            # v2 format
            if chain_id in sasa_data:
                for resnum in sorted(sasa_data[chain_id].keys()):
                    ra = sasa_data[chain_id][resnum]
                    abs_sasa = ra.total
                    rel_sasa = ra.relativeTotal if ra.hasRelativeAreas else min(abs_sasa / 150.0, 1.5)
                    chain_sasa.append([abs_sasa, rel_sasa])
        else:
            # v1 format
            for res_key in sorted(sasa_data.keys()):
                if isinstance(res_key, tuple) and len(res_key) == 3:
                    if str(res_key[0]) == chain_id:
                        ra = sasa_data[res_key]
                        abs_sasa = ra.totalArea
                        rel_sasa = min(abs_sasa / 150.0, 1.5)
                        chain_sasa.append([abs_sasa, rel_sasa])
        if chain_sasa:
            return np.array(chain_sasa, dtype=np.float64)
    except Exception as e:
        sys.stderr.write(f"  SASA error: {os.path.basename(pdb_path)}:{chain_id} - {e}\n")
    return None

print("=" * 60)
print("Fix SASA in custom_pdb_dict_AgAb.pickle")
print("=" * 60)

with open(PICKLE_PATH, 'rb') as f:
    data = pickle.load(f)
print(f"Loaded {len(data)} keys")

ag_keys = [k for k, v in data.items() if v.shape[1] == 66]
print(f"Antigens (66-dim): {len(ag_keys)}")

pdb_codes = set(k.split("_")[0].lower() for k in ag_keys)
print(f"Unique PDBs: {len(pdb_codes)}")

fixed = 0
missing = 0
pdb_cache = {}

for i, pdb_code in enumerate(sorted(pdb_codes)):
    pdb_path = os.path.join(PDB_DIR, f"{pdb_code}.pdb")
    if not os.path.exists(pdb_path):
        missing += 1
        continue

    ags = [k for k in ag_keys if k.lower().startswith(pdb_code)]
    for ag_key in ags:
        chain_id = ag_key.split("_")[-1]
        seq_len = len(data[ag_key])
        old_sasa = data[ag_key][:, 64:66]
        if old_sasa.sum() > 0:
            continue  # already has SASA

        sasa = compute_sasa(pdb_path, chain_id)
        if sasa is not None:
            n = min(seq_len, len(sasa))
            data[ag_key][:n, 64:66] = sasa[:n]
            fixed += 1

    if (i + 1) % 300 == 0:
        sys.stderr.write(f"  Progress: {i+1}/{len(pdb_codes)} PDBs, {fixed} fixed\n")
        sys.stderr.flush()

print(f"\nFixed: {fixed} antigens, Missing PDBs: {missing}")

# Verify
sasa_nz = sum(1 for k in ag_keys if data[k][:, 64:66].sum() > 0)
print(f"SASA non-zero: {sasa_nz}/{len(ag_keys)}")
if sasa_nz > 0:
    sample = [k for k in ag_keys if data[k][:, 64:66].sum() > 0][0]
    print(f"  Example {sample}: SASA={data[sample][0, 64:66]}")

print(f"\nSaving to {PICKLE_PATH}...")
with open(PICKLE_PATH, 'wb') as f:
    pickle.dump(data, f)
print("Done!")
