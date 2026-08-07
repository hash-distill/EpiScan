---
name: episcan-reproduction-plan
description: EpiScan reproduction — custom dataset 5-fold CV fully working after bug fixes
metadata:
  type: project
---

EpiScan 复现两大任务：DB1 基准 + 自定义数据集 5 折 CV。

**核心结构**：项目根 `/mnt/Data4/24zhs/EpiScan/`，包在 `EpiScan/EpiScan/`，PYTHONPATH=`/mnt/Data4/24zhs/EpiScan/EpiScan`，conda 环境 `episcan`。

**Custom 数据（已全部预处理完成）**：
- `custom_all_proteins.h5`（30GB，**训练必须用这个**，含全部 9482 蛋白）⭐
- `custom_pdb_dict_AgAb.pickle`、`custom_cdr_dict.pickle`、`custom_catsite_dict.pickle`
- `custom_train.tsv`(1368) / `custom_test.tsv`(156) / `custom_fold{0-4}_{train,val}.tsv`

**调试排掉的 bug（custom 数据训练已跑通，AUC≈0.60@1epoch）**：
1. H5 必须含全部蛋白（否则 KeyError）
2. 移除 libauc `CompositionalAUCLoss`（与 PyTorch 2.5 不兼容）
3. `contact_sep.py` CDR 索引 off-by-one + clamp + 空索引填 `[0]`（否则 gather 越界 / NaN）
4. `interaction_eval` 用 `torch.cat` 拼标签（非 `stack`）
5. PDB/TSV 序列长度不一致 → 特征长度对齐（pad/截断）
6. `torch.clamp(ph,1e-7,1-1e-7)` 保护 loss 输入

**5 折 CV 命令**：5 终端并行，`custom_fold${N}` + `--device ${N}`，汇总用 `scripts/summarize_cv.py`。

**完整文档**: [[复现任务.md]]（含最终正确命令）

**Why**: 大量 bug 修复过程复杂，需要记录最终可用状态
**How to apply**: 跑 custom 训练一定用 `custom_all_proteins.h5`，代码已修复无需再改
