"""
汇总 5 折交叉验证结果

用法: conda run -n episcan python scripts/summarize_cv.py --results-dir ./output/ --prefix custom_fold --out ./output/cv_summary.md
"""
import argparse, re, os, sys
import numpy as np
import pandas as pd


def parse_log(log_file):
    """从训练日志中提取每个 epoch 的验证集指标"""
    metrics = []
    epoch_pattern = re.compile(
        r'Finished Epoch (\d+)/\d+: '
        r'Loss=([\d.]+), Accuracy=([\d.%]+), MSE=([\d.]+), '
        r'Precision=([\d.]+), Recall=([\d.]+), F1=([\d.]+), '
        r'MCC=([\d.\-]+), AUPR=([\d.]+), AUC=([\d.]+), DICE=([\d.]+)'
    )
    if not os.path.exists(log_file):
        print(f"  ⚠ 文件不存在: {log_file}")
        return metrics

    with open(log_file) as f:
        for line in f:
            m = epoch_pattern.search(line)
            if m:
                metrics.append({
                    'epoch': int(m.group(1)),
                    'loss': float(m.group(2)),
                    'accuracy': float(m.group(3).rstrip('%')),
                    'mse': float(m.group(4)),
                    'precision': float(m.group(5)),
                    'recall': float(m.group(6)),
                    'f1': float(m.group(7)),
                    'mcc': float(m.group(8)),
                    'aupr': float(m.group(9)),
                    'auc': float(m.group(10)),
                    'dice': float(m.group(11)),
                })
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Summarize 5-fold CV results')
    parser.add_argument('--results-dir', default='./output/', help='Directory with training logs')
    parser.add_argument('--prefix', default='custom_fold', help='Log file prefix')
    parser.add_argument('--out', default='./output/cv_summary.csv', help='Output summary CSV')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds')
    args = parser.parse_args()

    results_dir = args.results_dir
    prefix = args.prefix
    n_folds = args.n_folds

    all_best = []
    for fold in range(n_folds):
        log_file = os.path.join(results_dir, f'{prefix}{fold}.log')
        metrics = parse_log(log_file)
        if not metrics:
            print(f'  Fold {fold}: 无数据')
            continue

        best_auc = max(metrics, key=lambda x: x['auc'])
        best_aupr = max(metrics, key=lambda x: x['aupr'])
        best_f1 = max(metrics, key=lambda x: x['f1'])
        final = metrics[-1]

        best_mcc = max(metrics, key=lambda m: m['mcc'])
        all_best.append({
            'fold': fold,
            'best_auc_epoch': best_auc['epoch'],
            'best_auc': best_auc['auc'],
            'best_auc_aupr': best_auc['aupr'],
            'best_auc_f1': best_auc['f1'],
            'best_auc_mcc': best_auc['mcc'],
            'best_aupr': best_aupr['aupr'],
            'best_aupr_epoch': best_aupr['epoch'],
            'best_f1': best_f1['f1'],
            'best_f1_epoch': best_f1['epoch'],
            'best_mcc': best_mcc['mcc'],
            'best_mcc_epoch': best_mcc['epoch'],
            'final_auc': final['auc'],
            'final_aupr': final['aupr'],
            'final_mcc': final['mcc'],
        })
        print(f'  Fold {fold}: Best AUC={best_auc["auc"]:.4f} (E{best_auc["epoch"]}), '
              f'Best AUPR={best_aupr["aupr"]:.4f} (E{best_aupr["epoch"]})')

    if not all_best:
        print('没有找到任何结果，请检查日志文件路径')
        return

    df = pd.DataFrame(all_best)
    csv_out = args.out if args.out.endswith('.csv') else args.out + '.csv'
    df.to_csv(csv_out, index=False)
    print(f'\n📄 逐折结果已保存: {csv_out}')

    # 计算平均值
    print('\n' + '=' * 60)
    print('  5-Fold CV Summary (mean ± std)')
    print('=' * 60)
    metrics_names = [
        ('best_auc', 'AUC'),
        ('best_aupr', 'AUPR'),
        ('best_f1', 'F1'),
        ('best_mcc', 'MCC'),
    ]
    for col, name in metrics_names:
        mean = df[col].mean()
        std = df[col].std()
        print(f'  {name:8s} = {mean:.4f} ± {std:.4f}')

    # 准备 Markdown 输出
    md_out = csv_out.replace('.csv', '.md')
    total_epochs_all = []

    with open(md_out, 'w') as f:
        f.write(f'# {prefix} 5-Fold Cross Validation Results\n\n')
        f.write('| Fold | Best AUC | Epoch | Best AUPR | Epoch | Best F1 | Epoch | Best MCC | Epoch | Final AUC |\n')
        f.write('|------|----------|-------|-----------|-------|---------|-------|----------|-------|-----------|\n')
        for _, row in df.iterrows():
            f.write(f"| {int(row['fold'])} | {row['best_auc']:.4f} | {int(row['best_auc_epoch'])} | "
                    f"{row['best_aupr']:.4f} | {int(row['best_aupr_epoch'])} | "
                    f"{row['best_f1']:.4f} | {int(row['best_f1_epoch'])} | "
                    f"{row['best_mcc']:.4f} | {int(row['best_mcc_epoch'])} | "
                    f"{row['final_auc']:.4f} |\n")

        f.write('|------|----------|-------|-----------|-------|---------|-------|----------|-------|-----------|\n')
        f.write(f"| **Mean** | **{df['best_auc'].mean():.4f}** | | **{df['best_aupr'].mean():.4f}** | | "
                f"**{df['best_f1'].mean():.4f}** | | **{df['best_mcc'].mean():.4f}** | | **{df['final_auc'].mean():.4f}** |\n")
        f.write(f"| **Std** | **{df['best_auc'].std():.4f}** | | **{df['best_aupr'].std():.4f}** | | "
                f"**{df['best_f1'].std():.4f}** | | **{df['best_mcc'].std():.4f}** | | **{df['final_auc'].std():.4f}** |\n")

    print(f'📄 Markdown 汇总已保存: {md_out}')
    print('=' * 60)


if __name__ == '__main__':
    main()
