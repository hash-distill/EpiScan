"""从 5 折训练日志中提取每折最佳指标并计算均值 ± 标准差"""
import re, os, sys, glob

def parse_log(log_file):
    metrics = []
    pat = re.compile(
        r'Finished Epoch (\d+)/\d+: '
        r'Loss=([\d.]+), Accuracy=([\d.%]+), MSE=([\d.]+), '
        r'Precision=([\d.]+), Recall=([\d.]+), F1=([\d.]+), '
        r'(?:MCC=([\d.\-]+), )?AUPR=([\d.]+), AUC=([\d.]+), DICE=([\d.]+)'
    )
    for line in open(log_file):
        m = pat.search(line)
        if m:
            metrics.append({
                'epoch': int(m.group(1)),
                'precision': float(m.group(5)),
                'recall': float(m.group(6)),
                'f1': float(m.group(7)),
                'mcc': float(m.group(8)) if m.group(8) else float('nan'),
                'aupr': float(m.group(9)),
                'auc': float(m.group(10)),
            })
    return metrics

prefix = sys.argv[1] if len(sys.argv) > 1 else 'DB1_cv_fold'
results = []
for fold in range(5):
    log_file = os.path.join('output', f'{prefix}{fold}.log')
    if not os.path.exists(log_file):
        print(f'Fold {fold}: 无日志 {log_file}')
        continue
    metrics = parse_log(log_file)
    if not metrics:
        print(f'Fold {fold}: 未解析到指标')
        continue
    best_auc = max(metrics, key=lambda x: x['auc'])
    best_aupr = max(metrics, key=lambda x: x['aupr'])
    best_f1 = max(metrics, key=lambda x: x['f1'])
    results.append({
        'fold': fold,
        'best_auc': best_auc['auc'],
        'best_auc_epoch': best_auc['epoch'],
        'best_auc_aupr': best_auc['aupr'],
        'best_auc_f1': best_auc['f1'],
        'best_auc_prec': best_auc['precision'],
        'best_auc_recall': best_auc['recall'],
        'best_aupr': best_aupr['aupr'],
        'best_aupr_epoch': best_aupr['epoch'],
        'best_f1': best_f1['f1'],
        'best_f1_epoch': best_f1['epoch'],
    })
    print(f"Fold {fold}: BestAUC={best_auc['auc']:.4f}@E{best_auc['epoch']} "
          f"(AUPR={best_auc['aupr']:.4f}, F1={best_auc['f1']:.4f}, "
          f"Prec={best_auc['precision']:.4f}, Rec={best_auc['recall']:.4f})")

if results:
    n = len(results)
    print('\n' + '=' * 60)
    print(f'  {prefix} {n}-Fold CV Summary (mean ± std)')
    print('=' * 60)
    for key, name in [('best_auc', 'AUC'), ('best_auc_aupr', 'AUPR'),
                      ('best_auc_f1', 'F1'), ('best_auc_prec', 'Precision'),
                      ('best_auc_recall', 'Recall')]:
        vals = [r[key] for r in results]
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
        print(f'  {name:10s} = {mean:.4f} ± {std:.4f}')
