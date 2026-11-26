#!/usr/bin/env python
"""Ensemble the five cross-validation folds on the held-out test set.

Two strategies are computed:
  - logits averaging: average the raw logits over folds, then softmax and argmax
  - probability averaging: softmax each fold first, then average and argmax

Reported results use logits averaging throughout.

For every fold the script reads
  - checkpoints/*.pth, whose filename encodes the best-validation epoch
  - draw_data/test_score.npy, the per-epoch logits
  - draw_data/test_y_true.npy, the per-epoch labels
All folds share the same test set, so predictions can be averaged sample by
sample.

Run from the repository root:
  python vote.py --base_dir <output directory>
Without --base_dir the most recent output directory is used."""
import os
import re
import glob
import json
import argparse
import numpy as np
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef)


def parse_checkpoint_filename(filepath):
    """Parse (epoch, val_mcc, test_mcc) out of a checkpoint filename. Epochs are 1-based."""
    filename = os.path.basename(filepath)
    pattern = r'_epoch(\d+)_mcc([\d.]+)_([\d.]+)\.pth'
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f'cannot parse checkpoint filename: {filename}')
    return int(match.group(1)), float(match.group(2)), float(match.group(3))


def find_checkpoint(fold_dir):
    """Locate the checkpoint of a fold; only the best-validation one is retained."""
    ckpt_dir = os.path.join(fold_dir, 'checkpoints')
    pths = glob.glob(os.path.join(ckpt_dir, '*.pth'))
    return pths[0] if pths else None


def load_fold_data(fold_dir, epoch, mode='test'):
    """Read (y_true, logits) at the best-validation epoch of a fold."""
    dd = os.path.join(fold_dir, 'draw_data')
    y_true_file = os.path.join(dd, f'{mode}_y_true.npy')
    score_file = os.path.join(dd, f'{mode}_score.npy')
    if not os.path.exists(y_true_file) or not os.path.exists(score_file):
        raise FileNotFoundError(f'missing data file: {y_true_file} or {score_file}')
    y_true_all = np.load(y_true_file)
    scores_all = np.load(score_file)
    idx = epoch - 1
    if idx < 0 or idx >= y_true_all.shape[0]:
        raise IndexError(f'epoch out of range: epoch={epoch}, n_epochs={y_true_all.shape[0]}')
    return y_true_all[idx], scores_all[idx]


def compute_metrics(y_true, y_pred):
    return {
        'acc': accuracy_score(y_true, y_pred),
        'pre': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }


def ensemble_voting(fold_data_list, strategy='logits'):
    """Average predictions over folds. strategy is 'logits' or 'probs'."""
    y_true = fold_data_list[0][0]
    if strategy == 'logits':
        all_scores = np.stack([s for _, s in fold_data_list], axis=0)
        avg_scores = np.mean(all_scores, axis=0)
        avg_probs = softmax(avg_scores, axis=1)
        y_pred = np.argmax(avg_probs, axis=1)
    elif strategy == 'probs':
        all_probs = np.stack([softmax(s, axis=1) for _, s in fold_data_list], axis=0)
        avg_probs = np.mean(all_probs, axis=0)
        y_pred = np.argmax(avg_probs, axis=1)
    else:
        raise ValueError(f'unknown strategy: {strategy}')
    return y_true, y_pred, avg_probs


def collect_folds(base_dir, mode='test'):
    """Collect (y_true, logits) and metadata for folds 1..5 under base_dir."""
    fold_data_list, fold_info_list = [], []
    for i in range(1, 6):
        fold_dir = os.path.join(base_dir, str(i))
        if not os.path.isdir(fold_dir):
            continue
        ckpt = find_checkpoint(fold_dir)
        if ckpt is None:
            print(f'  [warning] fold {i}: no checkpoint found')
            continue
        try:
            epoch, val_mcc, test_mcc = parse_checkpoint_filename(ckpt)
            y_true, scores = load_fold_data(fold_dir, epoch, mode)
            fold_data_list.append((y_true, scores))
            fold_info_list.append({'fold': i, 'epoch': epoch,
                                   'val_mcc': val_mcc, 'test_mcc': test_mcc})
        except Exception as e:
            print(f'  [error] fold {i}: {e}')
    return fold_data_list, fold_info_list


def latest_base_dir(name='duet_m7g_5fold'):
    root = os.path.join('models_log', name)
    if not os.path.isdir(root):
        return None
    subs = [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]
    if not subs:
        return None
    return max(subs, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description='Five-fold ensemble evaluation on the m7G benchmark')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Root directory of a five-fold run (holding subdirectories 1..5); defaults to the most recent one.')
    parser.add_argument('--mode', type=str, default='test', choices=['test'],
                        help='Evaluation set: test (the hold-out shared by all folds).')
    args = parser.parse_args()

    base_dir = args.base_dir or latest_base_dir()
    if base_dir is None or not os.path.isdir(base_dir):
        raise SystemExit(f'[error] base_dir does not exist: {base_dir}')

    print(f'base directory: {base_dir}\nevaluation set: {args.mode}\n')

    fold_data_list, fold_info_list = collect_folds(base_dir, args.mode)
    if not fold_data_list:
        raise SystemExit('[error] no valid fold data was collected')

    for i in range(1, len(fold_data_list)):
        if not np.array_equal(fold_data_list[0][0], fold_data_list[i][0]):
            print('[warning] y_true differs across folds; taking fold 1 as reference')
            break

    print(f"[folds] {len(fold_data_list)} collected")
    print('-' * 60)
    for info in fold_info_list:
        print(f"  Fold {info['fold']}: best-ep{info['epoch']:>3}  "
              f"val_mcc={info['val_mcc']:.4f}  test_mcc={info['test_mcc']:.4f}")

    single_test_mccs = [info['test_mcc'] for info in fold_info_list]
    single_val_mccs = [info['val_mcc'] for info in fold_info_list]
    print('-' * 60)
    print(f"  single-model test MCC per fold: {[f'{m:.4f}' for m in single_test_mccs]}")
    print(f"  single-model test MCC, mean±std: "
          f"{np.mean(single_test_mccs):.4f} ± {np.std(single_test_mccs):.4f}")
    print(f"  single-model validation MCC, mean±std: "
          f"{np.mean(single_val_mccs):.4f} ± {np.std(single_val_mccs):.4f}")

    y_true, y_pred_l, _ = ensemble_voting(fold_data_list, 'logits')
    m_logits = compute_metrics(y_true, y_pred_l)
    _, y_pred_p, _ = ensemble_voting(fold_data_list, 'probs')
    m_probs = compute_metrics(y_true, y_pred_p)

    def fmt(m):
        return (f"ACC={m['acc']:.4f} | PRE={m['pre']:.4f} | Recall={m['recall']:.4f} | "
                f"F1={m['f1']:.4f} | MCC={m['mcc']:.4f}")

    print(f"\n[ensemble: logits averaging]\n" + '-' * 60 + f"\n  {fmt(m_logits)}")
    print(f"\n[diagnostic, not reported: probability averaging]\n" + '-' * 60 + f"\n  {fmt(m_probs)}")

    out = {
        'base_dir': base_dir,
        'mode': args.mode,
        'n_folds': len(fold_data_list),
        'fold_info': fold_info_list,
        'single_test_mcc_mean': float(np.mean(single_test_mccs)),
        'single_test_mcc_std': float(np.std(single_test_mccs)),
        'ensemble_logits': m_logits,
        'ensemble_probs': m_probs,
    }
    out_path = os.path.join(base_dir, f'vote_result_{args.mode}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
    print(f'\nensemble result written to {out_path}')


if __name__ == '__main__':
    main()
