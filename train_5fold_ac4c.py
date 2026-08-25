"""Five-fold cross-validation training for the ac4C transfer experiment.

Identical to the m7G entry point in architecture and training regime; only the
window length differs (201 nt instead of 501 nt). Nothing about the model is
retuned for the new modification.

Ensemble the resulting folds with:
  python vote_ac4c.py --base_dir <output directory>

Run from the repository root:
  python train_5fold_ac4c.py
To resume: --root_dir <existing output directory> --start_fold <N>"""
import json
import os
import sys
import time
import random
from datetime import datetime
import torch
import numpy as np
from torch.amp import autocast
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from logging_utils import tqdm, alog, fmt_elapsed, suppress_hf_noise, Heartbeat

from utils import checkpoint_save_val_test
from data_utils.dataset_rnafm import RNAFMDataset
from model import DuetM7G

np.set_printoptions(threshold=np.inf)
sys.path.append('..')
sys.path.append('../..')

import warnings

warnings.filterwarnings("ignore")
suppress_hf_noise()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

batch_size = 16
num_workers = 1
pin_memory = True
use_amp = True

N_epochs = 100
model_lr = 5e-5
bert_lr = 1e-5

lr_milestones = [20]
lr_gamma = 1

L2_factor = 1e-2
label_smoothing = 0.05

N_FOLDS = 5
SEED = 17
FOLD_DATA_DIR = './data_ac4c/5fold'
TEST_FASTA = './data_ac4c/test.fasta'
window_size = 201
name = 'duet_ac4c_5fold'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def apply_bert_depth(model, depth):
    """Set how many DNABERT encoder layers are left trainable.

    depth=1, the default, unfreezes the final layer of each backbone, which is
    what the branch already does on construction. depth=0 freezes the backbones
    completely, matching the protocol of the RNA-FM branch. Only requires_grad
    is touched: no tensors are created and no random numbers are drawn.
    """
    for b in (model.bert.bertone, model.bert.berttwo):
        for p in b.bert.parameters():
            p.requires_grad = False
        for layer in (b.bert.encoder.layer[-depth:] if depth > 0 else []):
            for p in layer.parameters():
                p.requires_grad = True


def train(model, device, train_loader, optimizer, logger, epoch,
          list_train_acc, list_train_loss):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    correct = 0
    hb = Heartbeat('STEP', total=len(train_loader), ep=f'{epoch}/{N_epochs}')
    for batch_idx, (tok1, tok2, rnafm_ids, label) in enumerate(tqdm(train_loader)):
        tok1 = {k: v.to(device, non_blocking=True) for k, v in tok1.items()}
        tok2 = {k: v.to(device, non_blocking=True) for k, v in tok2.items()}
        rnafm_ids = rnafm_ids.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            output = model(tok1, tok2, rnafm_ids)
            loss = criterion(output, label.to(torch.int64))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = torch.max(output.detach(), 1)
        correct += torch.sum(pred == label)
        sum_loss += loss.detach().item()
        hb.beat(batch_idx + 1)
    acc = (correct / total_num).item()
    avg_loss = sum_loss / len(train_loader)
    logger.add_scalars('acc', {'train': acc}, epoch)
    logger.add_scalars('loss', {'train': avg_loss}, epoch)
    list_train_acc.append(acc)
    list_train_loss.append(avg_loss)
    return acc


def val(mode, model, device, test_loader, logger, epoch,
        list_test_acc, list_test_loss, list_test_pre, list_test_recall,
        list_test_f1, list_test_mcc, list_test_y_pred, list_test_y_true, list_test_score):
    if mode not in ['val', 'test']:
        raise Exception('mode should be either "val" or "test"')
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    y_pred, y_true, score_list = [], [], []
    with torch.no_grad(), autocast(enabled=use_amp, device_type='cuda', dtype=torch.bfloat16):
        for tok1, tok2, rnafm_ids, label in tqdm(test_loader):
            tok1 = {k: v.to(device, non_blocking=True) for k, v in tok1.items()}
            tok2 = {k: v.to(device, non_blocking=True) for k, v in tok2.items()}
            rnafm_ids = rnafm_ids.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(tok1, tok2, rnafm_ids)
            score_list.extend(output.detach().float().cpu().numpy())
            loss = criterion(output, label.to(torch.int64))
            _, pred = torch.max(output.detach(), 1)
            y_pred.extend(pred.view(-1).detach().cpu().numpy())
            y_true.extend(label.view(-1).detach().cpu().numpy())
            correct += torch.sum(pred == label)
            test_loss += loss.detach().item()
        acc = (correct / total_num).item()
        pre = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        mcc = metrics.matthews_corrcoef(y_true, y_pred)
        avgloss = test_loss / len(test_loader)
        logger.add_scalars('loss', {mode: avgloss}, epoch)
        logger.add_scalars('acc', {mode: acc}, epoch)
        list_test_acc.append(acc)
        list_test_loss.append(avgloss)
        list_test_pre.append(pre)
        list_test_recall.append(recall)
        list_test_f1.append(f1)
        list_test_mcc.append(mcc)
        list_test_y_pred.append(y_pred)
        list_test_y_true.append(y_true)
        list_test_score.append(score_list)
    return acc, avgloss


def run_fold(fold, root_dir, test_loader, alphabet, p_dict, use_compile=True,
             bert_depth=1, patience=0):
    """Train one fold and return the test metrics at its best validation epoch."""
    alog('FOLD_START', fold=fold, total=N_FOLDS)
    set_seed(SEED)

    train_fasta = os.path.join(FOLD_DATA_DIR, f'fold_{fold}_train.fasta')
    val_fasta = os.path.join(FOLD_DATA_DIR, f'fold_{fold}_val.fasta')
    train_dataset = RNAFMDataset(train_fasta, window_size, alphabet=alphabet)
    val_dataset = RNAFMDataset(val_fasta, window_size, alphabet=alphabet)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              persistent_workers=(num_workers > 0), pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            persistent_workers=(num_workers > 0), pin_memory=pin_memory)

    model = DuetM7G(4, 6, win=window_size)
    apply_bert_depth(model, bert_depth)
    model.to(device)

    bert_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('bert.bertone.') or n.startswith('bert.berttwo.'):
            bert_params.append(p)
        else:
            other_params.append(p)
    param_groups = []
    if bert_params:
        param_groups.append({'params': bert_params, 'lr': bert_lr})
    other_gi = len(param_groups)
    param_groups.append({'params': other_params, 'lr': model_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=L2_factor)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
    alog('PARAM_GROUPS', bert_depth=bert_depth,
         bert_tensors=len(bert_params), other_tensors=len(other_params),
         trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
         bert_lr=bert_lr, other_lr=model_lr)

    run_model = torch.compile(model) if use_compile else model
    if use_compile:
        alog('COMPILE', enabled=True, note='the first batch triggers compilation and is expected to be slow')

    fold_dir = os.path.join(root_dir, str(fold))
    logdir = os.path.join(fold_dir, 'logs')
    savedir = os.path.join(fold_dir, 'checkpoints')
    draw_data_dir = os.path.join(fold_dir, 'draw_data')
    tabledir = os.path.join(fold_dir, 'tables')
    for d in (logdir, tabledir, draw_data_dir, savedir):
        os.makedirs(d, exist_ok=True)
    logger = SummaryWriter(logdir)
    with open(os.path.join(tabledir, 'parameter.json'), "w", encoding="utf-8") as jf:
        json.dump({**p_dict, 'fold': fold}, jf, ensure_ascii=False, indent=4)

    list_train_acc, list_train_loss = [], []
    list_val_acc, list_val_loss, list_val_pre, list_val_recall, list_val_f1, list_val_mcc = [], [], [], [], [], []
    list_val_y_pred, list_val_y_true, list_val_score = [], [], []
    list_test_acc, list_test_loss, list_test_pre, list_test_recall, list_test_f1, list_test_mcc = [], [], [], [], [], []
    list_test_y_pred, list_test_y_true, list_test_score = [], [], []
    lr_list = []

    for epoch in range(1, N_epochs + 1):
        alog('EPOCH', fold=fold, ep=f'{epoch}/{N_epochs}')
        _t0 = time.time()
        train(run_model, device, train_loader, optimizer, logger, epoch, list_train_acc, list_train_loss)
        val('val', run_model, device, val_loader, logger, epoch,
            list_val_acc, list_val_loss, list_val_pre, list_val_recall, list_val_f1, list_val_mcc,
            list_val_y_pred, list_val_y_true, list_val_score)
        val('test', run_model, device, test_loader, logger, epoch,
            list_test_acc, list_test_loss, list_test_pre, list_test_recall, list_test_f1, list_test_mcc,
            list_test_y_pred, list_test_y_true, list_test_score)
        _epoch_sec = time.time() - _t0
        checkpoint_save_val_test(model, savedir, epoch, list_val_mcc[-1], list_test_mcc[-1],
                                 reserved_num=1, model_name=name)
        scheduler.step()
        cur_lr = optimizer.state_dict()['param_groups'][other_gi]['lr']
        lr_list.append(cur_lr)
        logger.add_scalar('lr', cur_lr, epoch)
        bi = int(np.argmax(list_val_mcc))
        alog('EPOCH_END', fold=fold, ep=f'{epoch}/{N_epochs}', sec=f'{_epoch_sec:.1f}',
             train_acc=f'{list_train_acc[-1]:.4f}', val_mcc=f'{list_val_mcc[-1]:.4f}', test_mcc=f'{list_test_mcc[-1]:.4f}',
             best_ep=bi + 1, best_val_mcc=f'{list_val_mcc[bi]:.4f}')

        if patience and (epoch - (bi + 1)) >= patience:
            alog('EARLY_STOP', fold=fold, ep=epoch, best_ep=bi + 1, patience=patience,
                 saved_epochs=N_epochs - epoch)
            break

    df_log = pd.DataFrame({'train_acc': list_train_acc, 'train_loss': list_train_loss,
                           'val_acc': list_val_acc, 'val_loss': list_val_loss,
                           'val_pre': list_val_pre, 'val_recall': list_val_recall, 'val_f1': list_val_f1,
                           'val_mcc': list_val_mcc,
                           'test_acc': list_test_acc, 'test_loss': list_test_loss,
                           'test_pre': list_test_pre, 'test_recall': list_test_recall, 'test_f1': list_test_f1,
                           'test_mcc': list_test_mcc,
                           'lr': lr_list})
    df_log.index = df_log.index + 1
    df_log.to_csv(os.path.join(tabledir, 'table.csv'), index=True, sep=',')

    np.save(os.path.join(draw_data_dir, 'val_y_pred.npy'), np.array(list_val_y_pred))
    np.save(os.path.join(draw_data_dir, 'val_y_true.npy'), np.array(list_val_y_true))
    np.save(os.path.join(draw_data_dir, 'val_score.npy'), np.array(list_val_score))
    np.save(os.path.join(draw_data_dir, 'test_y_pred.npy'), np.array(list_test_y_pred))
    np.save(os.path.join(draw_data_dir, 'test_y_true.npy'), np.array(list_test_y_true))
    np.save(os.path.join(draw_data_dir, 'test_score.npy'), np.array(list_test_score))

    logger.close()

    bi = int(np.argmax(list_val_mcc))
    fold_result = {
        'fold': fold,
        'best_epoch': bi + 1,
        'val_mcc': float(list_val_mcc[bi]),
        'test_mcc': float(list_test_mcc[bi]),
        'test_acc': float(list_test_acc[bi]),
        'test_pre': float(list_test_pre[bi]),
        'test_recall': float(list_test_recall[bi]),
        'test_f1': float(list_test_f1[bi]),
    }
    alog('FOLD_DONE', fold=fold, best_ep=bi + 1,
         val_mcc=f"{fold_result['val_mcc']:.4f}", test_mcc=f"{fold_result['test_mcc']:.4f}",
         test_acc=f"{fold_result['test_acc']:.4f}", test_f1=f"{fold_result['test_f1']:.4f}")

    del run_model, model, optimizer, scheduler, train_loader, val_loader, train_dataset, val_dataset
    torch.cuda.empty_cache()
    return fold_result


def fold_result_from_table(fold, root_dir):
    """Recover the best-validation test metrics of an already trained fold from its per-epoch table."""
    table = os.path.join(root_dir, str(fold), 'tables', 'table.csv')
    if not os.path.exists(table):
        return None
    df = pd.read_csv(table, index_col=0)
    bi = int(df['val_mcc'].idxmax())
    row = df.loc[bi]
    return {
        'fold': fold,
        'best_epoch': bi,
        'val_mcc': float(row['val_mcc']),
        'test_mcc': float(row['test_mcc']),
        'test_acc': float(row['test_acc']),
        'test_pre': float(row['test_pre']),
        'test_recall': float(row['test_recall']),
        'test_f1': float(row['test_f1']),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ac4C five-fold training (resumable)')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Reuse an existing output directory to resume; a new one is created by default.')
    parser.add_argument('--start_fold', type=int, default=1,
                        help='Fold to start from when resuming; finished folds are skipped and their table.csv is reused for the summary.')
    parser.add_argument('--compile', type=int, default=1,
                        help='Enable torch.compile (1 = on, 0 = off, falling back to eager).')
    parser.add_argument('--bert_depth', type=int, default=1,
                        help='Number of final DNABERT encoder layers left trainable (1 = default; 0 = fully frozen, matching the RNA-FM protocol).')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early-stopping patience (0 = disabled, run the full N_epochs).')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override N_epochs (for a smoke test).')
    parser.add_argument('--end_fold', type=int, default=N_FOLDS,
                        help='Last fold to train (for a smoke test).')
    cli = parser.parse_args()

    if cli.epochs:
        N_epochs = cli.epochs
    if cli.bert_depth != 1:
        name = f'{name}_bertdepth{cli.bert_depth}'

    _run_t0 = time.time()
    torch.cuda.set_device(0)
    alog('RUN_START', task=name, seed=SEED, n_folds=N_FOLDS, start_fold=cli.start_fold,
         compile=bool(cli.compile), bert_depth=cli.bert_depth, patience=cli.patience,
         n_epochs=N_epochs)

    p_dict = {
        'seed': SEED,
        'n_folds': N_FOLDS,
        'N_epochs': N_epochs,
        'batch_size': batch_size,
        'model_lr': model_lr,
        'bert_lr': bert_lr,
        'lr_milestones': lr_milestones,
        'lr_gamma': lr_gamma,
        'L2_factor': L2_factor,
        'label_smoothing': label_smoothing,
        'pool': 'attn',
        'second_branch': 'rnafm_frozen',
        'fusion_mode': 'self',
        'window_size': window_size,
        'bert_depth': cli.bert_depth,
        'patience': cli.patience,
        'task': 'ac4C generalization (N4-acetylcytidine, 201nt, C-centered)',
        'note': 'Five-fold cross-validation training of Duet-m7G on the ac4C benchmark.',
    }

    import fm
    _m, alphabet = fm.pretrained.rna_fm_t12()
    del _m

    if cli.root_dir:
        root_dir = cli.root_dir
        cur_time = os.path.basename(root_dir.rstrip('/\\'))
        alog('RESUME', root_dir=root_dir, start_fold=cli.start_fold)
    else:
        cur_time = get_cur_time()
        root_dir = os.path.join('models_log', name, cur_time)
    os.makedirs(root_dir, exist_ok=True)

    test_dataset = RNAFMDataset(TEST_FASTA, window_size, alphabet=alphabet)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             persistent_workers=(num_workers > 0), pin_memory=pin_memory)

    for fold in range(cli.start_fold, cli.end_fold + 1):
        run_fold(fold, root_dir, test_loader, alphabet, p_dict, use_compile=bool(cli.compile),
                 bert_depth=cli.bert_depth, patience=cli.patience)

    fold_results = []
    for fold in range(1, cli.end_fold + 1):
        r = fold_result_from_table(fold, root_dir)
        if r is None:
            alog('FOLD_MISSING', fold=fold, reason='no table.csv')
            continue
        fold_results.append(r)

    def mean_std(key):
        vals = np.array([r[key] for r in fold_results], dtype=float)
        return float(vals.mean()), float(vals.std())

    summary = {'name': name, 'time': cur_time, 'n_folds': N_FOLDS, 'seed': SEED,
               'folds': fold_results}
    for key in ['test_mcc', 'test_acc', 'test_f1', 'test_pre', 'test_recall', 'val_mcc']:
        m, s = mean_std(key)
        summary[f'{key}_mean'] = m
        summary[f'{key}_std'] = s

    with open(os.path.join(root_dir, 'cv_summary.json'), 'w', encoding='utf-8') as jf:
        json.dump(summary, jf, ensure_ascii=False, indent=4)
    pd.DataFrame(fold_results).to_csv(os.path.join(root_dir, 'cv_summary.csv'), index=False)

    for r in fold_results:
        alog('FOLD_RESULT', fold=r['fold'], best_ep=r['best_epoch'],
             val_mcc=f"{r['val_mcc']:.4f}", test_mcc=f"{r['test_mcc']:.4f}",
             test_acc=f"{r['test_acc']:.4f}", test_f1=f"{r['test_f1']:.4f}")
    alog('RUN_SUMMARY', name=name,
         test_mcc=f"{summary['test_mcc_mean']:.4f}±{summary['test_mcc_std']:.4f}",
         test_acc=f"{summary['test_acc_mean']:.4f}±{summary['test_acc_std']:.4f}",
         test_f1=f"{summary['test_f1_mean']:.4f}±{summary['test_f1_std']:.4f}",
         summary_json=os.path.join(root_dir, 'cv_summary.json'))
    alog('VOTE_HINT', cmd=f'python vote_ac4c.py --base_dir {root_dir}')
    alog('RUN_END', exit=0, elapsed=fmt_elapsed(time.time() - _run_t0))
