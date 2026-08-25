"""Convert the ac4C benchmark and build its stratified five-fold partition.

The trainset.txt / testset.txt files distributed by the ac4C-AFL web server are
converted to the FASTA conventions used here:

    # Sequences 1-2206 are positive samples
    # Sequences 2207-4412 are negative samples
    >uc002yel.3_915_1734_1171
    AGAACAGCGG... (201 nt, cytidine at the centre)

Four steps, each matching an assumption of data_utils/dataset.py:
  1. Rewrite headers as >ac4c_pos_{i} / >ac4c_neg_{i}, since labels are read
     from the header text; the positive/negative boundary comes from the
     comment lines of the original file.
  2. Drop the '#' comment lines, which would otherwise be read as sequences.
  3. Map U to T, the alphabet expected by the encoders and tokenizers.
  4. Discard sequences whose length is not 201. The download contains one
     positive sequence of 200 nt; padding it would mean inventing data, so it
     is dropped instead. This affects only the training pool, not the test set.

Writes, relative to the repository root:
  data_ac4c/train_pool.fasta   pool split by the five folds
  data_ac4c/test.fasta         held-out independent test set
  data_ac4c/5fold/fold_{1..5}_{train,val}.fasta

The raw downloads are expected in data_ac4c/raw/{trainset,testset}.txt.

Run from the repository root:
  python data_utils/make_ac4c_data.py"""
import os
import collections
from sklearn.model_selection import StratifiedKFold

RAW_DIR = './data_ac4c/raw'
OUT_DIR = './data_ac4c'
FOLD_DIR = os.path.join(OUT_DIR, '5fold')
SEQ_LEN = 201
N_SPLITS = 5
SEED = 17

TRAIN_N_POS = 2206
TEST_N_POS = 552


def read_raw(path):
    """Read the raw file into [(header, sequence), ...], dropping '#' comment lines."""
    recs = []
    header = None
    with open(path, encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.rstrip('\n')
            if not ln.strip() or ln.startswith('#'):
                continue
            if ln.startswith('>'):
                header = ln[1:]
            else:
                recs.append((header, ln.strip()))
                header = None
    return recs


def convert(recs, n_pos, split_name):
    """Label by index boundary, map U to T, drop sequences that are not 201 nt."""
    out = []
    dropped = []
    pos_i = neg_i = 0
    for idx, (orig_h, seq) in enumerate(recs):
        label = 1 if idx < n_pos else 0
        seq = seq.upper().replace('U', 'T')
        if len(seq) != SEQ_LEN:
            dropped.append((idx, orig_h, len(seq), label))
            continue
        if label == 1:
            new_h = f'>{split_name}_pos_{pos_i}'
            pos_i += 1
        else:
            new_h = f'>{split_name}_neg_{neg_i}'
            neg_i += 1
        out.append((new_h, seq, label))
    return out, dropped, pos_i, neg_i


def write_fasta(records, path):
    with open(path, 'w') as f:
        for header, seq, _ in records:
            f.write(f'{header}\n{seq}\n')


def sanity_check(records, tag):
    """Verify the alphabet, the length, the centre base and the class balance."""
    seqs = [s for _, s, _ in records]
    labels = [l for _, _, l in records]
    alpha = collections.Counter(c for s in seqs for c in s)
    lens = collections.Counter(len(s) for s in seqs)
    center = collections.Counter(s[len(s) // 2] for s in seqs)
    print(f'  [{tag}] n={len(records)}  pos={sum(labels)}  neg={len(labels) - sum(labels)}')
    print(f'         alphabet={dict(alpha)}  len={dict(lens)}  center={dict(center)}')
    assert set(alpha) <= set('ACGT'), f'non-ACGT letters: {set(alpha) - set("ACGT")}'
    assert set(lens) == {SEQ_LEN}, f'sequences that are not {SEQ_LEN} nt: {dict(lens)}'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FOLD_DIR, exist_ok=True)

    train_recs = read_raw(os.path.join(RAW_DIR, 'trainset.txt'))
    test_recs = read_raw(os.path.join(RAW_DIR, 'testset.txt'))
    print(f'raw records: train={len(train_recs)}, test={len(test_recs)}')

    train_out, train_drop, tp, tn = convert(train_recs, TRAIN_N_POS, 'ac4c_train')
    test_out, test_drop, ep, en = convert(test_recs, TEST_N_POS, 'test')

    print('\nafter conversion:')
    sanity_check(train_out, 'train_pool')
    sanity_check(test_out, 'test')
    if train_drop:
        print(f'  [train: dropped {len(train_drop)} sequences that are not {SEQ_LEN} nt]: {train_drop}')
    if test_drop:
        print(f'  [test: dropped {len(test_drop)} sequences that are not {SEQ_LEN} nt]: {test_drop}')

    write_fasta(train_out, os.path.join(OUT_DIR, 'train_pool.fasta'))
    write_fasta(test_out, os.path.join(OUT_DIR, 'test.fasta'))
    print(f'\nwritten: {OUT_DIR}/train_pool.fasta ({len(train_out)}) / test.fasta ({len(test_out)})')

    labels = [l for _, _, l in train_out]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    X_dummy = list(range(len(train_out)))
    print(f'\nstratified five-fold split (StratifiedKFold, seed={SEED}):')
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_dummy, labels), start=1):
        tr = [train_out[i] for i in tr_idx]
        va = [train_out[i] for i in va_idx]
        write_fasta(tr, os.path.join(FOLD_DIR, f'fold_{fold_idx}_train.fasta'))
        write_fasta(va, os.path.join(FOLD_DIR, f'fold_{fold_idx}_val.fasta'))
        tr_pos = sum(l for _, _, l in tr)
        va_pos = sum(l for _, _, l in va)
        print(f'  Fold {fold_idx}: train={len(tr)} (pos={tr_pos}) / val={len(va)} (pos={va_pos})')

    print('\ndone.')


if __name__ == '__main__':
    main()
