"""Build the stratified five-fold partition for the m7G task.

train.fasta and valid.fasta are pooled and split with StratifiedKFold so that
each fold keeps the class balance; test.fasta is left untouched and is shared
by all folds as the held-out set.

Writes data/5fold/fold_{i}_train.fasta and fold_{i}_val.fasta for i = 1..5. The
five validation sets are disjoint and together cover the pool exactly.

Labels are taken from the FASTA header, as in data_utils/dataset.py.

Run from the repository root:
  python data_utils/make_5fold.py"""
import os
from sklearn.model_selection import StratifiedKFold

DATA_DIR = './data'
INPUT_FASTAS = ['train.fasta', 'valid.fasta']
OUTPUT_DIR = './data/5fold'
N_SPLITS = 5
SEED = 17


def read_fasta_records(path):
    """Read a FASTA file into [(header, sequence), ...]."""
    records = []
    header = None
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                header = line
            else:
                records.append((header, line))
                header = None
    return records


def label_of(header):
    """0 if the header contains 'neg', otherwise 1."""
    return 0 if 'neg' in header.lower() else 1


def write_fasta(records, path):
    with open(path, 'w') as f:
        for header, seq in records:
            f.write(f'{header}\n{seq}\n')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_records = []
    for name in INPUT_FASTAS:
        recs = read_fasta_records(os.path.join(DATA_DIR, name))
        all_records.extend(recs)
        print(f'{name}: {len(recs)} sequences')
    labels = [label_of(h) for h, _ in all_records]
    n_pos = sum(labels)
    print(f'merged: {len(all_records)} sequences (pos={n_pos}, neg={len(all_records) - n_pos})')

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    X_dummy = list(range(len(all_records)))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dummy, labels), start=1):
        train_recs = [all_records[i] for i in train_idx]
        val_recs = [all_records[i] for i in val_idx]

        train_path = os.path.join(OUTPUT_DIR, f'fold_{fold_idx}_train.fasta')
        val_path = os.path.join(OUTPUT_DIR, f'fold_{fold_idx}_val.fasta')
        write_fasta(train_recs, train_path)
        write_fasta(val_recs, val_path)

        tr_pos = sum(label_of(h) for h, _ in train_recs)
        va_pos = sum(label_of(h) for h, _ in val_recs)
        print(f'  Fold {fold_idx}: train={len(train_recs)} (pos={tr_pos}) / '
              f'val={len(val_recs)} (pos={va_pos})')

    print(f'\ndone; folds written to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
