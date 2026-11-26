# Duet-m7G

Interpretable prediction of RNA N7-methylguanosine (m7G) sites by fusing two
complementary pretrained nucleotide language models.

Each sequence window is read twice: by DNABERT under two overlapping k-mer
tokenizations, and by RNA-FM, a frozen RNA foundation model. Each branch
refines its own token representations with self-attention and reduces them to a
single vector with a learnable attention pooling layer, whose per-position
weights are the hook used for interpretability. The two vectors are
concatenated and classified by a high-capacity nonlinear head.

The same architecture is reused for N4-acetylcytidine (ac4C) site prediction,
trained separately with no change other than the window size.

## Results

Independent test set (2,194 sequences), five-fold ensemble by logits averaging:

| ACC | SE | SP | MCC | AUC |
|---|---|---|---|---|
| 0.8997 | 0.8678 | 0.9316 | 0.8011 | 0.9646 |

Individual folds reach a test MCC of 0.7829 ± 0.0053 (sample s.d.), so the ensemble adds
about 0.018 MCC over a single fold.

## Installation

```bash
conda create -n duet-m7g python=3.10
conda activate duet-m7g
pip install -r requirements.txt
```

`transformers` is pinned to 4.30: DNABERT ships a vocabulary that later major
releases no longer load.

## Data

The benchmark datasets, including the five-fold partition used here, are
archived at Zenodo, DOI [10.5281/zenodo.22096235](https://doi.org/10.5281/zenodo.22096235).

Unpack them so that the repository root contains:

```
data/            train.fasta, valid.fasta, test.fasta, 5fold/
data_ac4c/       train_pool.fasta, test.fasta, 5fold/, raw/
```

Labels are read from the FASTA header: a header containing `neg` is negative,
anything else is positive.

The five-fold partition can also be rebuilt from `train.fasta` and
`valid.fasta`, which reproduces the shipped files exactly:

```bash
python data_utils/make_5fold.py
python data_utils/make_ac4c_data.py     # ac4C, from data_ac4c/raw/
```

## Pretrained encoders

- **DNABERT** (4-mer and 6-mer). Download the pretrained models from the
  [DNABERT repository](https://github.com/jerryji1993/DNABERT) and place them as
  `pretrained/DNABERT/DNABERT_{3,4,5,6}mer`, or point `DNABERT_DIR` at your own
  location. `dnabert.py::get_dnabert_path` is the only place this path is
  defined.
- **RNA-FM** is fetched by the `fm` package on first use and is never updated
  during training.

## Usage

Run everything from the repository root.

```bash
python train_5fold.py                  # five-fold training, m7G
python vote.py                         # ensemble the five folds on the test set
```

Each fold writes to `models_log/duet_m7g_5fold/<timestamp>/<fold>/` with
`checkpoints/`, per-epoch `tables/table.csv`, TensorBoard `logs/` and the raw
predictions used for figures in `draw_data/`. The run root additionally holds
`cv_summary.json`. `vote.py` reads that layout, evaluates both logits and
probability averaging, and writes `vote_result_test.json`; **the reported
numbers are the logits average.**

Useful flags:

| Flag | Meaning |
|---|---|
| `--start_fold N` / `--root_dir DIR` | Resume a run; finished folds are reused. |
| `--bert_depth N` | Final DNABERT layers left trainable (1 = default, 0 = frozen). |
| `--patience N` | Early stopping (0 = run all epochs). |
| `--window N` | Length of the centered input window (501 by default; 201 for the input-window comparison). m7G script only. |
| `--compile 0` | Disable `torch.compile`. |

The ac4C experiment is the same pair of commands with `train_5fold_ac4c.py` and
`vote_ac4c.py`, which use a 201-nt window and write to
`models_log/duet_ac4c_5fold/`.

Training is not bit-wise reproducible because of cuDNN autotuning and
non-deterministic backward reductions. Across 11 pairs of identical runs, the
difference in a fold's test MCC between the two runs had a standard deviation
of 0.0085. Compare arms within one run where possible.

## Layout

```
model.py                  Duet-m7G: DNABERT + frozen RNA-FM branches
model_conv_control.py     shared blocks, and the CNN control architecture
conv_branch.py            convolutional encoder used by the control
dnabert.py                DNABERT branches and the weight location
data_utils/               datasets, encodings, five-fold partitioning
train_5fold.py            five-fold training (m7G / ac4C variant alongside)
vote.py                   five-fold ensemble evaluation
utils.py, logging_utils.py
```

## Citation

A manuscript describing Duet-m7G is under review; citation details will be
added here once it is published.

## License

MIT, see [LICENSE](LICENSE).
