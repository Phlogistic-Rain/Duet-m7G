"""Dataset for Duet-m7G: DNABERT tokens plus RNA-FM residue token ids.

Centre cropping and DNABERT tokenization are inherited from `RNAENACDataset`,
so both branches see exactly the same window. RNA-FM tokenization is added once
in __init__ and stored as tensors, independently of the number of data-loading
workers.

Returns (tok1, tok2, rnafm_ids, label):
  - tok1/tok2 (dict): the two DNABERT branches
  - rnafm_ids (Tensor[WIN + 2]): RNA-FM tokens including BOS/EOS; the model
    slices out the residue tokens
  - label (Tensor): 0 or 1

The RNA-FM branch receives the centre window uppercased with T mapped back to
U, as expected by the RNA-FM alphabet."""
import torch
from torch.utils.data import Dataset

from data_utils.dataset import RNAENACDataset


class RNAFMDataset(Dataset):
    def __init__(self, fasta_path, win, alphabet=None, kmer1=4, kmer2=6):
        """
        Args:
            fasta_path (str): path to the FASTA file
            win (int): centre window length, shared by both branches
            alphabet: RNA-FM alphabet; loaded internally when omitted
            kmer1, kmer2 (int): k for the two DNABERT branches
        """
        self.base = RNAENACDataset(fasta_path, win, kmer1=kmer1, kmer2=kmer2)
        self.labels = self.base.labels

        if alphabet is None:
            import fm
            _m, alphabet = fm.pretrained.rna_fm_t12()
            del _m

        windowed = [self.base._extract_center_window(s) for s in self.base.sequences]
        rna_seqs = [w.upper().replace('T', 'U') for w in windowed]
        bc = alphabet.get_batch_converter()
        ids = []
        chunk = 1000
        for i in range(0, len(rna_seqs), chunk):
            data = [(str(j), s) for j, s in enumerate(rna_seqs[i:i + chunk])]
            _, _, tok = bc(data)
            ids.append(tok)
        widths = {t.shape[1] for t in ids}
        assert len(widths) == 1, f'inconsistent RNA-FM token widths (windows other than {win}): {widths}'
        self.rnafm_ids = torch.cat(ids, dim=0)
        print(f'  [RNAFMDataset] {fasta_path}: rnafm_ids {tuple(self.rnafm_ids.shape)}')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        tok1, tok2, _enc, label = self.base[idx]
        return tok1, tok2, self.rnafm_ids[idx], label


if __name__ == '__main__':
    ds = RNAFMDataset('./data/test.fasta', 501)
    print('size:', len(ds))
    tok1, tok2, rid, label = ds[0]
    print('tok1 input_ids:', tok1['input_ids'].shape,
          '| rnafm_ids:', rid.shape, '| label:', label)
