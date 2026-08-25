"""Dataset for the DNABERT branches, with ENAC encoding for the control model."""
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from tqdm import tqdm

from dnabert import get_dnabert_path


def _kmer_split(seq, k):
    """Split a sequence into a space-separated k-mer string, e.g. 'ACGT CGTA GTAC ...'."""
    return " ".join(seq[i:i + k] for i in range(len(seq) + 1 - k))


class RNAENACDataset(Dataset):
    """
    RNA sequence dataset with ENAC (Enhanced Nucleic Acid Composition) encoding
    and a fixed-length window taken from the centre of each sequence.

    All preprocessing happens once in __init__: centre cropping, ENAC encoding,
    and DNABERT tokenization for both k-mer branches, stored as tensors. The
    training loop only slices, so tokenization never competes with the GPU.

    Returns (tok1, tok2, encoded, label):
        - tok1/tok2 (dict): input_ids / token_type_ids / attention_mask for kmer1 and kmer2
        - encoded (Tensor): ENAC encoding of shape [seq_length - 1, 4]
        - label (Tensor): 0 or 1
    """

    def __init__(self, fasta_path, seq_length, kmer1=4, kmer2=6):
        """
        Args:
            fasta_path (str): path to the FASTA file
            seq_length (int): length of the window taken from the sequence centre
            kmer1, kmer2 (int): k for the two DNABERT branches
        """
        self.sequences, self.labels = self._read_fasta(fasta_path)
        self.seq_length = seq_length
        self.kmer1 = kmer1
        self.kmer2 = kmer2

        self.dinucleotide_dict = {
            'AA': [1, 0, 0, 0],
            'AC': [0.5, 0.5, 0, 0],
            'AG': [0.5, 0, 0.5, 0],
            'AT': [0.5, 0, 0, 0.5],
            'CA': [0.5, 0.5, 0, 0],
            'CC': [0, 1, 0, 0],
            'CG': [0, 0.5, 0.5, 0],
            'CT': [0, 0.5, 0, 0.5],
            'GA': [0.5, 0, 0.5, 0],
            'GC': [0, 0.5, 0.5, 0],
            'GG': [0, 0, 1, 0],
            'GT': [0, 0, 0.5, 0.5],
            'TA': [0.5, 0, 0, 0.5],
            'TC': [0, 0.5, 0, 0.5],
            'TG': [0, 0, 0.5, 0.5],
            'TT': [0, 0, 0, 1]
        }

        windowed = [self._extract_center_window(s) for s in self.sequences]
        self.encoded = [torch.tensor(self._encode_enac(s), dtype=torch.float32) for s in windowed]
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.tok1 = self._tokenize_kmer(windowed, kmer1)
        self.tok2 = self._tokenize_kmer(windowed, kmer2)

    def _tokenize_kmer(self, seqs, k, chunk=1000):
        """Tokenize every sequence at the given k, padded to a fixed length.

        max_length is (seq_length - k + 1) k-mers plus the two special tokens.
        Tokenization is chunked and the results concatenated.
        """
        tokenizer = BertTokenizerFast.from_pretrained(get_dnabert_path(k))
        kmer_strs = [_kmer_split(s, k) for s in seqs]
        max_len = self.seq_length - k + 3

        buffers = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        for i in tqdm(range(0, len(kmer_strs), chunk), desc=f'tokenize {k}mer', unit='chunk',
                      disable=not sys.stderr.isatty()):
            enc = tokenizer(
                kmer_strs[i:i + chunk], return_tensors='pt',
                padding='max_length', truncation=True, max_length=max_len,
            )
            for key in buffers:
                buffers[key].append(enc[key])
        return {key: torch.cat(v, dim=0) for key, v in buffers.items()}

    def _read_fasta(self, file_path):
        """
        Read a FASTA file.

        Labels are taken from the header: a header containing 'neg' is a
        negative sample, anything else is positive.

        Returns:
            sequences (list): the sequences
            labels (list): 0 for negative, 1 for positive
        """
        sequences = []
        labels = []

        with open(file_path, 'r') as fasta:
            for line in fasta:
                line = line.strip()
                if line.startswith('>'):
                    if 'neg' in line.lower():
                        labels.append(0)
                    else:
                        labels.append(1)
                else:
                    sequences.append(line)

        return sequences, labels

    def _extract_center_window(self, sequence):
        """
        Take a fixed-length window from the centre of a sequence.

        Returns:
            windowed_seq (str): the cropped sequence
        """
        seq_len = len(sequence)

        if seq_len <= self.seq_length:
            return sequence

        center = seq_len // 2

        half_window = self.seq_length // 2
        start = center - half_window
        end = start + self.seq_length

        return sequence[start:end]

    def _encode_enac(self, sequence):
        """
        Encode a sequence with the ENAC scheme: each sliding dinucleotide is
        mapped to a four-dimensional vector.

        Returns:
            encoded (numpy.ndarray): feature matrix of shape [length - 1, 4]
        """
        length = len(sequence) - 1

        encoded_features = []
        for i in range(length):
            dinucleotide = sequence[i:i + 2]
            encoded_features.append(self.dinucleotide_dict[dinucleotide])

        return np.array(encoded_features, dtype=np.float32)

    def __len__(self):
        """Number of samples."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Return one sample. Preprocessing already happened in __init__, so this
        only slices the stored tensors.

        Returns:
            tok1 (dict): input_ids / token_type_ids / attention_mask for the kmer1 branch
            tok2 (dict): the same for the kmer2 branch
            encoded (Tensor): ENAC encoding of shape [seq_length - 1, 4]
            label (Tensor): 0 or 1
        """
        tok1 = {key: v[idx] for key, v in self.tok1.items()}
        tok2 = {key: v[idx] for key, v in self.tok2.items()}
        return tok1, tok2, self.encoded[idx], self.labels[idx]



if __name__ == '__main__':
    window_size = 501
    dataset = RNAENACDataset('./data/test.fasta', window_size)

    print(f"Dataset size: {len(dataset)}")

    tok1, tok2, encoded, label = dataset[0]

    print('tok1 input_ids shape:', tok1['input_ids'].shape)
    print('tok2 input_ids shape:', tok2['input_ids'].shape)
    print('encoded shape:', encoded.shape)
    print('label:', label)
