"""Duet-m7G: a dual-branch architecture over two pretrained nucleotide encoders.

One branch reads the sequence with DNABERT under two overlapping k-mer
tokenizations; the other reads it with a frozen RNA foundation model (RNA-FM).
Each branch refines its own token representations with self-attention, pools
them with a learnable single-query attention layer, and the two pooled vectors
are concatenated and classified by a high-capacity nonlinear head.

    RNA sequence, centre-cropped to WIN nucleotides
    |-- DNABERT branch (4-mer and 6-mer, backbones frozen except the last layer)
    |     concatenate both k-mer token streams -> linear projection -> [B, Lx+Ly, 256]
    |-- RNA-FM branch (frozen throughout, forward under no_grad)
          layer-12 residue tokens without BOS/EOS -> linear projection 640 -> 256 -> [B, WIN, 256]
    each branch: self-attention refinement -> attention pooling -> [B, 256]
    concatenate -> [B, 512] -> fusion head -> logits [B, 2]

RNA-FM is loaded through the `fm` package; DNABERT through `transformers`."""
import torch
import torch.nn as nn

from dnabert import BERT
from model_conv_control import CrossAttentionBlock, AttnPool, FatHead

MODEL_DIM = 256


class BertBranch(nn.Module):
    """Two DNABERT encoders, projected, refined and pooled into a single vector.

    Both backbones are frozen except for their final encoder layer. The two
    k-mer token streams are concatenated along the sequence axis before
    projection, so the branch sees one joint representation of the window.
    """

    def __init__(self, kmer1=4, kmer2=6, model_dim=MODEL_DIM, num_heads=8):
        super().__init__()
        self.bertone = BERT(kmer1)
        self.berttwo = BERT(kmer2)
        for b in (self.bertone, self.berttwo):
            for p in b.bert.parameters():
                p.requires_grad = False
            for p in b.bert.encoder.layer[-1].parameters():
                p.requires_grad = True
        self.bert_proj = nn.Sequential(nn.Linear(768, model_dim), nn.LayerNorm(model_dim))
        self.refine = CrossAttentionBlock(model_dim, num_heads, dropout=0.3)
        self.pool = AttnPool(model_dim)

    def forward(self, tok1, tok2, return_w=False):
        sx = self.bertone(tok1, return_sequence=True)
        sy = self.berttwo(tok2, return_sequence=True)
        seq = torch.cat([sx, sy], dim=1)
        seq = self.bert_proj(seq)
        seq = self.refine(seq, seq)
        return self.pool(seq, return_w=return_w)


class RNAFMBranch(nn.Module):
    """A frozen RNA-FM encoder, projected, refined and pooled into a single vector.

    The backbone is never updated and is always run under `no_grad`, which
    keeps its memory and compute cost out of the training loop.
    """

    def __init__(self, win, model_dim=MODEL_DIM, num_heads=8):
        super().__init__()
        import fm
        self.model, _alphabet = fm.pretrained.rna_fm_t12()
        for p in self.model.parameters():
            p.requires_grad = False
        self.win = win
        self.proj = nn.Sequential(nn.Linear(640, model_dim), nn.LayerNorm(model_dim))
        self.refine = CrossAttentionBlock(model_dim, num_heads, dropout=0.3)
        self.pool = AttnPool(model_dim)

    def forward(self, rnafm_ids, return_w=False):
        with torch.no_grad():
            rep = self.model(rnafm_ids, repr_layers=[12])['representations'][12]
        rep = rep[:, 1:1 + self.win, :]
        s = self.proj(rep)
        s = self.refine(s, s)
        return self.pool(s, return_w=return_w)


class DuetM7G(nn.Module):
    """Duet-m7G: DNABERT and frozen RNA-FM branches with attention pooling.

    Construct as `DuetM7G(kmer1=4, kmer2=6, win=501)`; use win=201 for the
    ac4C task. `forward(tok1, tok2, rnafm_ids, return_attn=False)` returns the
    class logits, and with `return_attn=True` also the per-position pooling
    weights of both branches, which are the hook used for interpretability
    analysis.
    """

    def __init__(self, kmer1=4, kmer2=6, win=501, model_dim=MODEL_DIM, num_heads=8):
        super().__init__()
        self.bert = BertBranch(kmer1, kmer2, model_dim, num_heads)
        self.rnafm = RNAFMBranch(win, model_dim, num_heads)
        self.head = FatHead(model_dim * 2, num_classes=2)

    def forward(self, tok1, tok2, rnafm_ids, return_attn=False):
        if return_attn:
            bert_vec, w_bert = self.bert(tok1, tok2, return_w=True)
            rnafm_vec, w_rnafm = self.rnafm(rnafm_ids, return_w=True)
            out = self.head(torch.cat([bert_vec, rnafm_vec], dim=-1))
            return out, w_bert, w_rnafm
        bert_vec = self.bert(tok1, tok2)
        rnafm_vec = self.rnafm(rnafm_ids)
        return self.head(torch.cat([bert_vec, rnafm_vec], dim=-1))


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data_utils.dataset_rnafm import RNAFMDataset

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    win = 501
    ds = RNAFMDataset('./data/test.fasta', win)
    print(f'Dataset size: {len(ds)}')
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    tok1, tok2, rid, label = next(iter(loader))
    tok1 = {k: v.to(device) for k, v in tok1.items()}
    tok2 = {k: v.to(device) for k, v in tok2.items()}
    rid = rid.to(device)

    model = DuetM7G(4, 6, win=win).to(device)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {ntr:,}')
    out = model(tok1, tok2, rid)
    print('logits shape:', out.shape)
