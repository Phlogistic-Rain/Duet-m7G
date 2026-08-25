"""Shared building blocks, and the convolutional control architecture.

`CrossAttentionBlock`, `AttnPool` and `FatHead` are reused by the main
Duet-m7G model. `DuetM7GConv` replaces the RNA-FM branch with a convolutional
encoder trained from scratch and serves as the control arm in the component
ablation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dnabert import BERT
from data_utils.dataset import RNAENACDataset
from conv_branch import Convolution


class CrossAttentionBlock(nn.Module):
    """Token-level refinement block: multi-head attention, residual, layer norm.

    Each branch applies this to itself (query and key/value come from the same
    sequence), so refinement is within-branch; the two branches are combined
    only in the fusion head. The `cross_*` attribute names are retained for
    compatibility with released checkpoints.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_seq, kv_seq):
        """query_seq/kv_seq are [B, L, dim]; returns the context-enriched [B, Lq, dim]."""
        attn_out, _ = self.attn(query_seq, kv_seq, kv_seq)
        out = self.norm(query_seq + self.dropout(attn_out))
        return out


class FatHead(nn.Module):
    """High-capacity nonlinear classification head.

    The fused representation is first bottlenecked to d_model=16 and then
    passed through three TransformerEncoderLayer blocks. The sequence length is
    one, so self-attention is degenerate and contributes few parameters; the
    capacity comes from the 16 -> ff_dim -> 16 feed-forward network inside each
    of the three layers.
    """

    def __init__(self, in_dim, num_classes=2, d_model=16, nhead=2, num_layers=3, ff_dim=2048):
        super(FatHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
                                       batch_first=True),
            num_layers=num_layers
        )
        self.fc3 = nn.Linear(d_model, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.fc4(x)
        return x


class AttnPool(nn.Module):
    """Learnable single-query attention pooling, used instead of mean pooling.

    Every position is scored, the scores are normalised with a softmax, and the
    sequence is reduced to a weighted sum. The resulting weights are the
    per-position importances exposed for interpretability analysis.
    """

    def __init__(self, dim):
        super(AttnPool, self).__init__()
        self.q = nn.Linear(dim, 1)

    def forward(self, x, return_w=False):
        scores = self.q(x).squeeze(-1)
        w = torch.softmax(scores, dim=1)
        out = torch.bmm(w.unsqueeze(1), x).squeeze(1)
        return (out, w) if return_w else out


class DuetM7GConv(nn.Module):
    """Control architecture: DNABERT branches paired with a scratch-trained CNN.

    Identical to the main model except that the frozen RNA foundation model is
    replaced by a convolutional encoder of comparable size trained from random
    initialisation, which isolates the contribution of the pretrained RNA
    representation.
    """

    def __init__(self, kmer1, kmer2, seq_length, model_dim=256, num_heads=8):
        super(DuetM7GConv, self).__init__()

        self.bertone = BERT(kmer1)
        self.berttwo = BERT(kmer2)

        for param in self.bertone.bert.parameters():
            param.requires_grad = False

        for param in self.berttwo.bert.parameters():
            param.requires_grad = False

        for param in self.bertone.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        for param in self.berttwo.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        bert_hidden = 768
        self.bert_proj = nn.Sequential(
            nn.Linear(bert_hidden, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.COV = Convolution(seq_length, use_bn=True)

        conv_channels = 64
        self.conv_proj = nn.Sequential(
            nn.Linear(conv_channels, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.pool_bert = AttnPool(model_dim)
        self.pool_conv = AttnPool(model_dim)

        self.cross_bert_to_conv = CrossAttentionBlock(model_dim, num_heads, dropout=0.3)
        self.cross_conv_to_bert = CrossAttentionBlock(model_dim, num_heads, dropout=0.3)

        self.fusion_layer = FatHead(model_dim * 2, num_classes=2)

    def forward(self, tok1, tok2, encoded, return_attn=False):
        seqX = self.bertone(tok1, return_sequence=True)
        seqY = self.berttwo(tok2, return_sequence=True)
        bert_seq = torch.cat([seqX, seqY], dim=1)
        bert_seq = self.bert_proj(bert_seq)

        conv_map = self.COV(encoded, return_map=True)
        conv_seq = conv_map.transpose(1, 2)
        conv_seq = self.conv_proj(conv_seq)

        bert_ctx = self.cross_bert_to_conv(bert_seq, bert_seq)
        conv_ctx = self.cross_conv_to_bert(conv_seq, conv_seq)

        if return_attn:
            bert_vec, w_bert = self.pool_bert(bert_ctx, return_w=True)
            conv_vec, w_conv = self.pool_conv(conv_ctx, return_w=True)
        else:
            bert_vec = self.pool_bert(bert_ctx)
            conv_vec = self.pool_conv(conv_ctx)

        fused_features = torch.cat([bert_vec, conv_vec], dim=-1)

        output = self.fusion_layer(fused_features)

        if return_attn:
            return output, w_bert, w_conv
        return output


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    window_size = 501
    dataset = RNAENACDataset('./data/test.fasta', window_size)
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    tok1, tok2, encoded, label = next(iter(dataloader))

    tok1 = {k: v.to(device) for k, v in tok1.items()}
    tok2 = {k: v.to(device) for k, v in tok2.items()}
    encoded = encoded.to(device)
    label = label.to(device)

    model = DuetM7GConv(4, 6, seq_length=501)
    model = model.to(device)

    output = model(tok1, tok2, encoded)

    print(output)
