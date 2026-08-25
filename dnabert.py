"""DNABERT k-mer encoders and the single place where their weights are located."""
import os

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertConfig, BertModel


def get_dnabert_path(kmer):
    """Directory of the pretrained DNABERT weights for a given k.

    The weight location is defined here and nowhere else. It defaults to
    pretrained/DNABERT/DNABERT_{k}mer under the repository root and can be
    redirected with the DNABERT_DIR environment variable."""
    base = os.environ.get('DNABERT_DIR', os.path.join('pretrained', 'DNABERT'))
    return os.path.join(base, f'DNABERT_{kmer}mer')


class BERT(nn.Module):
    def __init__(self, kmer):
        super(BERT, self).__init__()
        self.kmer = kmer

        self.pretrainpath = get_dnabert_path(self.kmer)

        self.setting = BertConfig.from_pretrained(
            self.pretrainpath,
            num_labels=2,
            finetuning_task="dnaprom",
            cache_dir=None,
        )

        self.bert = BertModel.from_pretrained(self.pretrainpath, config=self.setting)

    def forward(self, tok, return_sequence=False):
        outputs = self.bert(
            input_ids=tok['input_ids'],
            token_type_ids=tok['token_type_ids'],
            attention_mask=tok['attention_mask'],
        )

        if return_sequence:
            return outputs["last_hidden_state"]

        return outputs["pooler_output"]

    def print_trainable_parameters(self):
        """Report how many parameters are trainable."""
        trainable_params = 0
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"trainable: {name}, shape: {param.shape}")

        print(f"\ntrainable parameters: {trainable_params:,}")
        print(f"total parameters: {all_param:,}")
        print(f"trainable share: {100 * trainable_params / all_param:.2f}%")


if __name__ == '__main__':

    bertone = BERT(4)

    for param in bertone.bert.parameters():
        param.requires_grad = False

    for param in bertone.bert.encoder.layer[-1].parameters():
        param.requires_grad = True

    bertone.print_trainable_parameters()
