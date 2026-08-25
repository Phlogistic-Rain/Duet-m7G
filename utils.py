"""Checkpoint retention and early stopping helpers."""
import os
import shutil

import torch

from logging_utils import alog


def checkpoint_save_val_test(model, savedir, epoch, val_mcc, test_mcc, reserved_num, model_name):
    filename = '{}_epoch{}_mcc{:.4f}_{:.4f}.pth'.format(model_name, epoch, val_mcc, test_mcc)
    file_path = os.path.join(savedir, filename)

    model_files = [f for f in os.listdir(savedir) if f.endswith('.pth')]
    model_files.sort(key=lambda x: float(x.split('_mcc')[1].split('_')[0]), reverse=True)

    if len(model_files) > 0:
        lowest_mcc = float(model_files[-1].split('_mcc')[1].split('_')[0])

        if val_mcc > lowest_mcc:
            torch.save(model.state_dict(), file_path)
            alog('CKPT', epoch=epoch, val_mcc=f'{val_mcc:.4f}', test_mcc=f'{test_mcc:.4f}')

            if len(model_files) >= reserved_num:
                os.remove(os.path.join(savedir, model_files[-1]))

    else:
        torch.save(model.state_dict(), file_path)
        alog('CKPT', epoch=epoch, val_mcc=f'{val_mcc:.4f}', test_mcc=f'{test_mcc:.4f}')



class EarlyStopping:
    """Stop training once the monitored metric has not improved for `patience` epochs."""

    def __init__(self, patience=5, min_delta=0, mode='max'):
        """
        Args:
            patience: number of epochs without improvement that are tolerated
            min_delta: smallest change that counts as an improvement
            mode: 'max' if larger is better (e.g. MCC), 'min' otherwise (e.g. loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta