import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
import copy

from data import SPECIAL_TOKENS

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std = 0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device = x.device)
        return self.emb(n)[None, :, :]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[1], :]
        return self.dropout(x)

class FaceTransformer(nn.Module):
    def __init__(
        self,
        keypoints_size,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        max_seq_len=512,
        special_tokens=SPECIAL_TOKENS
    ):
        super().__init__()
        self.keypoints_size = keypoints_size
        self.input_size = int(np.arange(keypoints_size).sum())
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.special_tokens = special_tokens

        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        self.token_embedding = nn.Parameter(torch.randn(len(self.special_tokens), self.hidden_size))
        self.pos_embedding = PositionalEmbedding(self.hidden_size, dropout=self.dropout, max_len=self.max_seq_len)

        encoder_layers = nn.TransformerEncoderLayer(self.hidden_size, self.num_heads, self.hidden_size * 4, self.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, self.num_layers)
        self.norm = nn.LayerNorm(self.hidden_size)

        self.mask_head = nn.Linear(self.hidden_size, self.input_size)
        self.nsp_head = nn.Linear(self.hidden_size, 2)

    def forward(self, arrays, special_mask, pretrain=True):
        arrays = torch.linalg.norm(arrays[..., None] - arrays, dim=-1, ord=2)
        mask = special_mask == self.special_tokens['MASK']
        arrays[mask] = 0
        
        array_embeddings = self.embedding(arrays)
        
        clear_mask = (special_mask == 0).to(torch.float)
        array_embeddings = array_embeddings * clear_mask
        
        idx_offset = min(self.special_tokens.values())
        special_add = torch.zeros_like(array_embeddings)
        idxs = torch.where(torch.logical_not(clear_mask))
        special_add[idxs] = special_add[idxs] + self.token_embedding[special_mask[idxs] - idx_offset]
        array_embeddings = array_embeddings + special_add
        
        embedded = self.pos_embedding(array_embeddings)

        encoded = self.encoder(embedded)
        encoded = self.norm(encoded)

        if not pretrain:
            return encoded

        mask_preds = self.mask_head(encoded[mask])
        nsp_preds = self.nsp_head(encoded[:,0])
        return mask_preds, nsp_preds
        

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    T_max : int
        The maximum number of iterations within the first cycle.
    eta_min : float, optional (default: 0)
        The minimum learning rate.
    last_epoch : int, optional (default: -1)
        The index of the last epoch.
    """

    def __init__(self,
                 optimizer,
                 T_max,
                 eta_min = 0.,
                 last_epoch = -1,
                 factor = 1.):
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.
        self._updated_cycle_len = T_max
        self._initialized = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs
