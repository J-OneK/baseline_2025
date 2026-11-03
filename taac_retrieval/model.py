"""Vectorized SASRec-style baseline and a controlled history-action ablation.

The attention and feed-forward blocks come from the supplied baseline. Both
variants share the same encoders, initialization, positions and training loss;
the action variant additionally sums an embedding of each observed past action.
"""

import math

import numpy as np
import torch
from torch import nn

from model import FlashMultiHeadAttention, PointWiseFeedForward
from .data import load_array


class SequenceRetriever(nn.Module):
    def __init__(self, data, hidden=32, blocks=1, heads=1, dropout=0.2, use_actions=False):
        super().__init__()
        self.use_actions = use_actions
        self.hidden = hidden
        meta = data.metadata
        self.item_fields, self.user_fields = meta['item_fields'], meta['user_fields']
        self.array_fields = meta['user_array_fields']
        for name in ['item_sparse', 'item_mm81', 'user_sparse', 'item_id_lookup']:
            self.register_buffer(name, torch.from_numpy(load_array(data.root, name)), persistent=False)
        for field in self.array_fields:
            self.register_buffer(f'user_array_{field}', torch.from_numpy(
                load_array(data.root, f'user_array_{field}')), persistent=False)
        self.item_emb = nn.Embedding(int(self.item_id_lookup.max()) + 1, hidden, padding_idx=0)
        self.user_emb = nn.Embedding(data.num_users + 1, hidden, padding_idx=0)
        self.sparse_emb = nn.ModuleDict({field: nn.Embedding(meta['feature_sizes'][field], hidden, padding_idx=0)
                                        for field in self.item_fields + self.user_fields + self.array_fields})
        self.mm_projection = nn.Linear(32, hidden)
        self.item_dnn = nn.Linear(hidden * (len(self.item_fields) + 2), hidden)
        self.user_dnn = nn.Linear(hidden * (1 + len(self.user_fields) + len(self.array_fields)), hidden)
        self.position_emb = nn.Embedding(data.max_history + 2, hidden, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.ModuleList([FlashMultiHeadAttention(hidden, heads, dropout) for _ in range(blocks)])
        self.ffn = nn.ModuleList([PointWiseFeedForward(hidden, dropout) for _ in range(blocks)])
        self.norm_attn = nn.ModuleList([nn.LayerNorm(hidden, eps=1e-8) for _ in range(blocks)])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(hidden, eps=1e-8) for _ in range(blocks)])
        self.final_norm = nn.LayerNorm(hidden, eps=1e-8)
        # Allocate in both variants so every shared parameter has identical RNG initialization.
        self.action_emb = nn.Embedding(4, hidden, padding_idx=0)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.xavier_normal_(module.weight)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_items(self, ids):
        lookup = self.item_id_lookup[ids]
        parts = [self.item_emb(lookup)]
        sparse = self.item_sparse[ids]
        parts.extend(self.sparse_emb[field](sparse[..., j]) for j, field in enumerate(self.item_fields))
        parts.append(self.mm_projection(self.item_mm81[ids]))
        result = torch.relu(self.item_dnn(torch.cat(parts, dim=-1)))
        return result * (ids != 0).unsqueeze(-1)

    def encode_users(self, ids):
        parts = [self.user_emb(ids)]
        sparse = self.user_sparse[ids]
        parts.extend(self.sparse_emb[field](sparse[..., j]) for j, field in enumerate(self.user_fields))
        parts.extend(self.sparse_emb[field](getattr(self, f'user_array_{field}')[ids]).sum(-2)
                     for field in self.array_fields)
        return torch.relu(self.user_dnn(torch.cat(parts, dim=-1)))

    def encode_history(self, items, kinds, actions, users):
        item_mask, user_mask, valid = kinds == 1, kinds == 2, kinds != 0
        hidden = self.encode_items(items) * item_mask.unsqueeze(-1)
        hidden = hidden + self.encode_users(users).unsqueeze(1) * user_mask.unsqueeze(-1)
        if self.use_actions:
            hidden = hidden + self.action_emb(actions) * item_mask.unsqueeze(-1)
        positions = valid.long().cumsum(1) * valid
        hidden = self.dropout(hidden * math.sqrt(self.hidden) + self.position_emb(positions))
        hidden = hidden * valid.unsqueeze(-1)
        length = items.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=items.device).tril()
        allowed = causal.unsqueeze(0) & valid.unsqueeze(1)
        # Padded queries attend to themselves only, avoiding fully masked softmax rows.
        allowed = allowed | ((~valid).unsqueeze(-1) & torch.eye(
            length, dtype=torch.bool, device=items.device).unsqueeze(0))
        for attention, ffn, norm_a, norm_f in zip(self.attention, self.ffn, self.norm_attn, self.norm_ffn):
            residual, _ = attention(hidden, hidden, hidden, attn_mask=allowed)
            hidden = norm_a(hidden + residual)
            hidden = norm_f(hidden + ffn(hidden)) * valid.unsqueeze(-1)
        return self.final_norm(hidden) * valid.unsqueeze(-1)

    def forward(self, batch, negative):
        states = self.encode_history(batch['items'], batch['kinds'], batch['actions'], batch['user'])
        return ((states * self.encode_items(batch['positive'])).sum(-1),
                (states * self.encode_items(negative)).sum(-1))

    def predict(self, batch):
        return self.encode_history(batch['items'], batch['kinds'], batch['actions'], batch['user'])[:, -1]
