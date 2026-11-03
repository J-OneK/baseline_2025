"""Array-backed sequence samples shared by every experiment variant."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_array(root, name):
    return np.load(Path(root) / f'{name}.npy', mmap_mode='c', allow_pickle=False)


class PreparedData:
    def __init__(self, root):
        self.root = Path(root)
        self.metadata = json.loads((self.root / 'metadata.json').read_text())
        for name in ['items', 'actions', 'timestamps', 'user_rids', 'lengths',
                     'train_cut', 'val_cut', 'test_cut', 'val_pos', 'test_pos',
                     'train_counts', 'eval_indices', 'catalog_rids', 'catalog_oids']:
            setattr(self, name, load_array(root, name))
        self.max_history = self.items.shape[1]
        self.num_users = len(self.items)
        self.num_items = len(self.catalog_rids) - 1

    def set_eval_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if (indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices)
                or indices.min() < 0 or indices.max() >= self.num_users):
            raise ValueError('Invalid evaluation cohort indices.')
        self.eval_indices = indices


def make_input(items, actions, max_history):
    """Left-pad history, retaining an explicit user token and the last event."""
    items, actions = np.asarray(items)[-max_history:], np.asarray(actions)[-max_history:]
    length = max_history + 1
    start = length - len(items) - 1
    seq = np.zeros(length, dtype=np.int64)
    kinds = np.zeros(length, dtype=np.int64)
    acts = np.zeros(length, dtype=np.int64)
    kinds[start] = 2
    seq[start + 1:] = items
    kinds[start + 1:] = 1
    acts[start + 1:] = actions
    return seq, kinds, acts


class SequenceDataset(Dataset):
    def __init__(self, data, split='train'):
        if split not in ['train', 'val', 'test']:
            raise ValueError(split)
        self.data, self.split = data, split
        self.indices = np.arange(data.num_users) if split == 'train' else data.eval_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        data = self.data
        cut = int(getattr(data, f'{self.split}_cut')[row])
        prefix = data.items[row, :cut]
        acts = data.actions[row, :cut]
        if self.split == 'train':
            seq, kinds, actions = make_input(prefix[:-1], acts[:-1], data.max_history)
            pos = np.zeros(data.max_history + 1, dtype=np.int64)
            pos_actions = np.zeros_like(pos)
            pos[-cut:] = prefix
            pos_actions[-cut:] = acts
            return dict(user=row + 1, row=row, items=seq, kinds=kinds, actions=actions,
                        positive=pos, target_actions=pos_actions,
                        seen=np.pad(prefix, (0, data.max_history - len(prefix))).astype(np.int64))
        seq, kinds, actions = make_input(prefix, acts, data.max_history)
        target = int(data.items[row, getattr(data, f'{self.split}_pos')[row]])
        return dict(user=row + 1, row=row, items=seq, kinds=kinds, actions=actions, target=target)


def sample_negatives(pool, seen, shape, generator):
    """Reject training-prefix items only; never consult held-out targets."""
    if pool.numel() <= seen.shape[1] and any(
            torch.isin(pool, history).all().item() for history in seen):
        raise ValueError('Negative pool is fully covered by a user history.')
    negative = pool[torch.randint(len(pool), shape, device=pool.device, generator=generator)]
    for _ in range(100):
        conflict = (negative.unsqueeze(-1) == seen.unsqueeze(1)).any(-1)
        if not conflict.any():
            return negative
        negative[conflict] = pool[torch.randint(len(pool), (int(conflict.sum()),),
                                              device=pool.device, generator=generator)]
    raise RuntimeError('Negative rejection sampling did not converge.')
