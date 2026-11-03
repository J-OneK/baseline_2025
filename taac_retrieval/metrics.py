"""Exact full-catalog retrieval and explicit recommendation metrics."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SequenceDataset


def recommendation_metrics(predictions, targets):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if predictions.ndim != 2 or len(predictions) != len(targets):
        raise ValueError('Expected [users, k] predictions and [users] targets.')
    matched = predictions == targets[:, None]
    hits = matched.any(axis=1).astype(np.float64)
    discounts = 1 / np.log2(np.arange(predictions.shape[1]) + 2)
    ndcg = (matched * discounts).sum(axis=1)
    return hits, ndcg


def summarize(predictions, targets, rows, data):
    hits, ndcg = recommendation_metrics(predictions, targets)
    warm = data.train_counts[targets] > 0
    result = {'HR@10': float(hits.mean()), 'NDCG@10': float(ndcg.mean()),
              'users': len(targets), 'candidate_items': data.num_items,
              'candidate_coverage': 1.0, 'filter_seen_items': True}
    for name, selected in [('warm', warm), ('cold', ~warm)]:
        result[name] = {'users': int(selected.sum()),
                        'HR@10': float(hits[selected].mean()) if selected.any() else None,
                        'NDCG@10': float(ndcg[selected].mean()) if selected.any() else None}
    per_user = dict(rows=np.asarray(rows), targets=np.asarray(targets),
                    predictions=np.asarray(predictions), hits=hits, ndcg=ndcg)
    return result, per_user


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def exact_topk(queries, item_vectors, seen, k=10, chunk_size=32768):
    """Streaming exact inner-product top-k; row 0 is padding and never returned."""
    if len(item_vectors) <= k:
        raise ValueError('The catalog must contain at least k nonpadding items.')
    batch_size = len(queries)
    scores = torch.full((batch_size, k), -torch.inf, device=queries.device)
    ids = torch.zeros((batch_size, k), dtype=torch.long, device=queries.device)
    for start in range(1, len(item_vectors), chunk_size):
        end = min(start + chunk_size, len(item_vectors))
        block = queries @ item_vectors[start:end].T
        blocked_rows, blocked_cols = torch.where((seen >= start) & (seen < end))
        block[blocked_rows, seen[blocked_rows, blocked_cols] - start] = -torch.inf
        take = min(k, end - start)
        block_scores, block_ids = block.topk(take, dim=1)
        joined_scores = torch.cat([scores, block_scores], dim=1)
        joined_ids = torch.cat([ids, block_ids + start], dim=1)
        scores, order = joined_scores.topk(k, dim=1)
        ids = joined_ids.gather(1, order)
    if not torch.isfinite(scores).all():
        raise ValueError('Not enough eligible finite-scoring candidates after history filtering.')
    return ids


@torch.inference_mode()
def evaluate(model, data, split, device, batch_size=256, candidate_chunk=32768):
    model.eval()
    # The candidate universe is fixed by the cache, independent of model/seed.
    vectors = torch.empty(data.num_items + 1, model.hidden, device=device)
    for start in range(0, len(vectors), 8192):
        end = min(start + 8192, len(vectors))
        vectors[start:end] = model.encode_items(torch.arange(start, end, device=device))
    if not torch.isfinite(vectors).all():
        raise FloatingPointError('Non-finite candidate embeddings.')
    loader = DataLoader(SequenceDataset(data, split), batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=str(device).startswith('cuda'))
    predictions, targets, rows = [], [], []
    for cpu_batch in loader:
        batch = to_device(cpu_batch, device)
        queries = model.predict(batch)
        if not torch.isfinite(queries).all():
            raise FloatingPointError('Non-finite user embeddings.')
        if (batch['items'] == batch['target'].unsqueeze(1)).any():
            raise AssertionError('Held-out target leaked into its history.')
        topk = exact_topk(queries, vectors, batch['items'], chunk_size=candidate_chunk)
        predictions.append(topk.cpu().numpy())
        targets.append(cpu_batch['target'].numpy())
        rows.append(cpu_batch['row'].numpy())
    return summarize(np.concatenate(predictions), np.concatenate(targets), np.concatenate(rows), data)


def evaluate_popularity(data, split):
    # Stable tie-breaking by local item ID; only training-prefix counts are used.
    ranking = np.lexsort((np.arange(len(data.train_counts)), -data.train_counts))
    ranking = ranking[ranking != 0]
    predictions, targets, rows = [], [], []
    for sample in SequenceDataset(data, split):
        seen = set(sample['items'])
        chosen = []
        for item in ranking:
            if item not in seen:
                chosen.append(item)
                if len(chosen) == 10:
                    break
        predictions.append(chosen)
        targets.append(sample['target'])
        rows.append(sample['row'])
    return summarize(np.asarray(predictions), np.asarray(targets), np.asarray(rows), data)
