"""Frozen content-only proposals for zero-click tail ads, with a fixed budget.

Catalog IDs are the project's local IDs, not rows of the auxiliary index.
No model parameter or buffer is changed; the main encoder remains independent.
"""

import numpy as np
import torch
from torch.nn import functional as F


def eligible_tail_ids(counts, clickcounts):
    """Return ascending local IDs with 1..5 training events and zero clicks."""
    counts, clickcounts = np.asarray(counts), np.asarray(clickcounts)
    if (counts.ndim != 1 or not counts.size or counts.shape != clickcounts.shape
            or not np.issubdtype(counts.dtype, np.integer)
            or not np.issubdtype(clickcounts.dtype, np.integer)
            or (counts < 0).any() or (clickcounts < 0).any()
            or (clickcounts > counts).any()):
        raise ValueError('Counts must be aligned nonnegative training-event subsets.')
    mask = (counts >= 1) & (counts <= 5) & (clickcounts == 0)
    mask[0] = False
    return np.flatnonzero(mask)


@torch.inference_mode()
def encode_tail_vectors(model, ids, score_mode='raw_ip'):
    """Encode local IDs with zero candidate identity; return [items, hidden].

    The learned attribute, 81 projection and fusion weights are retained. Cosine
    mode normalizes these learned vectors, rather than re-normalizing raw 81.
    """
    if model.training or score_mode not in ('raw_ip', 'cosine'):
        raise ValueError('Use an eval model and raw_ip or cosine scoring.')
    device = model.item_emb.weight.device
    ids = torch.as_tensor(ids, dtype=torch.long, device=device)
    if ids.ndim != 1 or not len(ids) or ids.min() <= 0 or ids.max() >= len(model.item_id_lookup):
        raise ValueError('Expected a nonempty vector of nonpadding local IDs.')
    vectors = []
    for chunk in ids.split(8192):
        parts = [model.item_emb.weight.new_zeros((len(chunk), model.hidden))]
        sparse = model.item_sparse[chunk]
        parts.extend(model.sparse_emb[field](sparse[:, j]) for j, field in enumerate(model.item_fields))
        parts.append(model.mm_projection(model.item_mm81[chunk]))
        vector = torch.relu(model.item_dnn(torch.cat(parts, dim=-1)))
        vectors.append(F.normalize(vector, dim=-1) if score_mode == 'cosine' else vector)
    result = torch.cat(vectors)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Auxiliary vectors must be finite.')
    return result


@torch.inference_mode()
def retrieve_tail(queries, vectors, local_ids, seen, main, k=25, chunk_size=32768):
    """Exact subset search, excluding seen and ALL main candidates before Top-K.

    ``vectors`` is a torch [items, hidden] tensor. ``local_ids`` must be strictly
    ascending and aligned with its rows. Return NumPy int64 [queries, k]; zeros
    mean insufficient finite, unseen, novel proposals. Query normalization has
    no effect on cosine ranking, so normalized vectors alone are sufficient.
    """
    queries = torch.as_tensor(queries, device=vectors.device, dtype=vectors.dtype)
    ids = torch.as_tensor(local_ids, device=vectors.device, dtype=torch.long)
    seen = torch.as_tensor(seen, device=vectors.device, dtype=torch.long)
    main = torch.as_tensor(main, device=vectors.device, dtype=torch.long)
    if (queries.ndim != 2 or vectors.ndim != 2 or queries.shape[1] != vectors.shape[1]
            or ids.ndim != 1 or len(ids) != len(vectors) or not len(ids)
            or ids[0] <= 0 or (ids[1:] <= ids[:-1]).any()
            or seen.ndim != 2 or main.ndim != 2 or len(seen) != len(queries)
            or len(main) != len(queries) or (seen < 0).any() or (main <= 0).any()
            or k < 1 or chunk_size < 1):
        raise ValueError('Invalid aligned queries, subset IDs, history, main list or budget.')
    if not torch.isfinite(queries).all() or not torch.isfinite(vectors).all():
        raise FloatingPointError('Search inputs must be finite.')
    blocked = torch.cat((seen, main), dim=1)
    mapped = torch.searchsorted(ids, blocked)
    valid = (mapped < len(ids)) & (ids[mapped.clamp(max=len(ids) - 1)] == blocked)
    blocked_rows, blocked_columns = torch.where(valid)
    blocked_columns = mapped[blocked_rows, blocked_columns]
    scores = vectors.new_full((len(queries), k), -torch.inf)
    found = torch.zeros((len(queries), k), device=vectors.device, dtype=torch.long)
    for start in range(0, len(ids), chunk_size):
        stop = min(start + chunk_size, len(ids))
        block = queries @ vectors[start:stop].T
        selected = (blocked_columns >= start) & (blocked_columns < stop)
        block[blocked_rows[selected], blocked_columns[selected] - start] = -torch.inf
        values, positions = block.topk(min(k, stop - start), dim=1)
        candidates = ids[start:stop][positions]
        joined_scores = torch.cat((scores, values), dim=1)
        joined_ids = torch.cat((found, candidates), dim=1)
        scores, order = joined_scores.topk(k, dim=1)
        found = joined_ids.gather(1, order)
    return torch.where(torch.isfinite(scores), found, 0).cpu().numpy()


def fuse_tail(main, proposals, counts, quota=5, protect_top=10):
    """Return (fixed-width NumPy candidates, applied quota per query).

    Callers supply only eligible-tail proposals. All main ads with frequency at
    most five and the protected prefix survive; replacements remove the latest
    unprotected nonrare positions and append genuinely new tail ads in order.
    """
    main, proposals, counts = np.asarray(main), np.asarray(proposals), np.asarray(counts)
    if (main.ndim != 2 or not main.size or proposals.ndim != 2 or len(main) != len(proposals)
            or counts.ndim != 1 or not np.issubdtype(main.dtype, np.integer)
            or not np.issubdtype(proposals.dtype, np.integer)
            or main.min() <= 0 or main.max() >= len(counts)
            or proposals.min(initial=0) < 0 or proposals.max(initial=0) >= len(counts)
            or not 0 <= protect_top <= main.shape[1] or not 0 <= quota <= main.shape[1]):
        raise ValueError('Invalid catalog IDs, aligned candidate lists or fusion budget.')
    if (np.diff(np.sort(main, axis=1), axis=1) == 0).any():
        raise ValueError('Main candidates must be unique.')
    valid = proposals > 0
    if ((counts[proposals[valid]] < 1) | (counts[proposals[valid]] > 5)).any():
        raise ValueError('Auxiliary proposals must be tail ads; zero is padding.')
    fused, applied = main.copy(), np.zeros(len(main), dtype=np.int32)
    for row, original in enumerate(main):
        used, novel = set(map(int, original)), []
        for item in proposals[row]:
            if item and int(item) not in used:
                novel.append(int(item))
                used.add(int(item))
        removable = [i for i in range(len(original) - 1, protect_top - 1, -1)
                     if counts[original[i]] > 5]
        take = min(quota, len(novel), len(removable))
        removed = set(removable[:take])
        fused[row] = [int(item) for i, item in enumerate(original) if i not in removed] + novel[:take]
        applied[row] = take
    return fused, applied
