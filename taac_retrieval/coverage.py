"""Validation-only exact retrieval coverage at several candidate budgets.

Ranks and bucket membership use local catalog IDs. A saved target_rank of zero
means the target is outside the largest requested Top-K, not rank zero.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SequenceDataset, load_array
from .evaluate_checkpoint import load_model
from .metrics import exact_topk, to_device


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def coverage_summary(predictions, targets, rows, train_counts, ks=(10, 100, 500), content_present=None):
    """Summarize one-target Recall/NDCG, without changing candidate budgets."""
    predictions = np.asarray(predictions)
    targets, rows, train_counts = map(np.asarray, (targets, rows, train_counts))
    ks = sorted(set(int(k) for k in ks))
    if not ks or ks[0] <= 0 or predictions.ndim != 2 or not len(predictions):
        raise ValueError('Expected nonempty ranked predictions and positive K values.')
    if ks[-1] > predictions.shape[1]:
        raise ValueError('Requested K exceeds saved retrieval width; budgets are never clipped.')
    if targets.shape != (len(predictions),) or rows.shape != targets.shape:
        raise ValueError('Targets and row IDs must have one value per query.')
    if len(np.unique(rows)) != len(rows):
        raise ValueError('Duplicate evaluation row IDs.')
    if (not np.issubdtype(predictions.dtype, np.integer)
            or not np.issubdtype(targets.dtype, np.integer)
            or predictions.min() <= 0 or targets.min() <= 0
            or predictions.max() >= len(train_counts) or targets.max() >= len(train_counts)):
        raise ValueError('Predictions and targets must be nonpadding local catalog IDs.')
    ranked = predictions[:, :ks[-1]]
    ordered = np.sort(ranked, axis=1)
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise ValueError('Duplicate predicted IDs within a query.')
    matches = ranked == targets[:, None]
    present = matches.any(1)
    ranks = np.where(present, matches.argmax(1) + 1, 0).astype(np.int32)
    frequencies = np.asarray(train_counts[targets], dtype=np.int64)
    if (frequencies < 0).any():
        raise ValueError('Training frequency cannot be negative.')
    groups = dict(all=np.ones(len(targets), dtype=bool), cold_0=frequencies == 0,
                  tail_1_5=(frequencies >= 1) & (frequencies <= 5),
                  mid_6_50=(frequencies >= 6) & (frequencies <= 50),
                  head_gt50=frequencies > 50)
    if content_present is not None:
        content_present = np.asarray(content_present, dtype=bool)
        if content_present.shape != targets.shape:
            raise ValueError('Content availability must align with query targets.')
        groups.update(content_present=content_present, content_missing=~content_present)
    per_user = dict(target_rank=ranks, target_train_frequency=frequencies)
    for k in ks:
        hit = (ranks > 0) & (ranks <= k)
        ndcg = np.zeros(len(ranks), dtype=np.float64)
        ndcg[hit] = 1 / np.log2(ranks[hit] + 1)
        per_user[f'hits_at_{k}'] = hit.astype(np.uint8)
        per_user[f'ndcg_at_{k}'] = ndcg
    result = {}
    for name, mask in groups.items():
        count = int(mask.sum())
        row = dict(users=count, targets_in_max_k=int((present & mask).sum()))
        for k in ks:
            hits = per_user[f'hits_at_{k}'][mask]
            row[f'Recall@{k}'] = float(hits.mean()) if count else None
            row[f'NDCG@{k}'] = float(per_user[f'ndcg_at_{k}'][mask].mean()) if count else None
            row[f'hits_at_{k}'] = int(hits.sum())
        in_largest = present & mask
        row[f'conditional_Recall@{ks[0]}_given_Top{ks[-1]}'] = (
            float(per_user[f'hits_at_{ks[0]}'][in_largest].mean()) if in_largest.any() else None)
        lower = 1
        bands = {}
        for k in ks:
            bands[f'{lower}_{k}'] = int(((ranks >= lower) & (ranks <= k) & mask).sum())
            lower = k + 1
        bands[f'outside_{ks[-1]}'] = int((~present & mask).sum())
        row['target_rank_bands'] = bands
        result[name] = row
    return result, per_user


@torch.inference_mode()
def diagnose(args):
    output = Path(args.output)
    json_path, npz_path = output.with_suffix('.json'), output.with_suffix('.npz')
    if json_path.exists() or npz_path.exists():
        raise FileExistsError('Coverage output exists; use a new prefix to preserve prior diagnostics.')
    ks = sorted(set(int(k) for k in args.ks))
    if not ks or min(ks) <= 0:
        raise ValueError('Provide positive retrieval budgets.')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    model, data, checkpoint = load_model(args.checkpoint, args.cache, args.device)
    if data.num_items < max(ks):
        raise ValueError('Catalog is smaller than requested K; budgets are never clipped.')
    vectors = torch.empty(data.num_items + 1, model.hidden, device=args.device)
    for start in range(0, len(vectors), 8192):
        end = min(start + 8192, len(vectors))
        vectors[start:end] = model.encode_items(torch.arange(start, end, device=args.device))
    if not torch.isfinite(vectors).all():
        raise FloatingPointError('Non-finite candidate vectors.')
    loader = DataLoader(SequenceDataset(data, 'val'), batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=str(args.device).startswith('cuda'))
    saved = {name: [] for name in ['predictions', 'scores', 'queries', 'seen', 'targets', 'rows']}
    for index, cpu_batch in enumerate(loader):
        batch = to_device(cpu_batch, args.device)
        queries = model.predict(batch)
        if not torch.isfinite(queries).all():
            raise FloatingPointError('Non-finite query vectors.')
        if (batch['items'] == batch['target'][:, None]).any():
            raise AssertionError('Validation target leaked into visible history.')
        found = exact_topk(queries, vectors, batch['items'], k=max(ks), chunk_size=args.candidate_chunk)
        scores = torch.einsum('bd,bkd->bk', queries, vectors[found])
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Non-finite selected scores.')
        saved['predictions'].append(found.cpu().numpy().astype(np.int32))
        saved['scores'].append(scores.cpu().numpy())
        saved['queries'].append(queries.cpu().numpy())
        saved['seen'].append(cpu_batch['items'].numpy().astype(np.int32))
        saved['targets'].append(cpu_batch['target'].numpy().astype(np.int32))
        saved['rows'].append(cpu_batch['row'].numpy().astype(np.int32))
        if index % 10 == 0:
            print(json.dumps(dict(batch=index, queries_completed=sum(len(a) for a in saved['rows']),
                                  seconds=round(time.monotonic() - started, 2))), flush=True)
    saved = {name: np.concatenate(values) for name, values in saved.items()}
    available = load_array(data.root, 'mm_present')[saved['targets']]
    summary, per_user = coverage_summary(saved['predictions'], saved['targets'], saved['rows'],
                                         data.train_counts, ks, available)
    saved.update(per_user, target_content_present=np.asarray(available, dtype=bool))
    config = checkpoint['config']
    result = dict(split='val', checkpoint=str(Path(args.checkpoint).resolve()),
                  checkpoint_sha256=file_digest(args.checkpoint), selected_epoch=checkpoint['epoch'],
                  training=config, cache=str(Path(args.cache).resolve()), cache_signature=checkpoint['cache_signature'],
                  evaluation_cohort_sha256=hashlib.sha256(np.asarray(data.eval_indices, dtype=np.int64).tobytes()).hexdigest(),
                  evaluation_users=len(saved['targets']), candidate_items=data.num_items,
                  candidate_coverage=1.0, filter_seen_items=True, metric='raw inner product; exact full-catalog retrieval',
                  ks=ks, groups=summary, elapsed_seconds=time.monotonic() - started,
                  code_sha256={str(p.name): file_digest(p) for p in [Path(__file__), Path(__file__).with_name('metrics.py')]},
                  rank_zero_means=f'outside Top-{max(ks)}; not an exact rank beyond that budget',
                  frequency_definition='all events in all user training prefixes, including exposure/click/unknown',
                  test_targets_read=False)
    np.savez_compressed(npz_path, **saved)
    with json_path.open('x') as destination:
        destination.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--checkpoint', required=True)
    result.add_argument('--cache', required=True)
    result.add_argument('--output', required=True, help='New prefix for .json and .npz')
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--ks', nargs='+', type=int, default=[10, 100, 500])
    result.add_argument('--batch-size', type=int, default=128)
    result.add_argument('--candidate-chunk', type=int, default=32768)
    return result


if __name__ == '__main__':
    diagnose(parser().parse_args())
