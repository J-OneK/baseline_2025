"""Independent validation-only raw-81 text recall and candidate-union diagnosis.

This channel uses cosine similarity in the published content space. It does not
replace the trained model's raw inner-product score or its persisted index.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PreparedData, SequenceDataset, load_array
from .metrics import exact_topk


EPSILON = 1e-12


def prepare_candidates(raw, available):
    """Keep finite, nonzero content; compact row zero remains padding."""
    if raw.ndim != 2 or available.shape != raw.shape[:1]:
        raise ValueError('Content and availability shapes do not match.')
    norms = torch.linalg.vector_norm(raw, dim=1)
    eligible = available.bool() & torch.isfinite(raw).all(1) & (norms > EPSILON)
    eligible[0] = False
    ids = torch.nonzero(eligible, as_tuple=False).flatten()
    compact_ids = torch.cat([torch.zeros(1, dtype=torch.long, device=raw.device), ids])
    vectors = torch.zeros((len(ids) + 1, raw.shape[1]), dtype=raw.dtype, device=raw.device)
    vectors[1:] = raw[ids] / norms[ids, None]
    mapping = torch.zeros(len(raw), dtype=torch.long, device=raw.device)
    mapping[ids] = torch.arange(1, len(ids) + 1, device=raw.device)
    return eligible, vectors, compact_ids, mapping


def content_queries(raw, eligible, items, actions):
    """Mean observed click content; fallback only to observed exposures."""
    if items.ndim != 2 or actions.shape != items.shape:
        raise ValueError('History items/actions must have the same [batch,length] shape.')
    available = eligible[items] & (items > 0)
    clicks = available & (actions == 2)
    exposures = available & (actions == 1)
    has_click = clicks.any(1)
    selected = torch.where(has_click[:, None], clicks, exposures)
    values = torch.where(selected[:, :, None], raw[items], torch.zeros_like(raw[items]))
    means = values.sum(1) / selected.sum(1).clamp_min(1)[:, None]
    norms = torch.linalg.vector_norm(means, dim=1)
    valid = selected.any(1) & torch.isfinite(means).all(1) & (norms > EPSILON)
    queries = torch.where(valid[:, None], means / norms.clamp_min(EPSILON)[:, None], torch.zeros_like(means))
    source = torch.where(valid, torch.where(has_click, 1, 2), 0).to(torch.uint8)
    return queries, valid, source


def summarize(predictions, targets, rows, train_counts, valid_query, target_eligible, ks):
    """All queries remain in denominators; padding is never a relevant item."""
    predictions, targets, rows = map(np.asarray, (predictions, targets, rows))
    valid_query, target_eligible = np.asarray(valid_query, dtype=bool), np.asarray(target_eligible, dtype=bool)
    ks = sorted(set(map(int, ks)))
    if (predictions.ndim != 2 or not len(predictions) or not ks or ks[0] <= 0
            or ks[-1] > predictions.shape[1] or targets.shape != (len(predictions),)
            or rows.shape != targets.shape or valid_query.shape != targets.shape
            or target_eligible.shape != targets.shape or len(np.unique(rows)) != len(rows)):
        raise ValueError('Invalid predictions, budgets or query identities.')
    if (targets <= 0).any() or (targets >= len(train_counts)).any() or (predictions < 0).any():
        raise ValueError('Targets must be nonpadding catalog IDs; predictions may use padding zero.')
    if (predictions >= len(train_counts)).any() or (predictions[~valid_query] != 0).any():
        raise ValueError('Invalid catalog ID or nonempty predictions for an invalid query.')
    ordered = np.sort(predictions, axis=1)
    if ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] > 0)).any():
        raise ValueError('Duplicate nonpadding predictions.')
    matches = (predictions[:, :ks[-1]] > 0) & (predictions[:, :ks[-1]] == targets[:, None])
    hit = matches.any(1)
    ranks = np.where(hit, matches.argmax(1) + 1, 0).astype(np.int32)
    frequencies = np.asarray(train_counts[targets], dtype=np.int64)
    groups = dict(all=np.ones(len(targets), dtype=bool), cold_0=frequencies == 0,
                  tail_1_5=(frequencies >= 1) & (frequencies <= 5),
                  mid_6_50=(frequencies >= 6) & (frequencies <= 50), head_gt50=frequencies > 50)
    per_user = dict(target_rank=ranks, target_train_frequency=frequencies)
    for k in ks:
        found = (ranks > 0) & (ranks <= k)
        ndcg = np.zeros(len(ranks), dtype=np.float64)
        ndcg[found] = 1 / np.log2(ranks[found] + 1)
        per_user[f'hits_at_{k}'] = found.astype(np.uint8)
        per_user[f'ndcg_at_{k}'] = ndcg
    summary = {}
    for name, selected in groups.items():
        count = int(selected.sum())
        row = dict(users=count, valid_queries=int(valid_query[selected].sum()),
                   targets_in_content_catalog=int(target_eligible[selected].sum()))
        for k in ks:
            row[f'Recall@{k}'] = float(per_user[f'hits_at_{k}'][selected].mean()) if count else None
            row[f'NDCG@{k}'] = float(per_user[f'ndcg_at_{k}'][selected].mean()) if count else None
            row[f'hits_at_{k}'] = int(per_user[f'hits_at_{k}'][selected].sum())
        summary[name] = row
    return summary, per_user


def union_summary(learned, content, targets, rows, train_counts, k):
    """Coverage of two Top-K sets; union budget is up to 2K, not Top-K."""
    with np.load(learned, allow_pickle=False) as source:
        np.testing.assert_array_equal(source['rows'], rows)
        np.testing.assert_array_equal(source['targets'], targets)
        reference = source['predictions'][:, :k]
    if reference.shape[1] < k:
        raise ValueError('Learned predictions are narrower than the requested union budget.')
    learned_hit = (reference == targets[:, None]).any(1)
    content_hit = (content[:, :k] == targets[:, None]).any(1)
    counts = train_counts[targets]
    groups = dict(all=np.ones(len(targets), dtype=bool), cold_0=counts == 0,
                  tail_1_5=(counts >= 1) & (counts <= 5), mid_6_50=(counts >= 6) & (counts <= 50), head_gt50=counts > 50)
    result = {}
    for name, selected in groups.items():
        n = int(selected.sum())
        result[name] = dict(users=n, learned_hits=int(learned_hit[selected].sum()),
                            content_hits=int(content_hit[selected].sum()),
                            union_hits=int((learned_hit | content_hit)[selected].sum()),
                            additional_hits_from_content=int((~learned_hit & content_hit)[selected].sum()),
                            union_recall=float((learned_hit | content_hit)[selected].mean()) if n else None)
    return dict(learned_predictions=str(Path(learned).resolve()), per_channel_k=k,
                maximum_union_candidates=2*k, union_is_unranked=True, groups=result)


@torch.inference_mode()
def diagnose(args):
    output = Path(args.output)
    if output.with_suffix('.json').exists() or output.with_suffix('.npz').exists():
        raise FileExistsError('Content diagnostic output exists; use a new prefix.')
    ks = sorted(set(args.ks))
    if not ks or min(ks) <= 0:
        raise ValueError('Provide positive K values.')
    started = time.monotonic()
    torch.set_num_threads(4)
    data = PreparedData(args.cache)
    data.set_eval_indices(np.load(args.cohort, allow_pickle=False))
    raw = torch.as_tensor(np.array(load_array(data.root, 'item_mm81')), device=args.device)
    available = torch.as_tensor(np.array(load_array(data.root, 'mm_present')), device=args.device)
    eligible, vectors, compact_ids, mapping = prepare_candidates(raw, available)
    if len(vectors) - 1 < max(ks) + data.max_history:
        raise ValueError('Content catalog is too small for K plus worst-case history filtering.')
    loader = DataLoader(SequenceDataset(data, 'val'), batch_size=args.batch_size, shuffle=False)
    saved = {name: [] for name in ['rows', 'targets', 'predictions', 'scores', 'queries', 'seen', 'valid_query', 'query_source']}
    for index, batch in enumerate(loader):
        items = batch['items'].to(args.device)
        actions = batch['actions'].to(args.device)
        if (items.cpu() == batch['target'][:, None]).any():
            raise AssertionError('Validation target appears in visible history.')
        query, valid, source = content_queries(raw, eligible, items, actions)
        found = torch.zeros((len(items), max(ks)), dtype=torch.long, device=args.device)
        scores = torch.full((len(items), max(ks)), -torch.inf, device=args.device)
        if valid.any():
            selected = exact_topk(query[valid], vectors, mapping[items[valid]], k=max(ks), chunk_size=args.candidate_chunk)
            found[valid] = compact_ids[selected]
            scores[valid] = torch.einsum('bd,bkd->bk', query[valid], vectors[selected])
        values = dict(rows=batch['row'], targets=batch['target'], predictions=found.to(torch.int32).cpu(),
                      scores=scores.cpu(), queries=query.cpu(), seen=items.to(torch.int32).cpu(),
                      valid_query=valid.cpu(), query_source=source.cpu())
        for name, value in values.items():
            saved[name].append(value.numpy())
        if index % 10 == 0:
            print(json.dumps(dict(batch=index, completed=sum(len(a) for a in saved['rows']),
                                  seconds=round(time.monotonic() - started, 2))), flush=True)
    saved = {name: np.concatenate(values) for name, values in saved.items()}
    target_eligible = eligible.cpu().numpy()[saved['targets']]
    summary, per_user = summarize(saved['predictions'], saved['targets'], saved['rows'], data.train_counts,
                                   saved['valid_query'], target_eligible, ks)
    saved.update(per_user, target_content_eligible=target_eligible)
    result = dict(split='val', cache=str(data.root.resolve()), cache_signature=data.metadata['preparation_config'],
                  cohort=str(Path(args.cohort).resolve()),
                  evaluation_cohort_sha256=hashlib.sha256(np.asarray(data.eval_indices, dtype=np.int64).tobytes()).hexdigest(),
                  evaluation_users=len(saved['targets']), complete_catalog_items=data.num_items,
                  content_catalog_items=len(vectors)-1, ks=ks, groups=summary,
                  query_source_counts={name: int((saved['query_source'] == code).sum())
                                       for code, name in [(0, 'no_usable_query'), (1, 'observed_click_mean'), (2, 'observed_exposure_fallback')]},
                  configuration=dict(channel='independent raw embedding-81 cosine', query='mean raw click vectors, then L2 normalize',
                                     fallback='mean observed exposure vectors if no eligible observed click',
                                     unknown_actions_used=False, eligible_candidate='mm_present and all-finite and nonzero norm; padding excluded',
                                     history_filter='all observed items regardless of action/content',
                                     invalid_query='empty predictions; remains in all metric denominators',
                                     epsilon=EPSILON, device=args.device),
                  elapsed_seconds=time.monotonic()-started, test_targets_read=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.learned_predictions:
        result['candidate_union'] = union_summary(args.learned_predictions, saved['predictions'], saved['targets'],
                                                   saved['rows'], data.train_counts, max(ks))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'), **saved)
    with output.with_suffix('.json').open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--cache', default='artifacts/objective_study/scale_data')
    result.add_argument('--cohort', default='artifacts/objective_study/confirm_eval_indices.npy')
    result.add_argument('--output', default='artifacts/cold_start/diagnostics/content_mean81')
    result.add_argument('--learned-predictions', default='artifacts/cold_start/diagnostics/click_seed2026.npz')
    result.add_argument('--device', default='cuda:6')
    result.add_argument('--ks', nargs='+', type=int, default=[10, 100, 500])
    result.add_argument('--batch-size', type=int, default=128)
    result.add_argument('--candidate-chunk', type=int, default=32768)
    return result


if __name__ == '__main__':
    diagnose(parser().parse_args())
