"""Run the real retrieval modules on a small, explicitly synthetic dataset.

This is a CPU teaching demonstration, not a benchmark or a reproduction of the
large confirmation experiment. It writes to a fresh output directory only.
"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from taac_retrieval.ann import search_filtered
from taac_retrieval.benchmark_ann import benchmark
from taac_retrieval.cohorts import freeze_cohort
from taac_retrieval.data import PreparedData, SequenceDataset, sample_negatives
from taac_retrieval.evaluate_checkpoint import evaluate_checkpoint, load_model
from taac_retrieval.export import export
from taac_retrieval.metrics import exact_topk
from taac_retrieval.objectives import sampled_info_nce, supervision_weights
from taac_retrieval.prepare import (ITEM_FIELDS, USER_FIELDS, USER_ARRAY_FIELDS,
                                    clean_and_split, prepare)
from taac_retrieval.recommend import recommend
from taac_retrieval.train import parser as training_parser, train


def serializable(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def source_identity():
    paths = ['taac_retrieval/prepare.py', 'taac_retrieval/cohorts.py',
             'taac_retrieval/evaluate_checkpoint.py', 'taac_retrieval/data.py',
             'taac_retrieval/model.py', 'model.py', 'taac_retrieval/train.py',
             'taac_retrieval/objectives.py', 'taac_retrieval/metrics.py',
             'taac_retrieval/export.py', 'taac_retrieval/ann.py',
             'taac_retrieval/benchmark_ann.py', 'taac_retrieval/recommend.py',
             'scripts/run_tutorial_demo.py']
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in paths}


def write_synthetic_source(root):
    """Write actual Parquet schemas and mapping dictionaries consumed by prepare."""
    for folder in ['seq', 'item_feat', 'user_feat', 'candidate', 'mm_emb/emb_81_32_parquet']:
        (root / folder).mkdir(parents=True)
    raw_actions = [0, 1, None, 0, 0, 1, 1, 0, 1, 0]
    rows = []
    for u in range(12):
        base = u * 9
        item_ids = [base + x for x in [1, 2, 3, 4, 5, 6, 6, 7, 8, 9]]
        if u == 1:
            # Another user's training history makes user 1's targets globally warm.
            item_ids[:2] = [6, 8]
        events = [dict(item_id=i, timestamp=(u * 1000 + (j + 1) * 10), action_type=a)
                  for j, (i, a) in enumerate(zip(item_ids, raw_actions))]
        rows.append(dict(user_id=u + 1, seq=events))
    event_type = pa.struct([('item_id', pa.int64()), ('action_type', pa.int32()),
                            ('timestamp', pa.int64())])
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([
        ('user_id', pa.int64()), ('seq', pa.list_(event_type))])), root / 'seq/part.parquet')
    users = []
    for uid in range(1, 13):
        row = dict(user_id=uid)
        row.update({f: (uid + j) % 4 + 1 for j, f in enumerate(USER_FIELDS)})
        row.update({f: ([uid % 4 + 1, (uid + 1) % 4 + 1] if j % 2 == 0 else None)
                    for j, f in enumerate(USER_ARRAY_FIELDS)})
        users.append(row)
    user_schema = pa.schema([('user_id', pa.int64())]
                            + [(f, pa.int64()) for f in USER_FIELDS]
                            + [(f, pa.list_(pa.int64())) for f in USER_ARRAY_FIELDS])
    pq.write_table(pa.Table.from_pylist(users, schema=user_schema), root / 'user_feat/part.parquet')
    item_ids = sorted({e['item_id'] for row in rows for e in row['seq']})
    items = [dict(item_id=rid, **{f: (rid + j) % 4 + 1 for j, f in enumerate(ITEM_FIELDS)})
             for rid in item_ids]
    pq.write_table(pa.Table.from_pylist(items), root / 'item_feat/part.parquet')
    oid_for = {rid: 20_000_000_000 + rid for rid in item_ids}
    rng = np.random.default_rng(2026)
    contents = []
    for rid in item_ids:
        vector = rng.normal(0, .1, 32)
        vector[rid % 4] += 1
        contents.append(dict(anonymous_cid=str(oid_for[rid]), emb=vector.tolist()))
    pq.write_table(pa.Table.from_pylist(contents), root / 'mm_emb/emb_81_32_parquet/part.parquet')
    candidates = [dict(item_id=oid_for[rid], retrieval_id=j,
                       **{'100': dict(cold_start=0, feature_value=str((rid % 4) + 1))})
                  for j, rid in enumerate(item_ids)]
    pq.write_table(pa.Table.from_pylist(candidates), root / 'candidate/part.parquet')
    indexer = dict(i={oid: rid for rid, oid in oid_for.items()},
                   u={f'synthetic_user_{uid}': uid for uid in range(1, 13)},
                   f={field: {f'synthetic_value_{j}': j for j in range(1, 5)}
                      for field in ITEM_FIELDS + USER_FIELDS + USER_ARRAY_FIELDS}, a={})
    with (root / 'indexer.pkl').open('wb') as handle:
        pickle.dump(indexer, handle)
    return dict(first_user=rows[0], second_user=rows[1], first_item=items[0],
                first_user_features=users[0], first_content=contents[0],
                first_candidate=candidates[0], oid_rid_pairs=list(indexer['i'].items())[:9])


def trace_checkpoint(root, cache, checkpoint, cohort):
    model, data, saved = load_model(checkpoint, cache, 'cpu')
    first_row = int(np.flatnonzero(data.user_rids == 1)[0])
    assert first_row in data.eval_indices, 'Teaching user must belong to the frozen cohort.'
    cohort_position = int(np.flatnonzero(data.eval_indices == first_row)[0])
    full_train = SequenceDataset(data, 'train')
    samples = {'train': full_train[first_row],
               'val': SequenceDataset(data, 'val')[cohort_position],
               'test': SequenceDataset(data, 'test')[cohort_position]}
    batch = {key: torch.as_tensor(value).unsqueeze(0) for key, value in samples['train'].items()}
    mask = ((batch['positive'] != 0)
            & (supervision_weights(batch['target_actions'], 'click_infonce') > 0))
    states = model.encode_history(batch['items'], batch['kinds'], batch['actions'], batch['user'])
    selected = states[mask]
    pool = torch.tensor(np.flatnonzero(data.train_counts), dtype=torch.long)
    rows = torch.where(mask)[0]
    negatives = sample_negatives(pool, batch['seen'][rows], (int(mask.sum()), 64),
                                 torch.Generator().manual_seed(2043))
    positive_scores = (selected * model.encode_items(batch['positive'][mask])).sum(-1)
    negative_scores = torch.einsum('md,mkd->mk', selected, model.encode_items(negatives))
    loss = sampled_info_nce(positive_scores, negative_scores)
    probabilities = torch.softmax(torch.cat([positive_scores[:, None], negative_scores], -1), -1)
    model.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {name: (None if p.grad is None else float(p.grad.norm()))
                 for name, p in model.named_parameters()
                 if name in ['item_emb.weight', 'user_emb.weight', 'mm_projection.weight',
                             'item_dnn.weight', 'attention.0.q_linear.weight', 'action_emb.weight']}
    model.zero_grad(set_to_none=True)
    valid = batch['kinds'] != 0
    length = batch['items'].shape[1]
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    allowed = (causal.unsqueeze(0) & valid.unsqueeze(1)) | (
        (~valid).unsqueeze(-1) & torch.eye(length, dtype=torch.bool).unsqueeze(0))
    with torch.inference_mode():
        vectors = model.encode_items(torch.arange(data.num_items + 1))
        val_batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in samples['val'].items()}
        query = model.predict(val_batch)
        top10 = exact_topk(query, vectors, val_batch['items'], k=10, chunk_size=32)
        raw_scores = (query @ vectors.T)[0]
        raw_scores[0] = -torch.inf
        raw_scores[val_batch['items'][0]] = -torch.inf
        target = int(val_batch['target'][0])
        rank = int((raw_scores > raw_scores[target]).sum()) + 1
    config = json.loads((root / 'model/config.json').read_text())
    return serializable(dict(
        note='Actual checkpoint computations on synthetic data. For the isolated gradient trace, '
             'model.eval() disables dropout; five-epoch training itself used model.train(). '
             'No optimizer step is applied to this saved checkpoint during tracing.',
        cohort_position=cohort_position, row=first_row, seed=config['seed'],
        best_epoch=saved['epoch'], samples=samples,
        item_id_lookup_first10=data.item_id_lookup[:10] if hasattr(data, 'item_id_lookup') else model.item_id_lookup[:10],
        first_catalog_oids=data.catalog_oids[:10], first_train_counts=data.train_counts[:10],
        catalog_rids=data.catalog_rids, catalog_oids=data.catalog_oids,
        shapes=dict(sequence=list(states.shape), selected=list(selected.shape),
                    negatives=list(negatives.shape), positive_logits=list(positive_scores.shape),
                    negative_logits=list(negative_scores.shape), query=list(query.shape),
                    candidate_vectors=list(vectors.shape)),
        supervised_positions=torch.where(mask)[1], negative_ids=negatives,
        positive_scores=positive_scores, negative_scores=negative_scores,
        positive_softmax_probability=probabilities[:, 0], loss=float(loss.detach()),
        gradient_norms=gradients, position_ids=valid.long().cumsum(1) * valid,
        attention_allowed=allowed[0].int(), query_vector=query[0],
        target_local_id=target, target_score=float(raw_scores[target]), target_rank=rank,
        exact_top10_local=top10[0], exact_top10_oids=data.catalog_oids[top10[0].numpy()],
        exact_top10_scores=raw_scores[top10[0]],
        candidate_target_vector=vectors[target], history_candidate_vectors=vectors[[1,2,3,4,5]],
        frozen_indices=np.load(cohort)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/tutorial_demo')
    parser.add_argument('--trace-output', default=None)
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    root = (root if root.is_absolute() else REPO / root).resolve()
    if root.exists():
        raise FileExistsError('Use a fresh --root; this teaching run never overwrites prior outputs.')
    root.mkdir(parents=True)
    raw = write_synthetic_source(root / 'source')
    output_log = root / 'execution.log'
    with output_log.open('w', buffering=1) as log, redirect_stdout(log):
        # Smaller source/cohort sizes are declared demo settings, not main.json.
        reference = prepare(SimpleNamespace(data_dir=str(root / 'source'), output=str(root / 'reference_data'),
                           max_users=6, seed=2027, max_history=10, min_history=5, eval_users=2))
        metadata = prepare(SimpleNamespace(data_dir=str(root / 'source'), output=str(root / 'data'),
                          max_users=12, seed=2026, max_history=10, min_history=5, eval_users=6))
        cohort = root / 'cohort.npy'
        cohort_meta = freeze_cohort(root / 'data', root / 'reference_data', cohort, count=6, seed=3107)
        training = training_parser().parse_args([
            '--cache', str(root / 'data'), '--output', str(root / 'model'),
            '--device', 'cpu', '--seed', '2026', '--epochs', '5', '--batch-size', '4',
            '--workers', '0', '--eval-batch-size', '6', '--candidate-chunk', '32',
            '--objective', 'click_infonce', '--num-negatives', '64', '--skip-test',
            '--eval-indices', str(cohort)])
        result = train(training)
        checkpoint = root / 'model/best.pt'
        evaluate_checkpoint(SimpleNamespace(output=str(root / 'evaluation'), checkpoint=str(checkpoint),
                            cache=str(root / 'data'), device='cpu', split='test'))
        exported = export(SimpleNamespace(output=str(root / 'export'), checkpoint=str(checkpoint),
                          cache=str(root / 'data'), device='cpu'))
        index_config = dict(faiss=dict(primary_seed=2026, metric='inner_product_without_normalization',
                            M=16, efConstruction=100, efSearch=[64,128,2048],
                            minimum_validation_ann_recall=.98, latency_queries=6, warmup_queries=2,
                            latency_repeats=1, latency_threads=1, build_threads=1, hnsw_geometry='mips_l2'))
        (root / 'index_config.json').write_text(json.dumps(index_config, indent=2) + '\n')
        indexed = benchmark(SimpleNamespace(export_dir=str(root / 'export'), output=str(root / 'ann'),
                            config=str(root / 'index_config.json')))
        trace = trace_checkpoint(root, root / 'data', checkpoint, cohort)
        demo_result = recommend(SimpleNamespace(checkpoint=str(checkpoint), cache=str(root / 'data'),
                                export_dir=str(root / 'export'), index_dir=str(root / 'ann'),
                                device='cpu', split='val', index=trace['cohort_position']))
    evaluation = json.loads((root / 'evaluation.json').read_text())
    history = json.loads((root / 'model/history.json').read_text())
    summary = serializable(dict(
        purpose='Executable synthetic teaching trace; not TencentGR-1M performance evidence.',
        executed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_sha256=source_identity(), environment=dict(python=sys.version.split()[0],
            torch=torch.__version__, numpy=np.__version__, faiss=faiss.__version__),
        settings=dict(users=12, sampled_reference_users=6, reference_seed=2027, reference_eval_users=2,
                      evaluation_users=6, seed=2026, max_history=10, hidden=32, blocks=1,
                      heads=1, epochs=5, negatives=64, batch_size=4, device='cpu',
                      hnsw_M=16, hnsw_efSearch_candidates=[64,128,2048]),
        raw=raw, clean_and_split=clean_and_split(raw['first_user']['seq'], max_history=10),
        prepared_metadata=metadata, reference_metadata=reference, cohort=cohort_meta,
        training=result, history=history, evaluation=evaluation,
        export=exported, index=indexed, trace=trace, recommendation=demo_result))
    assert len(demo_result['recommendations']) == 10
    assert trace['samples']['val']['target'] == 6 and trace['samples']['test']['target'] == 8
    assert trace['supervised_positions'] == [7]
    assert indexed['ids_and_inner_product_independently_verified']
    assert cohort_meta['overlap_with_previous_test_users'] == 0
    destination = Path(args.trace_output) if args.trace_output else root / 'trace.json'
    if not destination.is_absolute():
        destination = REPO / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(trace=str(destination), log=str(output_log),
                         users=metadata['users'], items=metadata['catalog_items'],
                         epochs=len(history), best_epoch=result['best_epoch'],
                         actual_recommendation=demo_result,
                         validation_ann_overlap=indexed['selected']['ANNRecall@10']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
