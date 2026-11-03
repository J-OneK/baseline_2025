"""Stream Parquet into compact arrays and construct a conservative click holdout."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

ITEM_FIELDS = ['100', '117', '118', '101', '102', '119', '120',
               '114', '112', '121', '115', '122', '116']
USER_FIELDS = ['103', '104', '105', '109']
USER_ARRAY_FIELDS = ['106', '107', '108', '110']
PROTOCOL_VERSION = 2


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def clean_and_split(events, max_history=100, min_history=5):
    """Last two first-appearance clicks, with strictly earlier histories.

    A click whose item has already appeared in this user's sequence is not a
    holdout candidate. This conservatively avoids exposing the held-out ad in
    an earlier impression when no impression/click attribution ID is provided.
    Unknown actions remain in context (code 3), but are not training targets.
    Codes: 0 padding/user, 1 exposure, 2 click, 3 unknown.
    """
    clean = [e for e in (events or []) if e is not None
             and e.get('item_id') is not None and int(e['item_id']) > 0
             and e.get('timestamp') is not None]
    clean.sort(key=lambda e: int(e['timestamp']))  # stable for timestamp ties
    clean = clean[-max_history:]
    seen, eligible = set(), []
    times = np.array([int(e['timestamp']) for e in clean], dtype=np.int64)
    for pos, event in enumerate(clean):
        item = int(event['item_id'])
        cut = int(np.searchsorted(times, times[pos], side='left'))
        if event.get('action_type') == 1 and item not in seen and cut >= min_history:
            eligible.append((pos, cut))
        seen.add(item)
    if len(eligible) < 2:
        return None
    test_pos, test_cut = eligible[-1]
    earlier = [(p, c) for p, c in eligible[:-1] if times[p] < times[test_pos]]
    if not earlier:
        return None
    val_pos, val_cut = earlier[-1]
    # Targets and all later events are absent from the training prefix.
    return {
        'items': [int(e['item_id']) for e in clean],
        'actions': [1 if e.get('action_type') == 0 else
                    2 if e.get('action_type') == 1 else 3 for e in clean],
        'times': times, 'train_cut': val_cut, 'val_cut': val_cut,
        'test_cut': test_cut, 'val_pos': val_pos, 'test_pos': test_pos,
    }


def source_signature(root):
    entries = []
    for folder in ['seq', 'item_feat', 'user_feat', 'candidate', 'mm_emb/emb_81_32_parquet']:
        for path in sorted((root / folder).glob('*.parquet')):
            stat = path.stat()
            entries.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    path = root / 'indexer.pkl'
    entries.append(('indexer.pkl', path.stat().st_size, path.stat().st_mtime_ns))
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def save_array(root, name, array):
    np.save(root / f'{name}.npy', array, allow_pickle=False)


def prepare(args):
    root, out = Path(args.data_dir).resolve(), Path(args.output).resolve()
    signature = source_signature(root)
    config = {k: getattr(args, k) for k in
              ['max_users', 'seed', 'max_history', 'min_history', 'eval_users']}
    config.update(protocol_version=PROTOCOL_VERSION, source_signature=signature)
    if (out / 'metadata.json').exists():
        old = json.loads((out / 'metadata.json').read_text())
        if old['preparation_config'] != config:
            raise ValueError('Cache settings/source differ; choose a new --output directory.')
        log(f"Reusing completed cache: {out}")
        return old
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rng = np.random.default_rng(args.seed)
    users_table = ds.dataset(root / 'user_feat', format='parquet')
    all_users = users_table.to_table(columns=['user_id'])['user_id'].to_numpy()
    sampled = np.sort(rng.choice(all_users, min(args.max_users, len(all_users)), replace=False))
    selected = np.zeros(int(all_users.max()) + 1, dtype=bool)
    selected[sampled] = True
    del all_users
    user_ids, records = [], []
    audit = Counter()
    sequences = ds.dataset(root / 'seq', format='parquet')
    log(f'Sampling {len(sampled)} users uniformly, scanning chronological sequences')
    for batch in sequences.scanner(columns=['user_id', 'seq'], batch_size=4096).to_batches():
        uids = batch.column('user_id').to_numpy()
        batch = batch.filter(pa.array(selected[uids]))
        for row in batch.to_pylist():
            audit['sampled_users'] += 1
            source_events = row['seq'] or []
            audit['raw_events'] += len(source_events)
            audit['null_actions'] += sum(e is not None and e.get('action_type') is None for e in source_events)
            split = clean_and_split(source_events, args.max_history, args.min_history)
            if split is None:
                audit['excluded_insufficient_eligible_clicks'] += 1
                continue
            user_ids.append(int(row['user_id']))
            records.append(split)
        if audit['sampled_users'] and audit['sampled_users'] // 20000 > audit.get('reported', 0):
            audit['reported'] = audit['sampled_users'] // 20000
            log(f"Read {audit['sampled_users']} sampled users; retained {len(records)}")
    if not records:
        raise ValueError('No eligible users; increase --max-users or lower --min-history.')
    order = np.argsort(user_ids)
    user_ids = np.asarray(user_ids, dtype=np.int32)[order]
    records = [records[i] for i in order]
    n = len(records)
    items = np.zeros((n, args.max_history), dtype=np.int32)
    actions = np.zeros_like(items, dtype=np.uint8)
    times = np.zeros_like(items, dtype=np.int64)
    vectors = {name: np.zeros(n, dtype=np.int32) for name in
               ['lengths', 'train_cut', 'val_cut', 'test_cut', 'val_pos', 'test_pos']}
    for i, record in enumerate(records):
        length = len(record['items'])
        items[i, :length] = record['items']
        actions[i, :length] = record['actions']
        times[i, :length] = record['times']
        vectors['lengths'][i] = length
        for name in vectors:
            if name != 'lengths':
                vectors[name][i] = record[name]
    del records
    catalog_rids = np.unique(items)
    if catalog_rids[0] != 0:
        catalog_rids = np.r_[np.int32(0), catalog_rids]
    # Local IDs are compact, independent of original integer/string OID types.
    rid_to_local = np.zeros(int(catalog_rids.max()) + 1, dtype=np.int32)
    rid_to_local[catalog_rids] = np.arange(len(catalog_rids), dtype=np.int32)
    items = rid_to_local[items]
    train_counts = np.zeros(len(catalog_rids), dtype=np.int64)
    for i, cut in enumerate(vectors['train_cut']):
        np.add.at(train_counts, items[i, :cut], 1)
    train_counts[0] = 0
    save_array(out, 'items', items)
    save_array(out, 'actions', actions)
    save_array(out, 'timestamps', times)
    save_array(out, 'user_rids', user_ids)
    save_array(out, 'catalog_rids', catalog_rids)
    save_array(out, 'train_counts', train_counts)
    id_lookup = np.zeros(len(catalog_rids), dtype=np.int32)
    id_lookup[train_counts > 0] = np.arange(1, np.count_nonzero(train_counts) + 1, dtype=np.int32)
    save_array(out, 'item_id_lookup', id_lookup)
    for name, value in vectors.items():
        save_array(out, name, value)
    eval_indices = np.sort(np.random.default_rng(args.seed + 1).choice(
        n, min(args.eval_users, n), replace=False)).astype(np.int32)
    save_array(out, 'eval_indices', eval_indices)
    log(f'{n} users, {len(catalog_rids)-1} catalog items; building feature arrays')
    with (root / 'indexer.pkl').open('rb') as handle:
        indexer = pickle.load(handle)
    feature_sizes = {k: len(indexer['f'][k]) + 1 for k in
                     ITEM_FIELDS + USER_FIELDS + USER_ARRAY_FIELDS}
    oids = np.zeros(len(catalog_rids), dtype=np.int64)
    for oid, rid in indexer['i'].items():
        if 0 < rid < len(rid_to_local) and rid_to_local[rid] > 0:
            oids[rid_to_local[rid]] = int(oid)
    assert np.all(oids[1:] > 0)
    del indexer
    save_array(out, 'catalog_oids', oids)
    item_sparse = np.zeros((len(catalog_rids), len(ITEM_FIELDS)), dtype=np.int32)
    table = ds.dataset(root / 'item_feat', format='parquet')
    for batch in table.scanner(columns=['item_id'] + ITEM_FIELDS, batch_size=65536).to_batches():
        rids = batch.column('item_id').to_numpy()
        keep = rids < len(rid_to_local)
        local = np.zeros(len(rids), dtype=np.int32)
        local[keep] = rid_to_local[rids[keep]]
        keep = local > 0
        for j, field in enumerate(ITEM_FIELDS):
            item_sparse[local[keep], j] = pc.fill_null(batch.column(field), 0).to_numpy()[keep]
    field_oov = {}
    for j, field in enumerate(ITEM_FIELDS):
        # Fit each categorical vocabulary only on training-visible items.
        vocabulary = np.unique(np.r_[np.int32(0), item_sparse[train_counts > 0, j]])
        pos = np.searchsorted(vocabulary, item_sparse[:, j])
        bounded = np.minimum(pos, len(vocabulary) - 1)
        known = (pos < len(vocabulary)) & (vocabulary[bounded] == item_sparse[:, j])
        field_oov[field] = float((~known[1:]).mean())
        item_sparse[:, j] = np.where(known, pos, 0)
        feature_sizes[field] = len(vocabulary)
        save_array(out, f'item_vocabulary_{field}', vocabulary)
    save_array(out, 'item_sparse', item_sparse)
    del item_sparse
    user_lookup = np.zeros(len(selected), dtype=np.int32)
    user_lookup[user_ids] = np.arange(1, n + 1, dtype=np.int32)
    user_sparse = np.zeros((n + 1, len(USER_FIELDS)), dtype=np.int32)
    user_arrays = {field: {} for field in USER_ARRAY_FIELDS}
    for batch in users_table.scanner(columns=['user_id'] + USER_FIELDS + USER_ARRAY_FIELDS,
                                     batch_size=16384).to_batches():
        local = user_lookup[batch.column('user_id').to_numpy()]
        keep = local > 0
        for j, field in enumerate(USER_FIELDS):
            user_sparse[local[keep], j] = pc.fill_null(batch.column(field), 0).to_numpy()[keep]
        for row in batch.filter(pa.array(keep)).to_pylist():
            uid = user_lookup[row['user_id']]
            for field in USER_ARRAY_FIELDS:
                user_arrays[field][uid] = [int(v) for v in (row[field] or []) if v is not None]
    save_array(out, 'user_sparse', user_sparse)
    for field, values in user_arrays.items():
        width = max(1, max(map(len, values.values()), default=0))
        array = np.zeros((n + 1, width), dtype=np.int32)
        for uid, value in values.items():
            array[uid, :len(value)] = value
        save_array(out, f'user_array_{field}', array)
    del user_arrays
    # Join integer OIDs after Arrow casts the string content keys to int64.
    mm = np.zeros((len(catalog_rids), 32), dtype=np.float32)
    mm_present = np.zeros(len(catalog_rids), dtype=bool)
    oid_order = np.argsort(oids)
    sorted_oids = oids[oid_order]
    source = ds.dataset(root / 'mm_emb/emb_81_32_parquet', format='parquet')
    for batch in source.scanner(columns=['anonymous_cid', 'emb'], batch_size=32768).to_batches():
        batch_oids = pc.cast(batch.column('anonymous_cid'), pa.int64()).to_numpy()
        pos = np.searchsorted(sorted_oids, batch_oids)
        bounded = np.minimum(pos, len(sorted_oids) - 1)
        keep = ((pos < len(sorted_oids)) & (sorted_oids[bounded] == batch_oids)
                & batch.column('emb').is_valid().to_numpy(zero_copy_only=False))
        if not keep.any():
            continue
        emb = batch.filter(pa.array(keep)).column('emb')
        lengths = pc.list_value_length(emb).to_numpy()
        if not np.all(lengths == 32):
            raise ValueError('Embedding 81 contains a vector with an unexpected dimension.')
        values = pc.cast(pc.list_flatten(emb), pa.float32()).to_numpy(zero_copy_only=False).reshape(-1, 32)
        usable = np.isfinite(values).all(axis=1)
        audit['unusable_mm81_vectors'] += int((~usable).sum())
        local = oid_order[pos[keep]]
        mm[local[usable]] = values[usable]
        mm_present[local[usable]] = True
    save_array(out, 'item_mm81', mm)
    save_array(out, 'mm_present', mm_present)
    val_targets = items[np.arange(n), vectors['val_pos']]
    test_targets = items[np.arange(n), vectors['test_pos']]
    candidate_ds = ds.dataset(root / 'candidate', format='parquet')
    official_oids = candidate_ds.to_table(columns=['item_id'])['item_id'].to_numpy()
    audit.pop('reported', None)
    metadata = {
        'preparation_config': config, 'source_data': str(root),
        'users': n, 'catalog_items': len(catalog_rids) - 1,
        'training_events': int(train_counts.sum()),
        'train_seen_items': int(np.count_nonzero(train_counts)),
        'evaluation_users_per_split': len(eval_indices),
        'candidate_pool': 'all unique items in the sampled public sequences; fixed for every model',
        'candidate_selection_uses_target_labels': False,
        'catalog_assumption': 'transductive public static catalog, not a global-time production backtest',
        'protocol': 'per-user last two first-appearance clicks; strict timestamp cutoffs; '
                    'train only before validation; validation click may enter later test history',
        'history_filter': 'first-appearance click targets exclude any earlier occurrence of the target ad',
        'unknown_action_policy': 'retain context with code 3; exclude unknown events from supervised targets',
        'item_fields': ITEM_FIELDS, 'user_fields': USER_FIELDS,
        'user_array_fields': USER_ARRAY_FIELDS, 'feature_sizes': feature_sizes,
        'categorical_vocabulary_policy': 'fit item ID and attribute vocabularies on training-visible items; OOV=0',
        'item_attribute_catalog_oov_fraction': field_oov,
        'mm81_catalog_coverage': float(mm_present[1:].mean()),
        'val_target_train_seen_fraction': float((train_counts[val_targets] > 0).mean()),
        'test_target_train_seen_fraction': float((train_counts[test_targets] > 0).mean()),
        'official_candidate_count': len(official_oids),
        'official_candidate_val_target_coverage': float(np.isin(oids[val_targets], official_oids).mean()),
        'official_candidate_test_target_coverage': float(np.isin(oids[test_targets], official_oids).mean()),
        'audit': dict(audit), 'preparation_seconds': time.monotonic() - started,
    }
    # Write completion metadata last, so an interrupted preparation is not reused.
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    log(json.dumps(metadata, ensure_ascii=False))
    return metadata


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data-dir', default='TencentGR-1M')
    result.add_argument('--output', default='artifacts/baseline/data')
    result.add_argument('--max-users', type=int, default=100000)
    result.add_argument('--seed', type=int, default=2026)
    result.add_argument('--max-history', type=int, default=100)
    result.add_argument('--min-history', type=int, default=5)
    result.add_argument('--eval-users', type=int, default=2000)
    return result


if __name__ == '__main__':
    prepare(parser().parse_args())
