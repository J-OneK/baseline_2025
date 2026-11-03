"""Check the entire prepared cohort before accepting experiment metrics."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import PreparedData, load_array
from .prepare import source_signature


def audit(cache):
    data = PreparedData(cache)
    n, width = data.items.shape
    positions = np.arange(width)[None, :]
    checks = {}

    def require(name, condition):
        if not bool(condition):
            raise AssertionError(f'Data audit failed: {name}')
        checks[name] = True

    require('source_files_match_cache', source_signature(Path(data.metadata['source_data'])) ==
            data.metadata['preparation_config']['source_signature'])
    require('unique_users', len(np.unique(data.user_rids)) == n)
    require('all_users_have_training_history', np.all(data.train_cut >=
            data.metadata['preparation_config']['min_history']))
    require('training_stops_before_validation', np.array_equal(data.train_cut, data.val_cut))
    require('validation_precedes_test', np.all(data.val_pos < data.test_pos))
    require('cuts_within_sequence', np.all(data.val_cut <= data.val_pos) and
            np.all(data.test_cut <= data.test_pos) and np.all(data.test_pos < data.lengths))
    require('only_documented_action_codes', np.isin(data.actions, [0, 1, 2, 3]).all())
    for split in ['val', 'test']:
        target_position = getattr(data, f'{split}_pos')
        cut = getattr(data, f'{split}_cut')
        targets = data.items[np.arange(n), target_position]
        target_time = data.timestamps[np.arange(n), target_position]
        observed = positions < cut[:, None]
        require(f'{split}_target_is_click', np.all(data.actions[np.arange(n), target_position] == 2))
        require(f'{split}_target_not_in_history', not np.any((data.items == targets[:, None]) & observed))
        require(f'{split}_history_strictly_earlier',
                not np.any((data.timestamps >= target_time[:, None]) & observed))
        require(f'{split}_target_in_fixed_catalog', np.all((targets > 0) & (targets <= data.num_items)))
    counts = np.bincount(data.items[positions < data.train_cut[:, None]], minlength=data.num_items + 1)
    require('popularity_and_negative_pool_use_training_prefixes_only', np.array_equal(counts, data.train_counts))
    sparse = load_array(cache, 'item_sparse')
    for j, field in enumerate(data.metadata['item_fields']):
        require(f'field_{field}_in_vocabulary', sparse[:, j].min() >= 0 and
                sparse[:, j].max() < data.metadata['feature_sizes'][field])
    mm = load_array(cache, 'item_mm81')
    require('content_vectors_are_finite', np.isfinite(mm).all())
    present = load_array(cache, 'mm_present')
    require('content_join_is_nonzero', float(np.any(mm[1:] != 0, axis=1).mean()) > .8)
    require('padding_has_no_content', np.all(mm[0] == 0))
    require('missing_content_is_zero_filled', np.all(mm[~present] == 0))
    lookup = load_array(cache, 'item_id_lookup')
    require('unseen_ids_use_oov_embedding', np.all(lookup[data.train_counts == 0] == 0))
    require('seen_ids_have_distinct_embeddings', len(np.unique(lookup[data.train_counts > 0])) ==
            np.count_nonzero(data.train_counts))
    return dict(checks=checks, check_count=len(checks), passed=True,
                users=n, candidate_items=data.num_items, protocol=data.metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='artifacts/baseline/data')
    parser.add_argument('--output', default='reports/baseline/data_audit.json')
    args = parser.parse_args()
    result = audit(args.cache)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'protocol'}, indent=2))
