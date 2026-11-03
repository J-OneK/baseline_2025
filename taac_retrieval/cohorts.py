"""Freeze an expanded evaluation cohort excluding previously viewed test users."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import PreparedData


def freeze_cohort(cache, previous_cache, output, count=10000, seed=3107):
    data, previous = PreparedData(cache), PreparedData(previous_cache)
    excluded = previous.user_rids[previous.eval_indices]
    eligible = np.flatnonzero(~np.isin(data.user_rids, excluded))
    if len(eligible) < count:
        raise ValueError('Not enough fresh evaluation users.')
    selected = np.sort(np.random.default_rng(seed).choice(eligible, count, replace=False)).astype(np.int64)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        np.testing.assert_array_equal(np.load(output), selected)
    else:
        np.save(output, selected)
    assert not np.isin(data.user_rids[selected], excluded).any()
    description = dict(cache=str(Path(cache).resolve()), users=count, seed=seed,
                       excluded_previous_test_users=len(excluded), overlap_with_previous_test_users=0,
                       indices_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                       role='validation selects epoch; later test clicks remain unopened until objective is locked')
    output.with_suffix('.json').write_text(json.dumps(description, indent=2) + '\n')
    return description


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--previous-cache', default='artifacts/day1_3/data')
    parser.add_argument('--output', required=True)
    parser.add_argument('--users', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=3107)
    args = parser.parse_args()
    print(json.dumps(freeze_cohort(args.cache, args.previous_cache, args.output, args.users, args.seed), indent=2))
