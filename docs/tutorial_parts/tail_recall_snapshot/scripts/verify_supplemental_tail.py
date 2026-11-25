"""Independently verify saved supplemental-tail proposals, fusion and metrics.

No model inference or training is performed. All cohort, candidate and target
checks use the already completed, locked study's saved arrays.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def array_sha(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def masks(targets, counts, clicks):
    frequency, support = counts[targets], clicks[targets]
    return dict(all=np.ones(len(targets), dtype=bool), cold_0=frequency == 0,
                tail_1_5=(frequency >= 1) & (frequency <= 5),
                zero_click_tail=(frequency >= 1) & (frequency <= 5) & (support == 0),
                clicked_tail=(frequency >= 1) & (frequency <= 5) & (support > 0),
                mid_6_50=(frequency >= 6) & (frequency <= 50), head_gt50=frequency > 50)


def group_summary(main, fused, targets, counts, clicks):
    before, after = (main == targets[:, None]).any(1), (fused == targets[:, None]).any(1)
    result = {}
    for name, selected in masks(targets, counts, clicks).items():
        assert selected.any(), f'Empty group needs explicit reporting: {name}'
        result[name] = dict(users=int(selected.sum()), baseline_hits=int(before[selected].sum()),
            hits=int(after[selected].sum()), gained=int((~before & after & selected).sum()),
            lost=int((before & ~after & selected).sum()),
            baseline_recall=float(before[selected].mean()), recall=float(after[selected].mean()))
    return result


def compare(actual, expected):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, key
            compare(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            compare(value, reference)
    elif isinstance(expected, (float, np.floating)):
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def reconstruct(main, proposals, counts, quota, protected):
    removable = [position for position in range(len(main) - 1, protected - 1, -1)
                 if counts[main[position]] > 5]
    used = set(map(int, main))
    novel = []
    for value in proposals:
        item = int(value)
        if item and item not in used:
            novel.append(item)
            used.add(item)
    take = min(quota, len(novel), len(removable))
    removed = set(removable[:take])
    return np.asarray([int(value) for position, value in enumerate(main) if position not in removed]
                      + novel[:take], dtype=main.dtype), take


def aggregate(workers, key, plan):
    combinations = [worker['combinations'][key] for worker in workers]
    groups = {group: {field: float(np.mean([row['groups'][group][field] for row in combinations]))
                     for field in ['baseline_recall', 'recall', 'baseline_hits', 'hits', 'gained', 'lost']}
              for group in combinations[0]['groups']}
    reasons = []
    if groups['all']['recall'] < plan['guards']['overall_retention'] * groups['all']['baseline_recall']:
        reasons.append('overall_retention')
    for group in ['cold_0', 'clicked_tail']:
        if groups[group]['recall'] < groups[group]['baseline_recall']:
            reasons.append(group + '_decreased')
    positive = sum(row['groups']['zero_click_tail']['hits'] > row['groups']['zero_click_tail']['baseline_hits']
                   for row in combinations)
    if groups['zero_click_tail']['recall'] <= groups['zero_click_tail']['baseline_recall']:
        reasons.append('no_zero_click_tail_gain')
    if positive < plan['guards']['minimum_positive_seeds']:
        reasons.append('insufficient_positive_seed_models')
    return dict(key=key, score_mode=combinations[0]['score_mode'], quota=combinations[0]['quota'],
                groups=groups, passed=not reasons, reasons=reasons, positive_seed_models=positive)


def verify(args):
    root, reports = Path(args.root).resolve(), Path(args.report_dir).resolve()
    plan, identity, manifest = read(root / 'plan.json'), read(root / 'identity.json'), read(root / 'source_manifest.json')
    report = read(reports / 'metrics.json')
    assert report['plan'] == plan and report['source_manifest'] == manifest
    assert all(sha(REPO / path) == value for path, value in manifest.items())
    cache = Path(identity['cache'])
    assert sha(cache / 'metadata.json') == identity['cache_metadata_sha256']
    assert sha(identity['counts_file']) == identity['training_click_counts_sha256']
    counts = np.load(cache / 'train_counts.npy', allow_pickle=False)
    clicks = np.load(identity['counts_file'], allow_pickle=False)
    assert counts.shape == clicks.shape and (clicks >= 0).all() and (clicks <= counts).all()
    eligible = np.flatnonzero((counts >= 1) & (counts <= 5) & (clicks == 0))
    eligible = eligible[eligible > 0]
    eligible_mask = np.zeros(len(counts), bool)
    eligible_mask[eligible] = True
    user_rids = np.load(cache / 'user_rids.npy', allow_pickle=False)
    excluded = []
    for source in identity['sources']:
        assert sha(source['indices']) == source['indices_sha256']
        assert sha(source['user_rids']) == source['user_rids_sha256']
        rids = np.load(source['user_rids'], allow_pickle=False)
        rows = np.load(source['indices'], allow_pickle=False)
        excluded.extend(rids[rows].tolist())
    excluded = np.unique(excluded)
    development = np.load(root / 'development.npy', allow_pickle=False)
    confirmation = np.load(root / 'confirmation.npy', allow_pickle=False)
    assert not np.intersect1d(development, confirmation).size
    assert not np.isin(user_rids[confirmation], excluded).any()
    for name, rows in [('development', development), ('confirmation', confirmation)]:
        assert array_sha(rows) == identity[name + '_indices_sha256']
        assert array_sha(user_rids[rows]) == identity[name + '_rids_sha256']
    old_rows = np.load(identity['sources'][2]['indices'], allow_pickle=False)
    np.testing.assert_array_equal(development, old_rows)
    parameter_hashes = {}
    for seed, entry in identity['checkpoints'].items():
        assert sha(entry['path']) == entry['sha256'] == identity['initial_checkpoints_sha256'][seed]
        checkpoint = torch.load(entry['path'], map_location='cpu', weights_only=False)
        value = hashlib.sha256()
        # This model stores only trainable parameter tensors in its state_dict;
        # feature buffers are nonpersistent. Ordering matches named_parameters.
        for name, parameter in checkpoint['model'].items():
            value.update(name.encode())
            value.update(parameter.detach().cpu().numpy().tobytes())
        parameter_hashes[int(seed)] = value.hexdigest()
        del checkpoint
    for seed, reference in identity['references'].items():
        assert sha(reference['path']) == reference['sha256']
        original = read(Path(reference['path']).with_suffix('.json'))
        assert original['checkpoint_sha256'] == identity['checkpoints'][seed]['sha256']
    selected = report['selected']
    lock = read(root / 'confirmation_lock.json') if selected is not None else None
    if lock:
        assert lock['checkpoints'] == identity['checkpoints']
        assert lock['cohort_sha256'] == array_sha(confirmation)
        assert lock['score_mode'] == selected['score_mode'] and lock['quota'] == selected['quota']
    else:
        assert not (root / 'confirmation').exists()
        assert not report['confirmation_targets_evaluated']
    items = np.load(cache / 'items.npy', mmap_mode='r', allow_pickle=False)
    cutoffs = np.load(cache / 'val_cut.npy', allow_pickle=False)
    positions = np.load(cache / 'val_pos.npy', allow_pickle=False)
    timestamps = np.load(cache / 'timestamps.npy', mmap_mode='r', allow_pickle=False)
    phase_workers, saved_hit_deltas, saved_targets = {}, [], None
    checked_fusions, checked_queries, phase_counts = 0, 0, {}
    top10_equal = True
    for phase in ['development', *(['confirmation'] if lock else [])]:
        rows = development if phase == 'development' else confirmation
        expected_targets = items[rows, positions[rows]]
        expected_seen = np.zeros((len(rows), items.shape[1] + 1), dtype=np.int64)
        for index, row in enumerate(rows):
            cut = int(cutoffs[row])
            expected_seen[index, -cut:] = items[row, :cut]
            assert timestamps[row, cut - 1] < timestamps[row, positions[row]]
        modes = plan['score_modes'] if phase == 'development' else [lock['score_mode']]
        quotas = plan['quotas'] if phase == 'development' else [lock['quota']]
        workers = []
        for seed in plan['seeds']:
            prefix = root / phase / f'seed{seed}'
            worker = read(prefix.with_suffix('.json'))
            assert worker['source_manifest'] == manifest
            assert worker['model_parameters_unchanged'] and worker['shared_parameter_sha256'] == parameter_hashes[seed]
            assert worker['checkpoint'] == identity['checkpoints'][str(seed)]
            assert worker['auxiliary_pool_items'] == len(eligible)
            assert worker['eligible_ids_sha256'] == array_sha(eligible)
            if phase == 'confirmation':
                assert prefix.with_suffix('.npz').stat().st_mtime >= (root / 'confirmation_lock.json').stat().st_mtime
            with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as source:
                main, seen, targets = source['predictions'], source['seen'], source['targets']
                np.testing.assert_array_equal(source['rows'], rows)
                np.testing.assert_array_equal(targets, expected_targets)
                np.testing.assert_array_equal(seen, expected_seen)
                assert main.shape == (len(rows), 500) and (main > 0).all() and (main < len(counts)).all()
                assert not (np.diff(np.sort(main, axis=1), axis=1) == 0).any()
                assert np.isfinite(source['queries']).all() and source['queries'].shape == (len(rows), 32)
                assert not (seen == targets[:, None]).any()
                if phase == 'development':
                    with np.load(identity['references'][str(seed)]['path'], allow_pickle=False) as old:
                        for name in ['predictions', 'queries', 'seen', 'rows', 'targets']:
                            np.testing.assert_array_equal(source[name], old[name])
                for mode in modes:
                    proposals = source[f'proposals_{mode}']
                    assert proposals.shape == (len(rows), plan['proposal_k']) and (proposals >= 0).all()
                    assert (proposals < len(counts)).all() and eligible_mask[proposals[proposals > 0]].all()
                    for index, proposed in enumerate(proposals):
                        valid = proposed[proposed > 0]
                        assert len(np.unique(valid)) == len(valid)
                        blocked = set(map(int, seen[index])) | set(map(int, main[index]))
                        assert not blocked.intersection(map(int, valid))
                    for quota in quotas:
                        key = f'{mode}_q{quota}'
                        fused = source[key]
                        assert fused.shape == main.shape and (fused > 0).all() and (fused < len(counts)).all()
                        assert not (np.diff(np.sort(fused, axis=1), axis=1) == 0).any()
                        np.testing.assert_array_equal(fused[:, :10], main[:, :10])
                        replaced = []
                        for index in range(len(rows)):
                            expected, applied = reconstruct(main[index], proposals[index], counts, quota, plan['protected_top'])
                            np.testing.assert_array_equal(fused[index], expected)
                            selected_ids = set(map(int, fused[index]))
                            assert not selected_ids.intersection(set(map(int, seen[index])) - {0})
                            assert set(map(int, main[index][counts[main[index]] <= 5])).issubset(selected_ids)
                            assert all(eligible_mask[item] for item in selected_ids - set(map(int, main[index])))
                            replaced.append(applied)
                        groups = group_summary(main, fused, targets, counts, clicks)
                        compare(worker['combinations'][key]['groups'], groups)
                        compare(worker['combinations'][key]['mean_replacements'], float(np.mean(replaced)))
                        assert groups['cold_0']['gained'] == groups['cold_0']['lost'] == 0
                        assert groups['clicked_tail']['gained'] == groups['clicked_tail']['lost'] == 0
                        assert groups['zero_click_tail']['lost'] == 0
                        assert groups['mid_6_50']['gained'] == groups['head_gt50']['gained'] == 0
                        # Equality of prefix IDs implies equal per-user HR/NDCG@10.
                        discounts = 1 / np.log2(np.arange(10) + 2)
                        np.testing.assert_array_equal((main[:, :10] == targets[:, None]).any(1),
                                                      (fused[:, :10] == targets[:, None]).any(1))
                        np.testing.assert_array_equal(((main[:, :10] == targets[:, None]) * discounts).sum(1),
                                                      ((fused[:, :10] == targets[:, None]) * discounts).sum(1))
                        checked_fusions += 1
                if phase == 'confirmation':
                    if saved_targets is not None:
                        np.testing.assert_array_equal(saved_targets, targets)
                    saved_targets = targets.copy()
                    fused = source[selected['key']]
                    saved_hit_deltas.append((fused == targets[:, None]).any(1).astype(float)
                                            - (main == targets[:, None]).any(1).astype(float))
            workers.append(worker)
            checked_queries += len(rows)
        phase_workers[phase] = workers
        phase_counts[phase] = len(workers)
    development_rows = [aggregate(phase_workers['development'], f'{mode}_q{quota}', plan)
                        for mode in plan['score_modes'] for quota in plan['quotas']]
    compare(report['development'], development_rows)
    compare(read(root / 'development_selection.json'), development_rows)
    eligible_rows = [row for row in development_rows if row['passed']]
    expected_selection = max(eligible_rows, key=lambda row: (row['groups']['zero_click_tail']['recall'],
        row['groups']['all']['recall'], -row['quota'], row['score_mode'] == 'raw_ip')) if eligible_rows else None
    assert selected == expected_selection
    interval_result = None
    if lock:
        compare(report['confirmation'], aggregate(phase_workers['confirmation'], selected['key'], plan))
        rng = np.random.default_rng(plan['bootstrap']['seed'])
        delta, interval_result = np.stack(saved_hit_deltas), {}
        all_masks = masks(saved_targets, counts, clicks)
        for group in ['all', 'cold_0', 'tail_1_5', 'zero_click_tail']:
            chosen = delta[:, all_masks[group]]
            bootstrap = []
            for _ in range(plan['bootstrap']['replicates']):
                seed_indices = rng.integers(0, 3, 3)
                user_indices = rng.integers(0, chosen.shape[1], chosen.shape[1])
                bootstrap.append(float(chosen[seed_indices][:, user_indices].mean()))
            interval_result[group] = dict(mean=float(chosen.mean()),
                ci95=np.quantile(bootstrap, [.025, .975]).tolist(), users=chosen.shape[1],
                unique_users_with_gain=int((chosen > 0).any(0).sum()))
        compare(report['paired_intervals'], interval_result)
    test_log = Path(args.test_log).read_text()
    match = re.search(r'Ran (\d+) tests', test_log)
    assert match and '\nOK\n' in test_log
    preflight = read(args.preflight)
    assert preflight['unchanged_model_parameters'] and preflight['no_seen_or_main_overlap'] and preflight['protected_top10']
    assert preflight['eligible_items'] == len(eligible) and preflight['candidate_budget'] == 500
    assert preflight['source_sha256'] == sha(REPO / 'taac_retrieval/tail_channel.py')
    result = dict(status=report['status'], frozen_source_files=len(manifest), source_manifest_matches=True,
        original_checkpoint_files_verified=3, original_parameter_hashes_verified=True,
        original_models_retrained=False, original_main_replaced=False,
        eligible_zero_click_tail_pool=len(eligible), excluded_previous_user_rids=len(excluded),
        development_users=len(development), confirmation_users=len(confirmation),
        development_matches_old_cohort=True, development_confirmation_overlap=0,
        confirmation_previous_evaluation_overlap=0, phase_worker_arrays_verified=phase_counts,
        saved_query_rows_verified=checked_queries, fused_arrays_independently_reconstructed=checked_fusions,
        fixed_500_unique_candidates=True, main_top10_and_per_user_hr_ndcg10_unchanged=top10_equal,
        original_cold_tail_candidates_preserved=True, zero_click_tail_only_new_candidates=True,
        novel_proposals_exclude_all_main500_and_visible_history=True,
        all_per_model_groups_gainloss_and_aggregates_recomputed=True,
        score_and_quota_selected_on_development_only=True,
        confirmation_locked_before_saved_arrays=lock is not None,
        confirmation_targets_evaluated=lock is not None,
        paired_seed_user_bootstrap_independently_recomputed=lock is not None,
        zero_click_tail_confirmation_interval=None if interval_result is None else interval_result['zero_click_tail'],
        cpu_tests_passed=int(match.group(1)), gpu_16_user_preflight_passed=True,
        verifier_source_sha256=sha(__file__))
    reports.mkdir(parents=True, exist_ok=True)
    (reports / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    (reports / 'acceptance_tests.log').write_text(test_log)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/supplemental_tail_v1')
    parser.add_argument('--report-dir', default='reports/supplemental_tail_study')
    parser.add_argument('--test-log', default='artifacts/supplemental_tail_acceptance_tests.log')
    parser.add_argument('--preflight', default='artifacts/supplemental_tail_preflight.json')
    verify(parser.parse_args())
