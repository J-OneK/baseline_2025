"""Read-only training/validation audit of sparse supervision and content coverage.

No test split positions or targets are opened. Outputs are kept outside the
prepared cache; repeated execution verifies and reuses identical artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(root, name):
    return np.load(root / f'{name}.npy', mmap_mode='r', allow_pickle=False)


def fraction(values):
    return float(np.mean(values)) if len(values) else None


def group_masks(counts):
    return dict(cold_0=counts == 0, tail_1_5=(counts >= 1) & (counts <= 5),
                mid_6_50=(counts >= 6) & (counts <= 50), head_gt50=counts > 50)


def audit(args):
    cache = Path(args.cache).resolve()
    output, click_path = Path(args.output).resolve(), Path(args.click_counts).resolve()
    if cache in output.parents or cache in click_path.parents:
        raise ValueError('Audit outputs must remain outside the prepared cache.')
    metadata_path = cache / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    items, actions, cuts, counts = [load(cache, name) for name in ['items', 'actions', 'train_cut', 'train_counts']]
    sparse, mm_present, raw_text = [load(cache, name) for name in ['item_sparse', 'mm_present', 'item_mm81']]
    vectors = np.load(args.item_vectors, mmap_mode='r', allow_pickle=False)
    if vectors.shape != raw_text.shape or len(counts) != len(vectors):
        raise ValueError('Candidate vectors/content and training frequency table differ.')
    if '121' not in metadata['item_fields']:
        raise ValueError('This audit requires the actual field-121 feature mapping.')
    field121 = sparse[:, metadata['item_fields'].index('121')]
    click_counts = np.zeros(len(counts), dtype=np.int64)
    recomputed_counts = np.zeros(len(counts), dtype=np.int64)
    positions = np.arange(items.shape[1])[None, :]
    for start in range(0, len(items), args.batch_size):
        stop = min(start + args.batch_size, len(items))
        prefix = positions < cuts[start:stop, None]
        selected = items[start:stop][prefix]
        if (selected <= 0).any():
            raise AssertionError('Padding appeared inside a training prefix.')
        recomputed_counts += np.bincount(selected, minlength=len(counts))
        clicked = prefix & (actions[start:stop] == 2)
        click_counts += np.bincount(items[start:stop][clicked], minlength=len(counts))
    np.testing.assert_array_equal(recomputed_counts, counts)
    if (click_counts > counts).any() or click_counts[0] != 0:
        raise AssertionError('Click-positive frequency is inconsistent with all-event frequency.')
    item_norms = np.empty(len(counts), dtype=np.float32)
    text_eligible = np.zeros(len(counts), dtype=bool)
    for start in range(0, len(counts), 65536):
        stop = min(start + 65536, len(counts))
        if not np.isfinite(vectors[start:stop]).all():
            raise ValueError('Non-finite exported item vectors.')
        item_norms[start:stop] = np.linalg.norm(vectors[start:stop], axis=1)
        raw = raw_text[start:stop]
        text_eligible[start:stop] = (mm_present[start:stop] & np.isfinite(raw).all(1)
                                    & (np.linalg.norm(raw, axis=1) > 1e-12))
    text_eligible[0] = False
    groups = group_masks(counts)
    groups['cold_0'][0] = False
    catalog = {}
    for name, selected in groups.items():
        ids = np.flatnonzero(selected)
        norms = item_norms[ids]
        catalog[name] = dict(items=len(ids),
                             zero_training_click_fraction=fraction(click_counts[ids] == 0),
                             text_present_fraction=fraction(mm_present[ids]),
                             usable_text_fraction=fraction(text_eligible[ids]),
                             field121_zero_or_oov_fraction=fraction(field121[ids] == 0),
                             mean_exported_vector_norm=fraction(norms),
                             exported_vector_norm_quantiles=dict(zip(['p10', 'p50', 'p90'],
                                 np.quantile(norms, [.1, .5, .9]).tolist())) if len(ids) else None)
    train_values = field121[counts > 0]
    histogram = np.bincount(train_values, minlength=metadata['feature_sizes']['121'])
    category_sizes = histogram[train_values]
    field_audit = dict(train_visible_items=len(train_values),
                       distinct_train_categories=int(np.count_nonzero(histogram)),
                       distinct_nonzero_train_categories=int(np.count_nonzero(histogram[1:])),
                       fraction_of_items_with_singleton_category=fraction(category_sizes == 1),
                       category_size_quantiles_per_item=dict(zip(['p10', 'p50', 'p90', 'p99'],
                           np.quantile(category_sizes, [.1, .5, .9, .99]).tolist())),
                       semantics='anonymous high-cardinality field; business meaning is not inferred',
                       vocabulary_policy=metadata['categorical_vocabulary_policy'])
    cohort = np.asarray(np.load(args.cohort, allow_pickle=False), dtype=np.int64)
    if (cohort.ndim != 1 or len(np.unique(cohort)) != len(cohort)
            or cohort.min() < 0 or cohort.max() >= len(items)):
        raise ValueError('Invalid frozen validation cohort.')
    val_pos, val_cut = load(cache, 'val_pos'), load(cache, 'val_cut')
    targets = items[cohort, val_pos[cohort]]
    visible = np.where(positions < val_cut[cohort, None], items[cohort], 0)
    if (targets <= 0).any() or (visible == targets[:, None]).any():
        raise AssertionError('Invalid validation target or target identity leaked into visible history.')
    validation = {}
    for name, selected in group_masks(counts[targets]).items():
        ids = targets[selected]
        validation[name] = dict(users=len(ids),
                                zero_training_click_fraction=fraction(click_counts[ids] == 0),
                                text_missing_fraction=fraction(~mm_present[ids]),
                                usable_text_fraction=fraction(text_eligible[ids]),
                                field121_zero_or_oov_fraction=fraction(field121[ids] == 0),
                                median_training_click_count=float(np.median(click_counts[ids])) if len(ids) else None)
    if sum(row['users'] for row in validation.values()) != len(cohort):
        raise AssertionError('Frequency groups do not partition validation users.')
    click_path.parent.mkdir(parents=True, exist_ok=True)
    if click_path.exists():
        np.testing.assert_array_equal(np.load(click_path, mmap_mode='r', allow_pickle=False), click_counts)
    else:
        with click_path.open('xb') as handle:
            np.save(handle, click_counts, allow_pickle=False)
    result = dict(configuration=dict(cache=str(cache), cohort=str(Path(args.cohort).resolve()),
                                     item_vectors=str(Path(args.item_vectors).resolve()),
                                     click_counts_output=str(click_path), split='val'),
                  cache_signature=metadata['preparation_config'], cache_metadata_sha256=file_digest(metadata_path),
                  code_sha256=file_digest(Path(__file__)),
                  evaluation_cohort_sha256=hashlib.sha256(cohort.tobytes()).hexdigest(),
                  num_users=len(items), candidate_items=len(counts)-1, validation_users=len(cohort),
                  training_events=int(counts.sum()), click_positive_training_positions=int(click_counts.sum()),
                  exposure_or_unknown_training_events=int(counts.sum()-click_counts.sum()),
                  frequency_definition='all events in all user training prefixes, including exposure/click/unknown',
                  positive_frequency_definition='action code 2 in each user training prefix; matches click InfoNCE supervised positions',
                  catalog_groups=catalog, validation_target_groups=validation, field121=field_audit,
                  exported_norm_scope='one saved seed2026 learned raw-IP candidate export; norms alone do not establish ranking causality',
                  click_counts_sha256=file_digest(click_path),
                  checks=dict(train_counts_recomputed_exactly=True, clicks_subset_of_all_events=True,
                              validation_groups_partition_cohort=True, validation_target_absent_from_history=True,
                              cache_modified=False, test_positions_or_targets_opened=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise FileExistsError('An audit with different provenance/results exists; use a new output path.')
    else:
        with output.open('x') as destination:
            destination.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--cache', default='artifacts/objective_study/scale_data')
    result.add_argument('--cohort', default='artifacts/objective_study/confirm_eval_indices.npy')
    result.add_argument('--item-vectors', default='artifacts/objective_study/export/item_vectors.npy')
    result.add_argument('--output', default='artifacts/cold_start/frequency_audit.json')
    result.add_argument('--click-counts', default='artifacts/cold_start/diagnostics/train_click_counts.npy')
    result.add_argument('--batch-size', type=int, default=10000)
    return result


if __name__ == '__main__':
    audit(parser().parse_args())
