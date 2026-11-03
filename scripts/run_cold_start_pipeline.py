"""Reproduce cold-start prerequisites, masking training and validation report.

The already-frozen run_cold_start.py remains the training/coverage coordinator.
This wrapper supplies its baseline, content, feature-audit and fusion inputs.
Existing artifacts are reused only after identity checks; test labels are never
used by this pipeline.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def absolute(path):
    return (REPO / path).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compatible_source(source, checksum, snapshot):
    """Permit report-wording updates only when the original code is preserved."""
    if digest(source) == checksum:
        return True
    if not snapshot.exists() or digest(snapshot) != checksum:
        return False

    def computation(path):
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body
                     if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != 'write_report']
        return ast.dump(tree, include_attributes=False)

    return computation(source) == computation(snapshot)


def pair_exists(prefix):
    present = [prefix.with_suffix(suffix).exists() for suffix in ['.json', '.npz']]
    require(present[0] == present[1], f'Partial diagnostic output: {prefix}. Preserve it and use a new root.')
    return all(present)


def identity(prefix, metadata, cohort, targets, seen=None):
    result = read(prefix.with_suffix('.json'))
    require(result.get('split') == 'val' and result.get('test_targets_read') is False,
            f'{prefix}: require explicitly validation-only artifacts.')
    require(result.get('cache_signature') == metadata['preparation_config'], f'{prefix}: cache signature changed.')
    expected = hashlib.sha256(np.asarray(cohort, dtype=np.int64).tobytes()).hexdigest()
    require(result.get('evaluation_cohort_sha256') == expected, f'{prefix}: cohort changed.')
    require(result.get('ks') == [10, 100, 500], f'{prefix}: expected exact budgets 10/100/500.')
    with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['rows'], cohort)
        np.testing.assert_array_equal(saved['targets'], targets)
        require(saved['predictions'].shape == (len(cohort), 500), f'{prefix}: incomplete Top-500.')
        if seen is not None:
            np.testing.assert_array_equal(saved['seen'], seen)
        history = saved['seen'].copy()
    return result, history


def verify_fusion(directory, config, baselines, extra, protected_plan=None, original=None):
    metrics, selection = read(directory / 'metrics.json'), read(directory / 'selection.json')
    require(metrics.get('split') == 'val' and metrics.get('test_targets_read') is False,
            f'{directory}: require validation-only fusion.')
    require(metrics['seeds'] == config['seeds'] and metrics['candidate_budget'] == 500,
            f'{directory}: fusion seeds or budget changed.')
    require(metrics['selection'] == selection, f'{directory}: selection and metrics differ.')
    require(metrics['extra_json_sha256'] == digest(extra.with_suffix('.json'))
            and metrics['extra_npz_sha256'] == digest(extra.with_suffix('.npz')),
            f'{directory}: text input changed.')
    require(absolute(metrics['extra_artifact']) == extra, f'{directory}: wrong text artifact path.')
    input_records = {entry['seed']: entry for entry in metrics['input_artifacts']}
    require(set(input_records) == set(config['seeds']), f'{directory}: baseline seed set changed.')
    for seed, prefix in baselines.items():
        record = input_records[seed]
        require(absolute(record['baseline']) == prefix
                and record['json_sha256'] == digest(prefix.with_suffix('.json'))
                and record['npz_sha256'] == digest(prefix.with_suffix('.npz')),
                f'{directory}: baseline input changed for seed {seed}.')
        with np.load(directory / f'selected_seed{seed}.npz', allow_pickle=False) as saved, \
                np.load(prefix.with_suffix('.npz'), allow_pickle=False) as base:
            for field in ['rows', 'targets', 'seen']:
                np.testing.assert_array_equal(saved[field], base[field])
            require(saved['predictions'].shape == (len(base['rows']), 500), f'{directory}: wrong selected budget.')
            np.testing.assert_array_equal(saved['predictions'][:, :10], base['predictions'][:, :10])
    if protected_plan is None:
        require(metrics['registered_configuration'] == config['fusion'], f'{directory}: fusion rules changed.')
        require(compatible_source(REPO / 'taac_retrieval/candidate_fusion.py', metrics['source_sha256'],
                                  directory / 'source_snapshot/candidate_fusion.py'),
                f'{directory}: naive fusion source changed.')
    else:
        require(metrics['followup_plan_sha256'] == digest(protected_plan), f'{directory}: follow-up plan changed.')
        expected = {(original / name).resolve() for name in ['metrics.json', 'selection.json']}
        actual = {absolute(name) for name in metrics['original_fusion_sha256']}
        require(actual == expected, f'{directory}: different original failure artifacts.')
        for name, checksum in metrics['original_fusion_sha256'].items():
            require(digest(absolute(name)) == checksum, f'{directory}: original failure artifact changed.')
        for name, checksum in metrics['source_sha256'].items():
            require(compatible_source(REPO / 'taac_retrieval' / name, checksum,
                                      directory / 'source_snapshot' / name), f'{directory}: source {name} changed.')
    return metrics


def rewrite_report_links(path, root, config_path):
    """Relocate generated canonical links to this run without changing evidence."""
    path = Path(path)
    if not path.exists():
        return
    body = path.read_text()

    def replace(match):
        target = match.group(1)
        if not target.startswith('../../'):
            return match.group(0)
        name, separator, anchor = target[6:].partition('#')
        if name.startswith('artifacts/cold_start/'):
            destination = root / name[len('artifacts/cold_start/'):]
        elif name == 'configs/cold_start.json':
            destination = config_path
        else:
            destination = REPO / name
        relative = os.path.relpath(destination, path.parent)
        return '(' + relative + (separator + anchor if separator else '') + ')'

    body = re.sub(r'\(([^)\n]+)\)', replace, body)
    relative_root = os.path.relpath(root, REPO)
    body = body.replace('artifacts/cold_start/', relative_root + '/')
    path.write_text(body)


def write_diagnostic_summary(reports, baselines):
    """Supply the baseline-coverage page when reproducing in a fresh root."""
    path = reports / 'diagnostics.md'
    if path.exists():
        return
    records = [read(prefix.with_suffix('.json')) for prefix in baselines.values()]
    lines = ['# 固定模型的验证候选覆盖', '',
             '使用同一冻结验证队列、同一目录和历史过滤，三个既有点击模型分别做精确全库检索。',
             '下面是训练种子均值；每人只有一个留出目标，Recall@10 与 HR@10 数值相同。', '',
             '| 分组 | 用户数 | Recall@10 | Recall@100 | Recall@500 |', '|---|---:|---:|---:|---:|']
    for name in ['all', 'cold_0', 'tail_1_5', 'mid_6_50', 'head_gt50']:
        groups = [row['groups'][name] for row in records]
        values = [np.mean([group[f'Recall@{k}'] for group in groups]) for k in [10, 100, 500]]
        lines.append(f"| {name} | {groups[0]['users']} | " + ' | '.join(f'{value:.4%}' for value in values) + ' |')
    lines += ['', '这只是验证集候选覆盖诊断，不是独立测试或排序模型效果。目录采用公开静态目录假设；分桶频次来自所有训练前缀事件。', '',
              '## 来源', '']
    for seed, prefix in baselines.items():
        target = os.path.relpath(prefix.with_suffix('.json'), path.parent)
        lines.append(f'- [seed={seed} 完整诊断]({target})；同名前缀 `.npz` 保存 Top-500 与逐用户指标。')
    path.write_text('\n'.join(lines) + '\n')


def prerequisites(args):
    root, reports, config_path = absolute(args.root), absolute(args.report_dir), absolute(args.config)
    config = read(config_path)
    cache, reference = absolute(config['cache']), absolute(config['reference_root'])
    cohort_path = absolute(config['evaluation_indices'])
    cohort = np.asarray(np.load(cohort_path, allow_pickle=False), dtype=np.int64)
    metadata = read(cache / 'metadata.json')
    items = np.load(cache / 'items.npy', mmap_mode='r', allow_pickle=False)
    val_pos = np.load(cache / 'val_pos.npy', mmap_mode='r', allow_pickle=False)
    targets = items[cohort, val_pos[cohort]]
    require(config['retrieval_budgets'] == [10, 100, 500], 'This pipeline requires the frozen 10/100/500 budgets.')
    require(config['seeds'] == [2026, 2027, 2028], 'This reproduction expects the three existing reference seeds.')
    root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (root / 'diagnostics').mkdir(exist_ok=True)
    gpus = [int(value) for value in args.gpus.split(',')]
    require(bool(gpus) and len(gpus) == len(set(gpus)), 'Provide unique GPU indices.')

    def run(arguments, log):
        print('Running:', ' '.join([sys.executable] + list(map(str, arguments))), flush=True)
        with Path(log).open('w') as handle:
            subprocess.run([sys.executable] + list(map(str, arguments)), cwd=REPO,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)

    baselines, seen = {}, None
    for index, seed in enumerate(config['seeds']):
        prefix = root / 'diagnostics' / f'click_seed{seed}'
        checkpoint = reference / 'confirmation' / f'selected_seed{seed}' / 'best.pt'
        if not pair_exists(prefix):
            run(['-m', 'taac_retrieval.coverage', '--checkpoint', checkpoint, '--cache', cache,
                 '--output', prefix, '--device', f'cuda:{gpus[index % len(gpus)]}'], prefix.with_suffix('.log'))
        saved, history = identity(prefix, metadata, cohort, targets, seen)
        require(saved['checkpoint_sha256'] == digest(checkpoint), f'{prefix}: different reference checkpoint.')
        require(absolute(saved['cache']) == cache, f'{prefix}: wrong cache path.')
        for name, checksum in saved['code_sha256'].items():
            require(digest(REPO / 'taac_retrieval' / name) == checksum, f'{prefix}: diagnostic source changed.')
        if seen is None:
            seen = history
        baselines[seed] = prefix
        print(f'Verified baseline seed {seed}: {prefix}', flush=True)
    run(['scripts/audit_cold_start_features.py', '--cache', cache, '--cohort', cohort_path,
         '--item-vectors', reference / 'export/item_vectors.npy', '--output', root / 'frequency_audit.json',
         '--click-counts', root / 'diagnostics/train_click_counts.npy'], root / 'frequency_audit_pipeline.log')
    extra = root / 'diagnostics/content_mean81'
    if not pair_exists(extra):
        run(['-m', 'taac_retrieval.content_recall', '--cache', cache, '--cohort', cohort_path,
             '--output', extra, '--learned-predictions', baselines[config['seeds'][0]].with_suffix('.npz'),
             '--device', f'cuda:{gpus[-1]}'], extra.with_suffix('.log'))
    saved, _ = identity(extra, metadata, cohort, targets, seen)
    require(saved['source_sha256'] == digest(REPO / 'taac_retrieval/content_recall.py'),
            'Text-channel implementation differs from its saved result.')
    require(absolute(saved['cache']) == cache, 'Text-channel cache path differs.')
    baseline_pattern = str(root / 'diagnostics/click_seed{seed}')
    common = ['--config', config_path, '--baseline-pattern', baseline_pattern, '--extra-prefix', extra]
    fusion = root / 'fusion'
    if not (fusion / 'metrics.json').exists() and not (fusion / 'selection.json').exists():
        run(['-m', 'taac_retrieval.candidate_fusion', *common, '--output', fusion,
             '--report', reports / 'fusion.md'], root / 'fusion_pipeline.log')
    verify_fusion(fusion, config, baselines, extra)
    plan_path = root / 'fusion_protected_plan.json'
    if not plan_path.exists():
        template_path = absolute(args.followup_plan_template)
        template = read(template_path)
        template.update(registered_at_utc=datetime.now(timezone.utc).isoformat(),
                        reproduction_of=str(template_path), reproduction_of_sha256=digest(template_path),
                        scope='Reproduction of an already explored validation follow-up; not a newly independent confirmation.')
        with plan_path.open('x') as destination:
            destination.write(json.dumps(template, indent=2) + '\n')
    plan = read(plan_path)
    require(plan.get('split') == 'val' and plan.get('test_targets_read') is False
            and plan.get('developer_evidence_only') and plan.get('candidate_budget') == 500,
            'The protected-fusion follow-up must remain explicitly validation-only and fixed-500.')
    protected = root / 'fusion_protected'
    if not (protected / 'metrics.json').exists() and not (protected / 'selection.json').exists():
        run(['-m', 'taac_retrieval.protected_fusion', *common, '--plan', plan_path,
             '--original-fusion', fusion, '--output', protected, '--report', reports / 'fusion_protected.md'],
            root / 'fusion_protected_pipeline.log')
    verify_fusion(protected, config, baselines, extra, plan_path, fusion)
    for name in ['fusion.md', 'fusion_protected.md']:
        rewrite_report_links(reports / name, root, config_path)
    write_diagnostic_summary(reports, baselines)
    print(f'All validation-only prerequisites are ready: {root}', flush=True)
    return root, reports, config_path


def masked_followup(root, reports, config_path, template_path):
    """Add the separately registered post-masking candidate experiment."""
    config = read(config_path)
    plan_path = root / 'masked_fusion_plan.json'
    if not plan_path.exists():
        template_path = absolute(template_path)
        plan = read(template_path)
        plan.update(registered_at_utc=datetime.now(timezone.utc).isoformat(),
                    reproduction_of=str(template_path), reproduction_of_sha256=digest(template_path),
                    reproduction_scope='Reproduction of the validation-driven masked-channel follow-up; not independent confirmation.')
        with plan_path.open('x') as destination:
            destination.write(json.dumps(plan, indent=2) + '\n')
    plan = read(plan_path)
    require(plan.get('split') == 'val' and plan.get('test_targets_read') is False
            and plan.get('extra_arm') == 'id_mask_70' and plan.get('candidate_budget') == 500
            and plan.get('quotas') == [0, 10, 25, 50], 'Masked-channel follow-up plan changed.')
    directory = root / 'fusion_masked'
    if not (directory / 'metrics.json').exists() and not (directory / 'selection.json').exists():
        command = [sys.executable, '-m', 'taac_retrieval.masked_candidate_fusion',
                   '--config', str(config_path), '--plan', str(plan_path),
                   '--baseline-pattern', str(root / 'diagnostics/click_seed{seed}'),
                   '--extra-pattern', str(root / 'diagnostics/id_mask_70_seed{seed}'),
                   '--output', str(directory), '--report', str(reports / 'fusion_masked.md')]
        print('Running:', ' '.join(command), flush=True)
        with (root / 'fusion_masked_pipeline.log').open('w') as handle:
            subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)
    metrics, selection = read(directory / 'metrics.json'), read(directory / 'selection.json')
    require(metrics.get('split') == 'val' and metrics.get('test_targets_read') is False,
            'Require explicitly validation-only masked-channel fusion.')
    require(metrics['followup_plan_sha256'] == digest(plan_path) and metrics['extra_arm'] == 'id_mask_70',
            'Masked-channel fusion plan or arm changed.')
    require(metrics['selection'] == selection and metrics['seeds'] == config['seeds']
            and metrics['candidate_budget'] == 500, 'Masked-channel selection or budget changed.')
    cache = absolute(config['cache'])
    metadata = read(cache / 'metadata.json')
    cohort = np.asarray(np.load(absolute(config['evaluation_indices']), allow_pickle=False), dtype=np.int64)
    items = np.load(cache / 'items.npy', mmap_mode='r', allow_pickle=False)
    positions = np.load(cache / 'val_pos.npy', mmap_mode='r', allow_pickle=False)
    targets = items[cohort, positions[cohort]]
    require(metrics['cache_signature'] == metadata['preparation_config'], 'Masked-channel cache changed.')
    require(metrics['evaluation_cohort_sha256'] == hashlib.sha256(cohort.tobytes()).hexdigest(),
            'Masked-channel cohort changed.')
    inputs = {entry['seed']: entry for entry in metrics['input_artifacts']}
    require(set(inputs) == set(config['seeds']), 'Masked-channel seed set changed.')
    for seed in config['seeds']:
        prefix = root / 'diagnostics' / f'click_seed{seed}'
        baseline_meta, seen = identity(prefix, metadata, cohort, targets)
        for role, prefix in [('baseline', prefix), ('extra', root / 'diagnostics' / f'id_mask_70_seed{seed}')]:
            meta, _ = identity(prefix, metadata, cohort, targets, seen)
            record = inputs[seed][role]
            require(absolute(record['prefix']) == prefix
                    and record['json_sha256'] == digest(prefix.with_suffix('.json'))
                    and record['npz_sha256'] == digest(prefix.with_suffix('.npz'))
                    and record['checkpoint_sha256'] == meta['checkpoint_sha256'],
                    f'Masked-channel {role} input changed for seed {seed}.')
            if role == 'extra':
                require(meta['checkpoint_sha256'] == digest(root / 'training' / f'id_mask_70_seed{seed}' / 'best.pt'),
                        f'ID-mask checkpoint changed for seed {seed}.')
        with np.load(directory / f'selected_seed{seed}.npz', allow_pickle=False) as saved, \
                np.load((root / 'diagnostics' / f'click_seed{seed}').with_suffix('.npz'), allow_pickle=False) as base:
            for field in ['rows', 'targets', 'seen']:
                np.testing.assert_array_equal(saved[field], base[field])
            require(saved['predictions'].shape == (len(cohort), 500), 'Masked-channel selected budget changed.')
            np.testing.assert_array_equal(saved['predictions'][:, :10], base['predictions'][:, :10])
    for name, checksum in metrics['source_sha256'].items():
        require(compatible_source(REPO / 'taac_retrieval' / name, checksum,
                                  directory / 'source_snapshot' / name), f'Masked-channel source {name} changed.')
    preserved = metrics['preserved_previous_fusion_sha256']
    expected = {(root / name / filename).resolve() for name in ['fusion', 'fusion_protected']
                for filename in ['metrics.json', 'selection.json']}
    require({absolute(name) for name in preserved} == expected,
            'Masked-channel follow-up did not preserve the complete earlier fusion evidence.')
    for name, checksum in preserved.items():
        require(digest(absolute(name)) == checksum, f'Earlier fusion evidence changed: {name}.')
    rewrite_report_links(reports / 'fusion_masked.md', root, config_path)
    subprocess.run([sys.executable, '-m', 'taac_retrieval.cold_start_report', '--root', str(root),
                    '--report-dir', str(reports)], cwd=REPO, check=True)
    rewrite_report_links(reports / 'README.md', root, config_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/cold_start.json')
    parser.add_argument('--root', default='artifacts/cold_start')
    parser.add_argument('--report-dir', default='reports/cold_start')
    parser.add_argument('--gpus', default='0,2,3,4,5,6')
    parser.add_argument('--stage', choices=['all', 'finish'], default='all')
    parser.add_argument('--followup-plan-template', default='artifacts/cold_start/fusion_protected_plan.json')
    parser.add_argument('--masked-followup-plan-template', default='artifacts/cold_start/masked_fusion_plan.json')
    parser.add_argument('--prerequisites-only', action='store_true', help='Prepare/verify diagnostics and stop before model training/reporting.')
    args = parser.parse_args()
    root, reports, config_path = prerequisites(args)
    if args.prerequisites_only:
        return
    subprocess.run([sys.executable, str(REPO / 'scripts/run_cold_start.py'), '--config', str(config_path),
                    '--root', str(root), '--report-dir', str(reports), '--gpus', args.gpus,
                    '--stage', args.stage], cwd=REPO, check=True)
    masked_followup(root, reports, config_path, args.masked_followup_plan_template)


if __name__ == '__main__':
    main()
