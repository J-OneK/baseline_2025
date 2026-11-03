"""Matched-initialization, validation-only evidence for sequence depth studies."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .coverage import coverage_summary
from .data import PreparedData, make_input
from .week_report import confidence_intervals


GROUPS = ['all', 'cold_0', 'tail_1_5', 'mid_6_50', 'head_gt50']
KS = [10, 100, 500]
MATCHED_FIELDS = ['variant', 'seed', 'epochs', 'batch_size', 'eval_batch_size', 'candidate_chunk',
                  'workers', 'hidden', 'heads', 'dropout', 'lr', 'max_steps', 'objective',
                  'num_negatives', 'exposure_weight', 'temperature', 'skip_test',
                  'torch_version', 'numpy_version']


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def scalar(row, group, metric):
    value = row['groups'][group][metric]
    return float(value['mean'] if isinstance(value, dict) else value)


def select_depth(rows):
    """Select mean best-validation NDCG, then HR, then the smaller depth."""
    depths = [row['depth'] for row in rows]
    if not rows or len(set(depths)) != len(depths) or 1 not in depths:
        raise ValueError('Require distinct depths and one depth-1 control.')
    evaluated = []
    for row in rows:
        depth = row['depth']
        if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)) or depth < 1:
            raise ValueError('Depth must be a positive integer.')
        ndcg = scalar(row, 'all', 'NDCG@10')
        hr = scalar(row, 'all', 'Recall@10')
        if not np.isfinite([ndcg, hr]).all() or not (0 <= ndcg <= 1 and 0 <= hr <= 1) or ndcg > hr + 1e-12:
            raise ValueError('Selection metrics must be finite values in [0,1].')
        evaluated.append(dict(depth=int(depth), name=row.get('name', f'depth_{depth}'),
                               validation_ndcg10=ndcg, validation_hr10=hr))
    winner = min(evaluated, key=lambda row: (-row['validation_ndcg10'], -row['validation_hr10'], row['depth']))
    return dict(selected_depth=winner['depth'], selected_arm=winner['name'], baseline_retained=winner['depth'] == 1,
                criterion='mean best-validation NDCG@10, then HR@10, then fewer layers',
                rows=sorted(evaluated, key=lambda row: row['depth']), selection_split='val',
                repeated_validation_development=True, test_targets_read=False, deployed=False)


def verify_nested_initializations(parameter_maps):
    """All parameters present in a shallower model must match the deeper one."""
    if 1 not in parameter_maps or not parameter_maps[1]:
        raise AssertionError('The one-layer initial parameter hashes are required.')
    checks = []
    depths = sorted(parameter_maps)
    for lower, upper in zip(depths, depths[1:]):
        shallow, deep = parameter_maps[lower], parameter_maps[upper]
        if not shallow or not set(shallow).issubset(deep):
            raise AssertionError(f'Depth {upper} is missing a shared initial parameter from depth {lower}.')
        mismatched = [name for name in shallow if shallow[name] != deep[name]]
        if mismatched:
            raise AssertionError(f'Unmatched initial parameters for depth {lower}/{upper}: {mismatched[:5]}')
        checks.append(dict(shallower_depth=lower, deeper_depth=upper, matching_parameters=len(shallow)))
    return checks


def parameter_counts(checkpoint):
    saved = torch.load(checkpoint, map_location='meta', mmap=True, weights_only=False)['model']
    embedding_roots = {'item_emb', 'user_emb', 'sparse_emb', 'position_emb', 'action_emb'}
    total = sum(value.numel() for value in saved.values())
    embedding = sum(value.numel() for name, value in saved.items() if name.split('.')[0] in embedding_roots)
    return dict(total=total, embedding=embedding, non_embedding=total - embedding)


def aggregate_groups(diagnostics):
    result = {}
    for group in GROUPS:
        counts = [row['groups'][group]['users'] for row in diagnostics]
        if len(set(counts)) != 1:
            raise AssertionError('Frequency cohorts differ across seeds.')
        row = result[group] = dict(users=counts[0])
        for k in KS:
            for metric in [f'Recall@{k}', f'NDCG@{k}']:
                values = [item['groups'][group][metric] for item in diagnostics]
                row[metric] = dict(mean=float(np.mean(values)) if counts[0] else None,
                                   sample_std=float(np.std(values, ddof=1)) if counts[0] and len(values) > 1 else None,
                                   values=values)
    return result


def reference_parameter_map(root, plan, reference):
    for key in ['initialization_parameter_sha256', 'reference_parameter_sha256']:
        if reference.get(key):
            return reference[key]
    seed = reference['seed']
    initializations = plan.get('initializations', [])
    if isinstance(initializations, dict):
        initializations = [dict(seed=int(key), **value) for key, value in initializations.items()]
    for item in initializations:
        if item['seed'] != seed:
            continue
        for key in ['reference_parameter_sha256', 'initialization_parameter_sha256']:
            if item.get(key):
                return item[key]
        path = item.get('manifest') or item.get('manifest_path')
        if path:
            path = Path(path)
            path = path if path.is_absolute() else root / path
            if item.get('manifest_sha256') and digest(path) != item['manifest_sha256']:
                raise AssertionError('Initial-weight manifest changed.')
            manifest = json.loads(path.read_text())
            for key in ['reference_parameter_sha256', 'initialization_parameter_sha256']:
                if manifest.get(key):
                    return manifest[key]
    raise AssertionError(f'Missing original one-layer initial parameter hashes for seed {seed}.')


def initialization_manifest(root, plan, seed):
    entries = plan.get('initializations', [])
    if isinstance(entries, dict):
        entries = [dict(seed=int(key), **value) for key, value in entries.items()]
    entry = next((item for item in entries if item['seed'] == seed), None)
    if entry is None:
        raise AssertionError(f'Missing initial-weight identity for seed {seed}.')
    path = Path(entry.get('manifest') or entry['manifest_path'])
    path = path if path.is_absolute() else root / path
    if entry.get('manifest_sha256') and digest(path) != entry['manifest_sha256']:
        raise AssertionError('Initial-weight manifest changed.')
    manifest = json.loads(path.read_text())
    weights = Path(entry['file'])
    weights = weights if weights.is_absolute() else root / weights
    fingerprint = digest(weights)
    if manifest['file_sha256'] != fingerprint:
        raise AssertionError('Initial-weight file changed after registration.')
    return weights, manifest, fingerprint


def load_diagnostic(prefix, data, rows, seen, checkpoint, config, history):
    description = json.loads(prefix.with_suffix('.json').read_text())
    if (description['split'] != 'val' or description['test_targets_read'] or description['ks'] != KS
            or not description['filter_seen_items']):
        raise AssertionError('Require validation-only full-catalog coverage at fixed budgets.')
    if (description['candidate_items'] != data.num_items or description['cache_signature'] != data.metadata['preparation_config']
            or Path(description['cache']).resolve() != data.root.resolve()
            or Path(description['checkpoint']).resolve() != checkpoint.resolve()
            or description['checkpoint_sha256'] != digest(checkpoint)):
        raise AssertionError('Diagnostic checkpoint/cache identity differs.')
    for key in ['blocks', 'hidden', 'heads', 'seed', 'objective', 'num_negatives']:
        if description['training'][key] != config[key]:
            raise AssertionError(f'Diagnostic uses a different {key}.')
    best = max(history, key=lambda row: (row['validation']['NDCG@10'], row['validation']['HR@10'], -row['epoch']))
    if description['selected_epoch'] != best['epoch']:
        raise AssertionError('Checkpoint was not selected solely by validation.')
    with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as saved:
        prediction = {name: saved[name].copy() for name in saved.files}
    np.testing.assert_array_equal(prediction['rows'], rows)
    np.testing.assert_array_equal(prediction['targets'], data.items[rows, data.val_pos[rows]])
    np.testing.assert_array_equal(prediction['seen'], seen)
    if (prediction['seen'] == prediction['targets'][:, None]).any():
        raise AssertionError('Held-out validation target appears in its history.')
    groups, per_user = coverage_summary(prediction['predictions'], prediction['targets'], rows, data.train_counts,
                                         KS, prediction['target_content_present'])
    for group in GROUPS:
        for k in KS:
            for metric in [f'Recall@{k}', f'NDCG@{k}']:
                if groups[group]['users']:
                    np.testing.assert_allclose(groups[group][metric], description['groups'][group][metric], atol=1e-12)
    for name, actual in per_user.items():
        np.testing.assert_array_equal(prediction[name], actual)
    cohort_sha = hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()
    if description['evaluation_cohort_sha256'] != cohort_sha or config['evaluation_cohort_sha256'] != cohort_sha:
        raise AssertionError('Frozen cohort differs.')
    return description, prediction


def load_evidence(root, plan):
    specification = plan['specification']
    repo = Path(__file__).resolve().parents[1]
    data = PreparedData((repo / specification['cache']).resolve())
    rows = np.load(repo / specification['evaluation_indices'], allow_pickle=False)
    seen = np.stack([make_input(data.items[row, :data.val_cut[row]],
                               data.actions[row, :data.val_cut[row]], data.max_history)[0] for row in rows])
    arms = {f'depth_{depth}': dict(name=f'depth_{depth}', depth=depth, runs=[], diagnostics=[], predictions=[])
            for depth in specification['depths']}
    jobs = {Path(job['folder']).name: job for job in plan['jobs']}
    initialization_checks = []
    for reference in plan['references']:
        seed = reference['seed']
        control = Path(reference['folder'])
        reference_config = json.loads((control / 'config.json').read_text())
        for name, expected in reference['files'].items():
            if digest(control / name) != expected:
                raise AssertionError('Reference evidence changed after registration.')
        maps = {1: reference_parameter_map(root, plan, reference)}
        initial_path, manifest, initial_sha = initialization_manifest(root, plan, seed)
        if manifest['historical_shared_initialization_sha256'] != reference_config['shared_initialization_sha256']:
            raise AssertionError('Initial-weight bundle is not tied to the historical initial model.')
        for depth in specification['depths']:
            name = f'depth_{depth}'
            folder = control if depth == 1 else Path(jobs[f'{name}_seed{seed}']['folder'])
            config = json.loads((folder / 'config.json').read_text())
            trained = json.loads((folder / 'result.json').read_text())
            history = json.loads((folder / 'history.json').read_text())
            if (config['blocks'] != depth or len(history) != specification['epochs'] or config['max_steps']
                    or not config['skip_test'] or trained['test'] is not None or config.get('item_id_dropout', 0.) != 0):
                raise AssertionError('Require full unmasked matched training without test evaluation.')
            for key in MATCHED_FIELDS:
                if config[key] != reference_config[key]:
                    raise AssertionError(f'{folder.name}: unmatched {key}.')
            if Path(config['cache']).resolve() != data.root.resolve():
                raise AssertionError('Training uses a different data cache.')
            if depth > 1:
                maps[depth] = config['initialization_parameter_sha256']
                expected_initial = manifest['composite_parameter_sha256']
                if any(name not in expected_initial or expected_initial[name] != value
                       for name, value in maps[depth].items()):
                    raise AssertionError('Initialized parameters differ from the registered composite state.')
                if (Path(config['initial_weights']).resolve() != initial_path.resolve()
                        or config['initial_weights_sha256'] != initial_sha
                        or config['common_initialization_sha256'] != reference_config['shared_initialization_sha256']):
                    raise AssertionError('Training did not use the registered shared initialization bundle.')
            checkpoint = folder / 'best.pt'
            description, prediction = load_diagnostic(root / 'diagnostics' / f'{name}_seed{seed}', data, rows,
                                                       seen, checkpoint, config, history)
            if trained['best_epoch'] != description['selected_epoch']:
                raise AssertionError('Training and diagnostic epochs differ.')
            counts = parameter_counts(checkpoint)
            if counts['total'] != config['parameters']:
                raise AssertionError('Reported model size differs from checkpoint.')
            arms[name]['diagnostics'].append(description)
            arms[name]['predictions'].append(prediction)
            arms[name]['runs'].append(dict(seed=seed, folder=str(folder), best_epoch=trained['best_epoch'],
                                            checkpoint_sha256=description['checkpoint_sha256'], parameters=counts,
                                            global_steps=trained['global_steps'],
                                            supervised_positions_per_epoch=[row['supervised_positions'] for row in history],
                                            training_seconds=float(sum(row['training_seconds'] for row in history)),
                                            elapsed_seconds=trained['elapsed_seconds'], peak_gpu_memory_mb=trained['peak_gpu_memory_mb'],
                                            initialization_parameter_count=len(maps[depth]),
                                            initial_weights_sha256=config.get('initial_weights_sha256'),
                                            common_initialization_sha256=config.get('common_initialization_sha256',
                                                                                   reference_config['shared_initialization_sha256']),
                                            history=[dict(epoch=row['epoch'], train_loss=row['train_loss'],
                                                          validation=row['validation']) for row in history]))
        initialization_checks.append(dict(seed=seed, pairs=verify_nested_initializations(maps)))
    control_prediction = arms['depth_1']['predictions'][0]
    baseline_count = arms['depth_1']['runs'][0]['parameters']
    for records in arms.values():
        for prediction in records['predictions']:
            for field in ['rows', 'targets', 'seen', 'target_train_frequency', 'target_content_present']:
                np.testing.assert_array_equal(prediction[field], control_prediction[field])
        for run in records['runs']:
            if (run['parameters']['embedding'] != baseline_count['embedding']
                    or run['parameters']['total'] - baseline_count['total'] != 6464 * (records['depth'] - 1)):
                raise AssertionError('Model capacity differs beyond the declared Transformer depth.')
        records['groups'] = aggregate_groups(records['diagnostics'])
    return data, arms, initialization_checks


def paired_differences(arms):
    reference = arms['depth_1']['predictions']
    counts = reference[0]['target_train_frequency']
    masks = dict(all=np.ones(len(counts), dtype=bool), cold_0=counts == 0,
                 tail_1_5=(counts >= 1) & (counts <= 5), mid_6_50=(counts >= 6) & (counts <= 50), head_gt50=counts > 50)
    result = {}
    for name, records in arms.items():
        if records['depth'] == 1:
            continue
        result[name] = {}
        for group, mask in masks.items():
            result[name][group] = {}
            for field, metric in [('ndcg_at_10', 'NDCG@10'), ('hits_at_10', 'Recall@10'),
                                  ('hits_at_100', 'Recall@100'), ('hits_at_500', 'Recall@500')]:
                if not mask.any():
                    result[name][group][metric] = dict(users=0, mean=None, per_seed=[], note='Empty frequency cohort.')
                    continue
                differences = np.stack([a[field][mask].astype(float) - b[field][mask].astype(float)
                                        for a, b in zip(records['predictions'], reference)])
                result[name][group][metric] = dict(users=int(mask.sum()), direction='deeper minus one layer',
                                                    mean=float(differences.mean()), per_seed=differences.mean(1).tolist(),
                                                    **confidence_intervals(differences))
    return result


def freeze_selection(path, result):
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('Locked depth selection changed; use a new study directory.')
    else:
        path.write_text(json.dumps(result, indent=2) + '\n')


def make_figures(reports, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = reports / 'figures'
    folder.mkdir(exist_ok=True)
    arms = list(result['arms'].values())
    depths = [row['depth'] for row in arms]
    colors = ['#356da6', '#db8a4c', '#4a9779', '#9269b5']
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for axis, (group, metric, factor) in zip(axes, [('all', 'NDCG@10', 1), ('cold_0', 'Recall@500', 100),
                                                 ('tail_1_5', 'Recall@500', 100)]):
        means = [scalar(row, group, metric) * factor for row in arms]
        errors = [row['groups'][group][metric]['sample_std'] * factor for row in arms]
        axis.bar(np.arange(len(arms)), means, yerr=errors, capsize=4, color=colors[:len(arms)])
        axis.set_xticks(np.arange(len(arms)), [str(depth) for depth in depths])
        axis.set_xlabel('Transformer blocks')
        axis.set_title(f'{group}: {metric}' + (' (%)' if factor == 100 else ''))
        axis.grid(axis='y', alpha=.2)
    figure.tight_layout()
    figure.savefig(folder / 'depth_comparison.png', dpi=160)
    plt.close(figure)
    seeds = result['specification']['seeds']
    figure, axes = plt.subplots(1, len(seeds), figsize=(4 * len(seeds), 3.5), squeeze=False)
    for index, seed in enumerate(seeds):
        for row, color in zip(arms, colors):
            history = row['runs'][index]['history']
            axes[0, index].plot([item['epoch'] for item in history],
                                [item['validation']['NDCG@10'] for item in history],
                                marker='o', label=f"{row['depth']} block(s)", color=color)
        axes[0, index].set_title(f'Seed {seed}')
        axes[0, index].set_xlabel('Epoch')
        axes[0, index].set_ylabel('Validation NDCG@10')
        axes[0, index].legend(fontsize=8)
        axes[0, index].grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(folder / 'validation_curves.png', dpi=160)
    plt.close(figure)


def percent(value):
    return f'{value:.4%}' if value is not None else 'N/A'


def outcome_summary(result):
    """Derive interpretation from the selected arm, not a hardcoded depth."""
    control_name = next(name for name, arm in result['arms'].items() if arm['depth'] == 1)
    selected_name = result['model_selection']['selected_arm']
    control, selected = result['arms'][control_name], result['arms'][selected_name]
    fixed = {name: {metric: float(np.mean([row[metric] for row in rows]))
                    for metric in ['NDCG@10', 'HR@10']}
             for name, rows in result['fixed_epoch5_validation'].items()}
    baseline_time = float(np.mean([run['training_seconds'] for run in control['runs']]))
    selected_time = float(np.mean([run['training_seconds'] for run in selected['runs']]))
    interval = (result['paired_differences'][selected_name]['all']['NDCG@10']['paired_seed_and_user_95pct']
                if selected_name != control_name else None)
    return dict(selected_name=selected_name, selected_depth=selected['depth'], fixed_epoch5_means=fixed,
                ndcg_interval=interval,
                ndcg_interval_crosses_zero=(interval[0] <= 0 <= interval[1]) if interval is not None else None,
                tail_recall500_improved=scalar(selected, 'tail_1_5', 'Recall@500') > scalar(control, 'tail_1_5', 'Recall@500'),
                best_mean_higher=scalar(selected, 'all', 'NDCG@10') > scalar(control, 'all', 'NDCG@10'),
                selected_fixed_epoch5_below_control=fixed[selected_name]['NDCG@10'] < fixed[control_name]['NDCG@10'],
                training_time_relative=(selected_time / baseline_time - 1) if baseline_time > 0 else None)


def write_readme(reports, result):
    selection = result['model_selection']
    outcome = outcome_summary(result)
    summary = []
    if outcome['selected_depth'] != 1:
        lo, hi = outcome['ndcg_interval']
        summary.append(f"相对一层，所选 {outcome['selected_depth']} 层的 NDCG@10 差值配对 95% 区间为 [{lo:+.6f}, {hi:+.6f}]。"
                       + ('区间跨过 0，不能把验证均值领先表述成已确认的稳定收益。'
                          if outcome['ndcg_interval_crosses_zero'] else
                          '区间未跨过 0，但这里已使用同一验证集选参，仍不能当作独立确认。'))
        summary.append('所选深度的长尾 Recall@500 没有改善。' if not outcome['tail_recall500_improved']
                       else '所选深度的长尾 Recall@500 均值提高，证据强弱仍需结合分桶配对区间。')
        if outcome['training_time_relative'] is not None:
            summary.append(f"本机记录的训练部分耗时相对一层变化 {outcome['training_time_relative']:+.1%}；这是共享机器上的描述性成本，不是严格性能结论。")
    repo = Path(__file__).resolve().parents[1]
    config_link = os.path.relpath(repo / 'configs/depth_study.json', reports)
    runner_link = os.path.relpath(repo / 'scripts/run_depth_study.py', reports)
    lines = ['# Transformer 深度对照：1／2／4 层', '',
             f"依据三种子平均最佳验证 NDCG@10，选中 **{selection['selected_depth']} 层**；这是复用验证集的开发选择，尚无独立测试确认。", '',
             *summary, '',
             '## 固定了什么', '',
             f"同一 {result['training_users']:,} 名训练用户、{result['candidate_items']:,} 个候选和 {result['evaluation_users']:,} 名验证用户。隐藏维度 32、单头、FFN 32→32→32、点击 InfoNCE-64、batch size 128、学习率与五轮预算保持一致，只改变 Transformer 层数。",
             '每个 checkpoint 按整体验证 NDCG@10、再按 HR@10 选 epoch，完全相同保留较早 epoch；深度方案按三个种子的均值选择，平分时优先更少层。冷启动、长尾指标报告但未被临时用作换选准则。', '',
             '### 共享初值不只靠相同 seed', '',
             '复用形状相同的编码器、第一层及最终归一化等参数的同一份初始权重；2 层与 4 层模型还共享第二层初值。逐参数 SHA 已核对。新增层独立初始化，没有使用已训练 checkpoint 给深模型热启动。不同层数完整参数集合不同，不要求 whole-model 初始化 SHA 相同。', '',
             '数据打乱和负采样随机流保持原设置。更深模型执行更多普通 dropout，因此后续 dropout 随机流不保证逐位置相同；也不把相同 epoch 称为等 FLOPs。', '',
             '## 主指标与候选覆盖：三种子均值', '',
             '| 层数 | NDCG@10 | HR@10 | 全体 Recall@500 | 冷启动 Recall@500 | 长尾 Recall@500 |',
             '|---:|---:|---:|---:|---:|---:|']
    for arm in result['arms'].values():
        lines.append(f"| {arm['depth']} | {scalar(arm,'all','NDCG@10'):.6f} | {percent(scalar(arm,'all','Recall@10'))} | "
                     f"{percent(scalar(arm,'all','Recall@500'))} | {percent(scalar(arm,'cold_0','Recall@500'))} | {percent(scalar(arm,'tail_1_5','Recall@500'))} |")
    lines += ['', '![深度与分桶覆盖](figures/depth_comparison.png)', '',
              '图中误差线是三个训练种子的样本标准差，下面表格中的区间才是用户与种子联合配对 bootstrap。', '',
              '![逐种子验证曲线](figures/validation_curves.png)', '',
              '## 每个种子的最佳验证 checkpoint', '',
              '| 层数 | 种子 | 最佳 epoch | 验证 NDCG@10 | 验证 HR@10 |', '|---:|---:|---:|---:|---:|']
    for arm in result['arms'].values():
        for run, diagnostic in zip(arm['runs'], arm['diagnostics']):
            row = diagnostic['groups']['all']
            lines.append(f"| {arm['depth']} | {run['seed']} | {run['best_epoch']} | {row['NDCG@10']:.6f} | {percent(row['Recall@10'])} |")
    lines += ['', '## 固定第 5 轮验证：不按最佳 epoch 取值', '',
              '下面每个模型都使用第 5 轮指标，再对三个种子取均值；与上面的“各自最佳验证 checkpoint”是不同统计口径。', '',
              '| 层数 | 第 5 轮平均 NDCG@10 | 第 5 轮平均 HR@10 |', '|---:|---:|---:|']
    for name, arm in result['arms'].items():
        row = outcome['fixed_epoch5_means'][name]
        lines.append(f"| {arm['depth']} | {row['NDCG@10']:.6f} | {percent(row['HR@10'])} |")
    if outcome['best_mean_higher'] and outcome['selected_fixed_epoch5_below_control']:
        lines += ['', f"所选 {outcome['selected_depth']} 层的最佳 checkpoint 均值较高，但固定第 5 轮的平均 NDCG@10 低于一层。当前优势依赖验证集选择较早的 epoch，不能概括为“网络越深越好”，也不代表训练到相同最后一轮仍有收益。"]
    else:
        lines += ['', '固定末轮表用于检查最佳 epoch 选择的影响；不同深度的收敛和过拟合轨迹可能不同，不能只凭一个最佳均值概括为“越深越好”。']
    lines += ['', '## 相对一层的配对差值', '',
              '同一个种子、同一个用户成对比较；只有三个训练种子，不能由 bootstrap 推出所有初始化和所有时间段都有效。参数选择已使用本验证集，区间只用于解释开发证据。', '',
              '| 方案 | 分组与指标 | 平均差值 | 95% 用户与种子配对区间 |', '|---|---|---:|---|']
    for name, groups in result['paired_differences'].items():
        for group, metric in [('all','NDCG@10'), ('all','Recall@10'), ('cold_0','Recall@500'), ('tail_1_5','Recall@500')]:
            row = groups[group][metric]
            if row['mean'] is None:
                continue
            lo, hi = row['paired_seed_and_user_95pct']
            lines.append(f"| {name} | {group}/{metric} | {row['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', '## 参数与成本', '',
              '| 层数 | 总参数 | Embedding 参数 | 非 Embedding 参数 | 平均五轮训练秒数 | 平均峰值显存 MiB |',
              '|---:|---:|---:|---:|---:|---:|']
    for arm in result['arms'].values():
        runs = arm['runs']
        counts = runs[0]['parameters']
        memory = [run['peak_gpu_memory_mb'] for run in runs if run['peak_gpu_memory_mb'] is not None]
        mem = f'{np.mean(memory):.1f}' if memory else 'N/A'
        lines.append(f"| {arm['depth']} | {counts['total']:,} | {counts['embedding']:,} | {counts['non_embedding']:,} | "
                     f"{np.mean([run['training_seconds'] for run in runs]):.1f} | {mem} |")
    lines += ['', '训练秒数为每轮训练部分之和，不含验证。参考一层模型来自此前运行，机器共享负载不同，所以这些时间是描述性成本，不是严格速度基准。加层不改变 32 维检索向量和 Embedding 表；每层新增 6,464 个参数。', '',
              '## 应怎样解读', '',
              '这组实验直接回答固定训练设置下，序列网络深度是否值得增加。即使深模型更好，也不能因此说冷启动与长尾已解决；需同时查看分桶和候选覆盖。即使更深没有变好，也只说明当前学习率、五轮与该初始化协议下未观察到收益，不能证明深网络普遍无用。',
              '一层注意力已经可以读取全部可见历史；增加层数增加组合交互能力。当前对照还没有改变 FFN 扩张宽度或 hidden 维度，下一步应据结果决定是否单独比较它们。', '',
              '## 证据与复现', '',
              '- `metrics.json`：完整分桶 Recall@10/100/500、全部种子、配对区间及固定第 5 轮的验证结果。',
              '- `model_selection.json`：只按验证集选择的层数、所有候选及输入 checkpoint SHA。',
              '- `verification.json`：训练完成、初值逐参数、checkpoint、目录和逐用户指标检查。',
              f'- [配置]({config_link})、[运行入口]({runner_link})。', '',
              '```bash', '.project_venv/bin/python scripts/run_depth_study.py \\',
              '  --root artifacts/depth_study_reproduction \\',
              '  --report-dir reports/depth_study_reproduction --gpus 0,2,3,4,5,6', '```', '',
              '已有训练结果可追加 `--stage finish` 继续覆盖诊断和报告。新配置使用新目录；原始 checkpoint、主模型和 Faiss 索引没有因此被覆盖。本轮仅构造验证 Dataset；缓存虽含测试位置元数据，未用它们访问测试目标或选模型。', '']
    (reports / 'README.md').write_text('\n'.join(lines))


def report(root, reports):
    root, reports = Path(root).resolve(), Path(reports).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    data, evidence, initialization_checks = load_evidence(root, plan)
    summaries = {name: {key: value for key, value in arm.items() if key not in ['predictions']}
                 for name, arm in evidence.items()}
    selection = select_depth(list(summaries.values()))
    selection['input_checkpoint_sha256'] = {name: [run['checkpoint_sha256'] for run in arm['runs']]
                                             for name, arm in summaries.items()}
    freeze_selection(root / 'model_selection.json', selection)
    result = dict(specification=plan['specification'], training_users=data.num_users, candidate_items=data.num_items,
                  evaluation_users=len(evidence['depth_1']['predictions'][0]['rows']), arms=summaries,
                  paired_differences=paired_differences(evidence), model_selection=selection,
                  nested_initialization_checks=initialization_checks, test_targets_read=False,
                  fixed_epoch5_validation={name: [run['history'][-1]['validation'] for run in arm['runs']]
                                           for name, arm in summaries.items()})
    reports.mkdir(parents=True, exist_ok=True)
    for name, value in [('metrics.json',result), ('plan.json',plan), ('model_selection.json',selection)]:
        (reports / name).write_text(json.dumps(value, indent=2) + '\n')
    verification = dict(new_complete_training_runs=sum(len(arm['runs']) for arm in summaries.values() if arm['depth'] != 1),
                        reused_one_layer_runs=len(summaries['depth_1']['runs']), nested_initial_parameter_hashes_matched=True,
                        first_two_layers_shared_between_depth2_and_depth4=True,
                        same_rows_targets_and_history=True, same_candidate_catalog=True, per_user_metrics_recomputed=True,
                        full_five_epochs=True, validation_only_checkpoint_and_depth_selection=True,
                        original_model_or_index_replaced=False, independent_test_confirmation=False,
                        report_source_sha256=digest(Path(__file__)))
    (reports / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    make_figures(reports, result)
    write_readme(reports, result)
    print(json.dumps(dict(report=str(reports / 'README.md'), selection=selection), indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/depth_study')
    parser.add_argument('--report-dir', default='reports/depth_study')
    args = parser.parse_args()
    report(args.root, args.report_dir)
