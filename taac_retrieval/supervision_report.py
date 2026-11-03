"""Paired evidence for the supplementary all-event/click-only InfoNCE ablation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import PreparedData, load_array
from .metrics import recommendation_metrics
from .week_report import confidence_intervals


METRICS = ['HR@10', 'NDCG@10']
ARMS = ['click_infonce', 'all_infonce']


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_prediction(path, expected):
    with np.load(path, allow_pickle=False) as saved:
        prediction = {key: saved[key].copy() for key in saved.files}
    hits, ndcg = recommendation_metrics(prediction['predictions'], prediction['targets'])
    np.testing.assert_allclose([hits.mean(), ndcg.mean()],
                               [expected[key] for key in METRICS], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(prediction['hits'], hits)
    np.testing.assert_allclose(prediction['ndcg'], ndcg)
    if len(np.unique(prediction['rows'])) != len(prediction['rows']):
        raise AssertionError('Duplicate evaluation users.')
    return prediction


def aggregate(runs, predictions):
    results = {}
    for split in ['validation', 'test']:
        rows = {}
        for arm in ARMS:
            rows[arm] = {}
            for metric in METRICS:
                values = [run[split][metric] for run in runs[arm]]
                rows[arm][metric] = dict(mean=float(np.mean(values)),
                                          sample_std=float(np.std(values, ddof=1)), values=values)
        differences = {}
        for metric, field in [('HR@10', 'hits'), ('NDCG@10', 'ndcg')]:
            paired = np.stack([b[field] - a[field] for a, b in
                               zip(predictions['click_infonce'][split], predictions['all_infonce'][split])])
            control = rows['click_infonce'][metric]['mean']
            experimental = rows['all_infonce'][metric]['mean']
            differences[metric] = dict(direction='all minus click', absolute=float(paired.mean()),
                                        relative_to_click=(experimental / control - 1) if control else None,
                                        per_seed=paired.mean(1).tolist(), **confidence_intervals(paired))
        results[split] = dict(aggregates=rows, differences=differences)
    return results


def grouped_test(data, predictions):
    reference = predictions['click_infonce']['test'][0]
    counts = data.train_counts[reference['targets']]
    present = load_array(data.root, 'mm_present')[reference['targets']]
    groups = {'cold_0': counts == 0, 'tail_1_5': (counts >= 1) & (counts <= 5),
              'mid_6_50': (counts >= 6) & (counts <= 50), 'head_gt50': counts > 50,
              'content_present': present, 'content_missing': ~present}
    result = {}
    for name, mask in groups.items():
        result[name] = dict(users=int(mask.sum()))
        for arm in ARMS:
            result[name][arm] = {metric: float(np.stack([p[field] for p in predictions[arm]['test']])[:, mask].mean())
                                 if mask.any() else None
                                 for metric, field in [('HR@10', 'hits'), ('NDCG@10', 'ndcg')]}
    return result


def load_evidence(root, plan):
    runs = {arm: [] for arm in ARMS}
    predictions = {arm: dict(validation=[], test=[]) for arm in ARMS}
    new_jobs = {job['arguments']['seed']: job for job in plan['jobs']}
    reference_root = Path(plan['references'][0]['output']).parent.parent
    for reference in plan['references']:
        seed = reference['seed']
        folders = dict(click_infonce=Path(reference['output']),
                       all_infonce=Path(new_jobs[seed]['output']))
        for name, expected in reference['files'].items():
            if digest(folders['click_infonce'] / name) != expected:
                raise AssertionError('An immutable reference artifact changed.')
        configs = {}
        for arm in ARMS:
            folder = folders[arm]
            cfg = configs[arm] = json.loads((folder / 'config.json').read_text())
            trained = json.loads((folder / 'result.json').read_text())
            history = json.loads((folder / 'history.json').read_text())
            if cfg['objective'] != arm or not cfg['skip_test'] or trained['test'] is not None:
                raise AssertionError('Wrong objective or training consumed test labels.')
            if cfg['max_steps'] != 0 or len(history) != plan['specification']['epochs']:
                raise AssertionError('Require full training, not a smoke run.')
            best = max(history, key=lambda row: (row['validation']['NDCG@10'],
                                                row['validation']['HR@10'], -row['epoch']))
            if trained['best_epoch'] != best['epoch']:
                raise AssertionError('Wrong best epoch.')
            pred = load_prediction(folder / f"validation_epoch{best['epoch']}.npz", best['validation'])
            predictions[arm]['validation'].append(pred)
            evaluation = (reference_root / 'evaluation' / f'selected_seed{seed}' if arm == 'click_infonce'
                          else root / 'evaluation' / f'all_infonce_seed{seed}')
            evaluated = json.loads(evaluation.with_suffix('.json').read_text())
            if (evaluated['selected_epoch'] != best['epoch'] or evaluated['split'] != 'test'
                    or evaluated['checkpoint_sha256'] != digest(folder / 'best.pt')
                    or evaluated['evaluation_cohort_sha256'] != cfg['evaluation_cohort_sha256']):
                raise AssertionError('Evaluation is not tied to the locked checkpoint/cohort.')
            pred = load_prediction(evaluation.with_suffix('.npz'), evaluated['metrics'])
            predictions[arm]['test'].append(pred)
            runs[arm].append(dict(seed=seed, output=str(folder), best_epoch=best['epoch'],
                                  validation=best['validation'], test=evaluated['metrics'],
                                  epoch5_validation=history[-1]['validation'],
                                  validation_history=[dict(epoch=row['epoch'], metrics=row['validation'],
                                                           train_loss=row['train_loss']) for row in history],
                                  supervised_positions_per_epoch=[row['supervised_positions'] for row in history],
                                  training_seconds=float(sum(row['training_seconds'] for row in history)),
                                  elapsed_seconds=trained['elapsed_seconds'],
                                  global_steps=trained['global_steps'], peak_gpu_memory_mb=trained['peak_gpu_memory_mb'],
                                  checkpoint_sha256=evaluated['checkpoint_sha256'],
                                  shared_initialization_sha256=cfg['shared_initialization_sha256'],
                                  evaluation_cohort_sha256=cfg['evaluation_cohort_sha256']))
        for key, expected in new_jobs[seed]['arguments'].items():
            if key == 'objective':
                continue
            if configs['click_infonce'][key] != expected or configs['all_infonce'][key] != expected:
                raise AssertionError(f'Unmatched {key} for seed {seed}.')
        for key in ['parameters', 'shared_initialization_sha256', 'evaluation_cohort_sha256',
                    'torch_version', 'numpy_version', 'device_name']:
            if configs['click_infonce'][key] != configs['all_infonce'][key]:
                raise AssertionError(f'Unmatched {key}.')
    for split in ['validation', 'test']:
        reference = predictions['click_infonce'][split][0]
        for arm in ARMS:
            for prediction in predictions[arm][split]:
                for field in ['rows', 'targets']:
                    np.testing.assert_array_equal(prediction[field], reference[field])
    return runs, predictions


def make_figure(reports, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['#356da6', '#dc8351']
    for axis, metric in zip(axes, METRICS):
        multiplier = 100 if metric == 'HR@10' else 1
        summary = result['comparison']['validation']['aggregates']
        means = [summary[arm][metric]['mean'] * multiplier for arm in ARMS]
        errors = [summary[arm][metric]['sample_std'] * multiplier for arm in ARMS]
        axis.bar([0, 1], means, yerr=errors, capsize=4, color=colors, alpha=.8)
        for position, arm in enumerate(ARMS):
            values = np.array(summary[arm][metric]['values']) * multiplier
            axis.scatter(position + np.linspace(-.08, .08, len(values)), values, color='#222222', s=18, zorder=5)
        axis.set_xticks([0, 1], ['Click only', 'Exposure + click'])
        axis.set_title(f"Best-validation {metric}" + (' (%)' if multiplier == 100 else ''))
        axis.grid(axis='y', alpha=.2)
    figure.suptitle('InfoNCE-64 supervision ablation; dots = training seeds')
    figure.tight_layout()
    figure.savefig(reports / 'validation_comparison.png', dpi=160)
    plt.close(figure)
    seeds = result['protocol']['seeds']
    figure, axes = plt.subplots(1, len(seeds), figsize=(4 * len(seeds), 3.5), squeeze=False)
    for index, seed in enumerate(seeds):
        axis = axes[0, index]
        for arm, label, color in zip(ARMS, ['Click only', 'Exposure + click'], colors):
            history = result['runs'][arm][index]['validation_history']
            axis.plot([row['epoch'] for row in history],
                      [row['metrics']['NDCG@10'] for row in history], marker='o', label=label, color=color)
        axis.set_title(f'Seed {seed}')
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Validation NDCG@10')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(reports / 'validation_curves.png', dpi=160)
    plt.close(figure)


def write_readme(reports, result):
    means = result['comparison']['validation']['aggregates']
    all_mean = means['all_infonce']['NDCG@10']['mean']
    click_mean = means['click_infonce']['NDCG@10']['mean']
    if np.isclose(all_mean, click_mean, rtol=0, atol=1e-12):
        verdict = '两个方案的三种子平均最佳验证 NDCG@10 相同。'
    elif all_mean > click_mean:
        verdict = '全部已知事件监督的三种子平均最佳验证 NDCG@10 更高。'
    else:
        verdict = '仅点击监督的三种子平均最佳验证 NDCG@10 更高。'
    difference = result['comparison']['validation']['differences']['NDCG@10']
    lo, hi = difference['paired_seed_and_user_95pct']
    uncertainty = ('用户与种子联合配对区间跨过 0，当前三种子结果不足以确认稳定优劣。'
                   if lo <= 0 <= hi else
                   '用户与种子联合配对区间没有跨过 0，支持本地协议下的差异；仍只覆盖三个训练种子。')
    lines = ['# 补充消融：全部事件 vs 仅点击，固定 InfoNCE-64', '', verdict + uncertainty, '',
             '## 实验问题与控制', '',
             '在相同 InfoNCE 损失和每个监督位置 64 个负样本下，曝光事件是否应当提供下一事件正样本监督？', '',
             '- `click_infonce`：只监督下一事件为点击的位置；曝光仍保留在输入历史。',
             '- `all_infonce`：下一事件为曝光或点击均监督，每个位置等权。',
             '- 未知行为与填充位置在两组中均不提供监督；曝光不是负样本。',
             f"- 同一缓存、固定队列、候选库、网络、学习率、初值、batch size {result['training_settings']['batch_size']}、{result['training_settings']['epochs']} 个 epoch、温度 {result['training_settings']['temperature']:g}。",
             '- 沿用 seed 2026/2027/2028 的已有点击版，新增三次全部事件版训练；共同初值 SHA 已逐个核对。',
             '- 负样本池、分布与排除策略相同；mask 改变会改变随机数消耗，实际负例不保证逐位置相同。',
             '- 监督位置数和负例总量不同，不是等算力或等训练时长的实验。', '',
             f"数据：{result['training_users']:,} 用户，{result['candidate_items']:,} 候选，{result['evaluation_users']:,} 名固定评估用户。", '',
             '## 主比较：最佳验证 checkpoint', '',
             '每个模型均只按验证 NDCG@10、再按 HR@10 选择 epoch；完全相同则保留较早 epoch。', '',
             '| 方案 | 种子 | 最佳 epoch | 验证 HR@10 | 验证 NDCG@10 |', '|---|---:|---:|---:|---:|']
    for arm, title in [('click_infonce', '仅点击'), ('all_infonce', '全部事件')]:
        for run in result['runs'][arm]:
            lines.append(f"| {title} | {run['seed']} | {run['best_epoch']} | {run['validation']['HR@10']:.4%} | {run['validation']['NDCG@10']:.6f} |")
    lines += ['', '| 验证均值 | HR@10 | NDCG@10 |', '|---|---:|---:|']
    for arm, title in [('click_infonce', '仅点击'), ('all_infonce', '全部事件')]:
        row = means[arm]
        lines.append(f"| {title} | {row['HR@10']['mean']:.4%} | {row['NDCG@10']['mean']:.6f} |")
    lines += ['', '![验证指标与种子分布](validation_comparison.png)', '',
              '![逐种子的验证曲线](validation_curves.png)', '',
              '以下差值统一定义为“全部事件 − 仅点击”；负值支持仅点击更好。区间对用户与三个训练种子联合配对 bootstrap。', '',
              '| 验证差值 | 平均差值 | 95% 区间 |', '|---|---:|---|']
    for metric in METRICS:
        difference = result['comparison']['validation']['differences'][metric]
        lo, hi = difference['paired_seed_and_user_95pct']
        lines.append(f"| {metric} | {difference['absolute']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', '只有三个训练种子，bootstrap 不等于证明所有初始化都有效。固定第 5 轮的验证结果也保存在 `metrics.json`，便于检查最佳 epoch 选择的影响。', '',
              '这组对照回答的是“固定 InfoNCE-64 时，增加曝光正目标是否有帮助”。监督范围的变化同时带来更多监督位置和负例总量，不能解读为等算力条件下的纯标签质量比较。也不能用它把旧基线到 InfoNCE 的全部收益归到某一个因素上。', '',
              '## 补充测试结果', '',
              '**这批用户的原测试结果此前已被查看。本次是用户提出后的补充消融，不是新的独立确认实验。**',
              '三个新 checkpoint 全部按验证固定后才计算其测试指标；没有重新挑选原模型或修改已有索引。测试是辅助证据，主比较仍是上面的验证结果。', '',
              '| 方案 | 种子 | 测试 HR@10 | 测试 NDCG@10 |', '|---|---:|---:|---:|']
    for arm, title in [('click_infonce', '仅点击'), ('all_infonce', '全部事件')]:
        for run in result['runs'][arm]:
            lines.append(f"| {title} | {run['seed']} | {run['test']['HR@10']:.4%} | {run['test']['NDCG@10']:.6f} |")
        summary = result['comparison']['test']['aggregates'][arm]
        lines.append(f"| {title}均值 | — | {summary['HR@10']['mean']:.4%} | {summary['NDCG@10']['mean']:.6f} |")
    lines += ['', '| 测试差值：全部事件 − 仅点击 | 平均差值 | 95% 用户与种子配对区间 |',
              '|---|---:|---|']
    for metric in METRICS:
        delta = result['comparison']['test']['differences'][metric]
        lo, hi = delta['paired_seed_and_user_95pct']
        lines.append(f"| {metric} | {delta['absolute']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    validation_delta = result['comparison']['validation']['differences']['NDCG@10']['absolute']
    test_delta = result['comparison']['test']['differences']['NDCG@10']['absolute']
    if validation_delta * test_delta < 0:
        lines += ['', '验证与测试的平均 NDCG 差值方向不同，应结合配对区间与种子分布判断，不能仅挑更有利的一张表。原模型选择继续保留，不依据本次已查看测试队列的微小差异更换主模型。']
    lines += ['', '## 监督预算与实测成本', '',
              '| 方案 | 每轮监督位置 | 平均 5 轮训练秒数 | 平均训练过程峰值显存 MiB |', '|---|---:|---:|---:|']
    for arm, title in [('click_infonce', '仅点击'), ('all_infonce', '全部事件')]:
        records = result['runs'][arm]
        memory = [r['peak_gpu_memory_mb'] for r in records if r['peak_gpu_memory_mb'] is not None]
        memory_text = f'{np.mean(memory):.1f}' if memory else 'N/A (CPU)'
        lines.append(f"| {title} | {records[0]['supervised_positions_per_epoch'][0]:,} | "
                     f"{np.mean([r['training_seconds'] for r in records]):.1f} | {memory_text} |")
    lines += ['', '训练秒数为各 epoch 的训练部分之和；参考组是此前运行，实验组是本次运行，机器共享负载不同，因此时间是实测描述，不是严格性能基准。显存为 Torch 记录的 allocated 峰值，不等于 nvidia-smi 的 reserved/进程总量。', '',
              '## 目标广告频次分组：测试 HR@10', '',
              '| 分组 | 用户数 | 仅点击 | 全部事件 |', '|---|---:|---:|---:|']
    for name in ['cold_0', 'tail_1_5', 'mid_6_50', 'head_gt50']:
        row = result['test_groups'][name]
        values = [f"{row[arm]['HR@10']:.4%}" if row[arm]['HR@10'] is not None else 'N/A' for arm in ARMS]
        lines.append(f"| {name} | {row['users']} | {values[0]} | {values[1]} |")
    lines += ['', '## 证据与复现', '',
              '- [配置](../../configs/infonce_supervision.json)、[编排入口](../../scripts/run_infonce_supervision.py)。',
              '- [监督策略与损失](../../taac_retrieval/objectives.py)、[训练分支](../../taac_retrieval/train.py)。',
              '- `metrics.json`：逐种子验证/测试、固定第 5 轮、配对区间、分桶与预算。',
              '- `plan.json`：训练前固定的协议、参考产物 SHA 和源码 SHA；原件及源码快照在实验目录中。',
              '- `verification.json`：配对检查与文件身份核对。', '', '```bash',
              '.project_venv/bin/python scripts/run_infonce_supervision.py \\',
              '  --root artifacts/infonce_supervision_reproduction \\',
              '  --report-dir reports/infonce_supervision_reproduction --gpus 0,2,4', '```', '',
              '完成训练的目录可用 `--stage finish` 继续评估和报告。已有点击版作为固定参考复用，原实验报告、模型选择和 Faiss 索引不会被覆盖。', '']
    (reports / 'README.md').write_text('\n'.join(lines))


def report(root, reports):
    root, reports = Path(root).resolve(), Path(reports).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    runs, predictions = load_evidence(root, plan)
    cache = Path(plan['jobs'][0]['arguments']['cache'])
    data = PreparedData(cache)
    comparison = aggregate(runs, predictions)
    result = dict(protocol=plan['specification'], training_settings=plan['jobs'][0]['arguments'],
                  training_users=data.num_users, candidate_items=data.num_items,
                  evaluation_users=len(predictions['click_infonce']['test'][0]['rows']), runs=runs,
                  comparison=comparison, test_groups=grouped_test(data, predictions),
                  reference_artifacts_reused=True, original_selection_changed=False,
                  fixed_epoch5_validation={arm: {metric: [run['epoch5_validation'][metric] for run in runs[arm]]
                                                 for metric in METRICS} for arm in ARMS})
    result['supervision_budget'] = {
        arm: dict(known_positions_per_epoch=runs[arm][0]['supervised_positions_per_epoch'][0],
                  negative_draws_per_epoch=runs[arm][0]['supervised_positions_per_epoch'][0]
                  * plan['specification']['num_negatives']) for arm in ARMS}
    reports.mkdir(parents=True, exist_ok=True)
    (reports / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    (reports / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    verification = dict(completed_new_training_runs=len(runs['all_infonce']),
                        reused_click_training_runs=len(runs['click_infonce']),
                        same_shared_initialization=True, same_training_configuration_except_objective=True,
                        same_cohort_and_target_ids=True, full_five_epochs=True,
                        validation_only_checkpoint_selection=True, scores_recomputed_from_predictions=True,
                        reference_checksums_verified=True, post_hoc_test_caveat_disclosed=True,
                        report_source_sha256=digest(Path(__file__)))
    (reports / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    make_figure(reports, result)
    write_readme(reports, result)
    print(json.dumps(dict(report=str(reports / 'README.md'), validation=comparison['validation'],
                          test=comparison['test']), indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/infonce_supervision')
    parser.add_argument('--report-dir', default='reports/infonce_supervision')
    args = parser.parse_args()
    report(args.root, args.report_dir)
