"""Exploratory fixed-budget fusion protecting already-retrieved rare items.

This is a separately registered validation-driven follow-up to the failed naive
quota fusion. It neither changes that result nor constitutes confirmation.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from .candidate_fusion import GROUPS, aligned_inputs, mean_groups, select_quota
from .coverage import coverage_summary, file_digest
from .objective_report import confidence_intervals


def protected_candidates(baseline, extra, train_counts, quota, budget=500, protected=10, rare_max=5):
    """Replace only bottom-ranked nonrare base items with unseen rare IDs."""
    baseline, extra, counts = map(np.asarray, (baseline, extra, train_counts))
    if (baseline.ndim != 2 or extra.ndim != 2 or len(baseline) != len(extra)
            or not len(baseline) or baseline.shape[1] < budget
            or not np.issubdtype(baseline.dtype, np.integer)
            or not np.issubdtype(extra.dtype, np.integer)):
        raise ValueError('Expected aligned integer candidates and a complete baseline.')
    if budget < protected or protected < 1 or quota < 0 or quota > budget - protected or rare_max < 0:
        raise ValueError('Invalid candidate budget, protected prefix, quota or rare threshold.')
    if counts.ndim != 1 or (counts < 0).any():
        raise ValueError('Training frequencies must be a nonnegative one-dimensional array.')
    result = np.empty((len(baseline), budget), dtype=np.int32)
    replaced = np.zeros(len(baseline), dtype=np.int32)
    positions = np.arange(budget)
    for row, (base_row, extra_row) in enumerate(zip(baseline, extra)):
        base_row = base_row[:budget]
        base_set = set(map(int, base_row))
        if (base_row <= 0).any() or base_row.max() >= len(counts) or len(base_set) != budget:
            raise ValueError('Baseline must contain budget unique nonpadding catalog IDs.')
        if (extra_row >= len(counts)).any():
            raise ValueError('Extra candidate exceeds the catalog.')
        rare = counts[base_row] <= rare_max
        removable = np.flatnonzero((~rare) & (positions >= protected))
        eligible, extra_seen = [], set()
        if quota and len(removable):
            for item in extra_row:
                item = int(item)
                if item <= 0 or item in base_set or item in extra_seen or counts[item] > rare_max:
                    continue
                eligible.append(item)
                extra_seen.add(item)
                if len(eligible) == quota:
                    break
        count = min(quota, len(eligible), len(removable))
        keep = np.ones(budget, dtype=bool)
        if count:
            keep[removable[-count:]] = False
        merged = np.concatenate([base_row[keep], np.asarray(eligible[:count], dtype=np.int32)])
        if len(merged) != budget or len(np.unique(merged)) != budget:
            raise AssertionError('Protected fusion changed the candidate budget or duplicated an ID.')
        if not set(map(int, base_row[rare])).issubset(set(map(int, merged))):
            raise AssertionError('An already-retrieved rare candidate was removed.')
        result[row], replaced[row] = merged, count
    np.testing.assert_array_equal(result[:, :protected], baseline[:, :protected])
    return result, replaced


def run(args):
    output, report_path = Path(args.output), Path(args.report)
    if (output / 'metrics.json').exists() or (output / 'selection.json').exists():
        raise FileExistsError('Protected fusion results exist; preserve them and use a new output directory.')
    plan_path, config_path = Path(args.plan), Path(args.config)
    plan = json.loads(plan_path.read_text())
    config = json.loads(config_path.read_text())
    if (plan.get('split') != 'val' or plan.get('test_targets_read') is not False
            or not plan.get('developer_evidence_only') or plan['candidate_budget'] != 500
            or not plan['preserve_top10']):
        raise ValueError('Require the separate validation-only fixed-500 follow-up plan.')
    if plan['quotas'] != [0, 50, 100, 250] or plan['protected_training_frequency_max'] != 5:
        raise ValueError('The registered rare-protection policy changed.')
    plan_sha = file_digest(plan_path)
    original_metrics = Path(args.original_fusion) / 'metrics.json'
    original_selection = Path(args.original_fusion) / 'selection.json'
    original_sha = {str(path): file_digest(path) for path in [original_metrics, original_selection]}
    cache, seeds = Path(config['cache']), config['seeds']
    metadata = json.loads((cache / 'metadata.json').read_text())
    counts = np.load(cache / 'train_counts.npy', mmap_mode='r')
    present = np.load(cache / 'mm_present.npy', mmap_mode='r')
    extra_prefix = Path(args.extra_prefix)
    extra_meta = json.loads(extra_prefix.with_suffix('.json').read_text())
    with np.load(extra_prefix.with_suffix('.npz'), allow_pickle=False) as source:
        extra = {key: source[key].copy() for key in ['rows', 'targets', 'seen', 'predictions']}
    quotas = plan['quotas']
    summaries = {quota: [] for quota in quotas}
    replacements = {quota: [] for quota in quotas}
    candidates, details, baselines, inputs = {}, {}, {}, []
    for seed in seeds:
        prefix = Path(args.baseline_pattern.format(seed=seed))
        baseline_meta = json.loads(prefix.with_suffix('.json').read_text())
        with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as source:
            base = {key: source[key].copy() for key in ['rows', 'targets', 'seen', 'predictions']}
        aligned_inputs(base, extra, baseline_meta, extra_meta, len(counts) - 1)
        if baseline_meta['cache_signature'] != metadata['preparation_config']:
            raise ValueError('Candidate files and cache have different signatures.')
        if base['predictions'].shape[1] < 500 or extra['predictions'].shape[1] < 500:
            raise ValueError('The follow-up requires both original Top-500 pools.')
        baselines[seed] = base
        inputs.append(dict(seed=seed, baseline=str(prefix.resolve()),
                           json_sha256=file_digest(prefix.with_suffix('.json')),
                           npz_sha256=file_digest(prefix.with_suffix('.npz')),
                           checkpoint_sha256=baseline_meta['checkpoint_sha256']))
        for quota in quotas:
            found, replaced = protected_candidates(base['predictions'], extra['predictions'][:, :500], counts, quota)
            groups, per_user = coverage_summary(found, base['targets'], base['rows'], counts,
                                                 content_present=present[base['targets']])
            np.testing.assert_array_equal(found[:, :10], base['predictions'][:, :10])
            summaries[quota].append(groups)
            replacements[quota].append(dict(seed=seed, mean=float(replaced.mean()),
                                             max=int(replaced.max()), users_with_replacements=int((replaced > 0).sum())))
            candidates[seed, quota] = found
            details[seed, quota] = dict(per_user=per_user, replaced=replaced)
        print(json.dumps(dict(seed=seed, quota_evaluations_complete=len(quotas))), flush=True)
    rows = [dict(quota=quota, groups=mean_groups(summaries[quota]),
                  per_seed=[dict(seed=seed, groups=groups) for seed, groups in zip(seeds, summaries[quota])],
                  mean_replaced_candidates=float(np.mean([row['mean'] for row in replacements[quota]])),
                  replacement_statistics=replacements[quota]) for quota in quotas]
    selection = select_quota(rows, config['fusion']['minimum_overall_recall500_retention'])
    selection.update(followup_of=str(Path(args.original_fusion).resolve()),
                     validation_driven_followup=True, independently_confirmed=False)
    chosen = selection['selected_quota']
    frequency = counts[extra['targets']]
    masks = dict(all=np.ones(len(frequency), dtype=bool), cold_0=frequency == 0,
                 tail_1_5=(frequency >= 1) & (frequency <= 5),
                 mid_6_50=(frequency >= 6) & (frequency <= 50), head_gt50=frequency > 50)
    difference = np.stack([details[seed, chosen]['per_user']['hits_at_500'].astype(float)
                           - details[seed, 0]['per_user']['hits_at_500'] for seed in seeds])
    intervals = {name: dict(users=int(mask.sum()), difference=float(difference[:, mask].mean()),
                            **confidence_intervals(difference[:, mask])) for name, mask in masks.items() if mask.any()}
    for path, expected in original_sha.items():
        if file_digest(path) != expected:
            raise AssertionError('The original failed-fusion evidence changed.')
    if file_digest(plan_path) != plan_sha:
        raise AssertionError('The follow-up plan changed while evaluating.')
    result = dict(split='val', seeds=seeds, candidate_budget=500, quota_sweep=rows, selection=selection,
                  selected_paired_recall500_intervals=intervals, protected_top10_unchanged=True,
                  baseline_rare_candidates_preserved=True, protected_training_frequency_max=5,
                  top10_improvement_claimed=False, test_targets_read=False,
                  followup_plan=plan, followup_plan_sha256=plan_sha,
                  followup_of=str(Path(args.original_fusion).resolve()), original_fusion_sha256=original_sha,
                  validation_driven_followup=True, independently_confirmed=False,
                  input_artifacts=inputs, extra_artifact=str(extra_prefix.resolve()),
                  extra_json_sha256=file_digest(extra_prefix.with_suffix('.json')),
                  extra_npz_sha256=file_digest(extra_prefix.with_suffix('.npz')),
                  cache_signature=metadata['preparation_config'],
                  evaluation_cohort_sha256=extra_meta['evaluation_cohort_sha256'],
                  selection_caveat='Exploratory amendment after naive validation failure; selection and CIs reuse development users.')
    output.mkdir(parents=True, exist_ok=True)
    source_files = ['protected_fusion.py', 'candidate_fusion.py', 'coverage.py', 'objective_report.py']
    result['source_sha256'] = {}
    for name in source_files:
        source = Path(__file__).with_name(name)
        destination = output / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        result['source_sha256'][name] = file_digest(source)
    for seed in seeds:
        base = baselines[seed]
        np.savez_compressed(output / f'selected_seed{seed}.npz', predictions=candidates[seed, chosen],
                            rows=base['rows'], targets=base['targets'], seen=base['seen'],
                            replaced_candidates=details[seed, chosen]['replaced'],
                            **details[seed, chosen]['per_user'])
    selection['selected_at_utc'] = datetime.now(timezone.utc).isoformat()
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    write_report(result, report_path)
    print(json.dumps(dict(selection=selection, rows=rows, intervals=intervals), indent=2), flush=True)
    return result


def write_report(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = result['selection']['selected_quota']
    lines = ['# 探索性修正：保护已召回稀有广告的固定500融合', '',
             '**这是首次简单配额融合失败后，根据验证现象提出的探索性 follow-up。它不是原计划的一部分，也不是独立确认结果。** 原失败实验和选择均保留，未读取测试目标。', '',
             '## 修改了什么', '',
             '- 固定候选预算500，固定配额0/50/100/250；准入、主指标与平分规则沿用原实验。',
             '- 完整保护基线前十及所有训练频次≤5的既有候选；这些冷启动、长尾候选不会被移走。',
             '- 内容通道只贡献训练频次≤5、正ID、去重后且不在整个基线前500中的候选。',
             '- 从基线尾部删除排名在10之后且频次>5的候选，数量为“配额、新稀有候选数、可删除候选数”的最小值；其余基线保持相对顺序，然后追加新稀有候选。',
             '- 每人仍为500个非零唯一广告，前十逐项不变。本次只检验候选覆盖，未训练排序，也未改善Top-10。', '',
             f'选中的配额为 **{chosen}**。', '',
             '| 配额 | 总体 Recall@500 | 冷启动 | 长尾1–5 | 中频6–50 | 头部>50 | 平均替换数 | 通过约束 |',
             '|---:|---:|---:|---:|---:|---:|---:|---|']
    eligibility = {row['quota']: row['eligible'] for row in result['selection']['rows']}
    for row in result['quota_sweep']:
        values = ' | '.join(f"{row['groups'][group]['Recall@500']:.4%}" for group in GROUPS)
        lines.append(f"| {row['quota']} | {values} | {row['mean_replaced_candidates']:.2f} | {'是' if eligibility[row['quota']] else '否'} |")
    lines += ['', '## 所选配额与基线的配对差值', '',
              '差值为“所选固定500 − 原基线固定500”。区间对用户与三个训练种子配对重采样；同一验证数据参与了提出假设、选择参数和计算区间，**只能作为开发性描述**。', '',
              '| 分组 | Recall@500差值 | 用户与种子配对95%区间 |', '|---|---:|---|']
    for name in GROUPS:
        row = result['selected_paired_recall500_intervals'].get(name)
        if row:
            lo, hi = row['paired_seed_and_user_95pct']
            lines.append(f"| {name} | {row['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    if chosen == 0:
        lines += ['', '所选零配额与基线相同，差值和区间恒0是恒等比较，不代表统计确定的改进。']
    lines += ['', '即使冷启动、长尾的候选覆盖增加，也不能直接声称Top-10改善；之后若做排序，仍需独立构造训练样本并验证。本文没有把两路≤1000候选的无序并集当成固定500的结果。', '',
              '## 证据与复现', '',
              '- [首次融合失败记录](fusion.md)、[固定模型诊断](diagnostics.md)。',
              '- [修正前固定计划](../../artifacts/cold_start/fusion_protected_plan.json)。',
              '- [实现](../../taac_retrieval/protected_fusion.py)、[测试](../../tests/test_protected_fusion.py)。',
              '- [完整指标](../../artifacts/cold_start/fusion_protected/metrics.json)、[选择](../../artifacts/cold_start/fusion_protected/selection.json)。',
              '- `artifacts/cold_start/fusion_protected/selected_seed*.npz` 保存同一10,000人队列所选配额的500个候选、历史、目标和逐用户指标。', '',
              '```bash', '.project_venv/bin/python -m taac_retrieval.protected_fusion \\',
              '  --output artifacts/cold_start/fusion_protected_reproduction \\',
              '  --report reports/cold_start/fusion_protected_reproduction.md', '```', '',
              '已有输出会拒绝覆盖。该实验没有替换原模型或持久化索引，也没有修改首次失败的融合产物。', '']
    path.write_text('\n'.join(lines))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--plan', default='artifacts/cold_start/fusion_protected_plan.json')
    result.add_argument('--config', default='configs/cold_start.json')
    result.add_argument('--baseline-pattern', default='artifacts/cold_start/diagnostics/click_seed{seed}')
    result.add_argument('--extra-prefix', default='artifacts/cold_start/diagnostics/content_mean81')
    result.add_argument('--original-fusion', default='artifacts/cold_start/fusion')
    result.add_argument('--output', default='artifacts/cold_start/fusion_protected')
    result.add_argument('--report', default='reports/cold_start/fusion_protected.md')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
