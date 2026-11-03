"""Validation-only fixed-budget quota fusion of learned and content candidates.

The baseline Top-10 order is protected. This module evaluates candidate coverage,
not a learned ranking model or a Top-10 recommendation improvement.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .coverage import coverage_summary, file_digest
from .week_report import confidence_intervals


GROUPS = ['all', 'cold_0', 'tail_1_5', 'mid_6_50', 'head_gt50']


def quota_candidates(baseline, extra, quota, budget=500, protected=10):
    """Keep a baseline prefix, add up to quota new IDs, refill from baseline."""
    baseline, extra = np.asarray(baseline), np.asarray(extra)
    if (baseline.ndim != 2 or extra.ndim != 2 or baseline.shape[0] != extra.shape[0]
            or not len(baseline) or baseline.shape[1] < budget
            or not np.issubdtype(baseline.dtype, np.integer)
            or not np.issubdtype(extra.dtype, np.integer)):
        raise ValueError('Expected aligned integer candidate matrices and a complete baseline.')
    if budget < protected or protected < 1 or quota < 0 or quota > budget - protected:
        raise ValueError('Quota must leave room for the protected baseline prefix.')
    result = np.empty((len(baseline), budget), dtype=np.int32)
    added = np.zeros(len(baseline), dtype=np.int32)
    retained = budget - quota
    for row, (base_row, extra_row) in enumerate(zip(baseline, extra)):
        base_row = base_row[:budget]
        if (base_row <= 0).any() or len(set(map(int, base_row))) != budget:
            raise ValueError('Baseline must contain budget unique nonpadding IDs.')
        chosen = list(map(int, base_row[:retained]))
        present = set(chosen)
        if quota:
            for item in extra_row:
                item = int(item)
                if item <= 0 or item in present:
                    continue
                chosen.append(item)
                present.add(item)
                added[row] += 1
                if added[row] == quota:
                    break
        for item in base_row[retained:]:
            if len(chosen) == budget:
                break
            item = int(item)
            if item not in present:
                chosen.append(item)
                present.add(item)
        if len(chosen) != budget:
            raise AssertionError('The complete baseline must permit filling the fixed budget.')
        result[row] = chosen
    np.testing.assert_array_equal(result[:, :protected], baseline[:, :protected])
    return result, added


def select_quota(rows, minimum_overall_retention=.95):
    """Cold/tail macro coverage subject to full-pool and per-group safeguards."""
    controls = [row for row in rows if row['quota'] == 0]
    if len(controls) != 1 or len({row['quota'] for row in rows}) != len(rows):
        raise ValueError('Require exactly one zero-quota baseline and unique quotas.')
    if not 0 < minimum_overall_retention <= 1:
        raise ValueError('Overall retention must lie in (0, 1].')
    baseline = controls[0]['groups']
    candidates = []
    for row in rows:
        g = row['groups']
        values = [g[name]['Recall@500'] for name in ['all', 'cold_0', 'tail_1_5']]
        if any(value is None or not np.isfinite(value) for value in values):
            raise ValueError('Selection requires nonempty overall, cold and tail groups.')
        reasons = []
        if g['all']['Recall@500'] + 1e-12 < baseline['all']['Recall@500'] * minimum_overall_retention:
            reasons.append('overall_recall500_below_retention_gate')
        for group in ['cold_0', 'tail_1_5']:
            if g[group]['Recall@500'] + 1e-12 < baseline[group]['Recall@500']:
                reasons.append(f'{group}_recall500_below_baseline')
        candidates.append(dict(quota=row['quota'], eligible=not reasons, rejection_reasons=reasons,
                               cold_tail_macro_recall500=(g['cold_0']['Recall@500'] + g['tail_1_5']['Recall@500']) / 2))
    chosen = min((row for row in candidates if row['eligible']),
                 key=lambda row: (-row['cold_tail_macro_recall500'], row['quota']))
    return dict(selected_quota=chosen['quota'], baseline_retained=chosen['quota'] == 0,
                minimum_overall_recall500_retention=minimum_overall_retention,
                criterion='maximum equally weighted mean cold/tail Recall@500, then smaller quota',
                rows=candidates, selection_split='val', test_targets_read=False,
                inference='development selection on reused validation users; not independent confirmation')


def aligned_inputs(baseline, extra, baseline_meta, extra_meta, num_items):
    for metadata in [baseline_meta, extra_meta]:
        if metadata.get('split') != 'val' or metadata.get('test_targets_read') is not False:
            raise ValueError('Only explicitly validation-only candidate artifacts are accepted.')
    for key in ['evaluation_cohort_sha256', 'cache_signature']:
        if baseline_meta.get(key) != extra_meta.get(key):
            raise ValueError(f'Candidate artifacts differ in {key}.')
    for field in ['rows', 'targets', 'seen']:
        if not np.array_equal(baseline[field], extra[field]):
            raise ValueError(f'Candidate inputs differ in {field}.')
    if baseline['predictions'].shape[0] != len(baseline['rows']) or extra['predictions'].shape[0] != len(baseline['rows']):
        raise ValueError('Candidate row count does not match query identity.')
    for source in [baseline, extra]:
        if (source['predictions'] > num_items).any():
            raise ValueError('Candidate exceeds the local catalog.')
        for candidates, seen in zip(source['predictions'], source['seen']):
            if np.isin(candidates[candidates > 0], seen[seen > 0]).any():
                raise ValueError('Unfiltered historical item appears in candidate inputs.')
    if (baseline['targets'] <= 0).any() or (baseline['targets'] > num_items).any():
        raise ValueError('Target is outside the local nonpadding catalog.')
    if (baseline['seen'] == baseline['targets'][:, None]).any():
        raise ValueError('Held-out target appears in visible history.')


def candidate_union(baseline, extra, targets, counts, k=500):
    """Unranked union of two Top-K pools; never label this Recall@K."""
    baseline, extra, targets = map(np.asarray, (baseline, extra, targets))
    base_hit = (baseline[:, :k] == targets[:, None]).any(1)
    extra_hit = ((extra[:, :k] > 0) & (extra[:, :k] == targets[:, None])).any(1)
    total = np.array([len(set(map(int, a[:k])) | {int(i) for i in b[:k] if i > 0})
                      for a, b in zip(baseline, extra)], dtype=np.int32)
    frequency = counts[targets]
    masks = dict(all=np.ones(len(targets), dtype=bool), cold_0=frequency == 0,
                 tail_1_5=(frequency >= 1) & (frequency <= 5),
                 mid_6_50=(frequency >= 6) & (frequency <= 50), head_gt50=frequency > 50)
    result = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        result[name] = dict(users=n, union_candidate_coverage=float((base_hit | extra_hit)[mask].mean()) if n else None,
                            union_hits=int((base_hit | extra_hit)[mask].sum()),
                            additional_hits=int((~base_hit & extra_hit)[mask].sum()),
                            mean_union_candidates=float(total[mask].mean()) if n else None)
    return dict(per_channel_budget=k, maximum_union_budget=2*k, unranked=True,
                is_fixed_500_comparison=False, groups=result)


def mean_groups(records):
    result = {}
    for group in records[0]:
        if any(row[group]['users'] != records[0][group]['users'] for row in records):
            raise AssertionError('Frequency group membership differs across training seeds.')
        result[group] = {'users': records[0][group]['users']}
        for metric in ['Recall@10', 'Recall@100', 'Recall@500', 'NDCG@10', 'NDCG@100', 'NDCG@500']:
            values = [row[group][metric] for row in records]
            result[group][metric] = float(np.mean(values)) if values[0] is not None else None
    return result


def run(args):
    output, report_path = Path(args.output), Path(args.report)
    if (output / 'selection.json').exists() or (output / 'metrics.json').exists():
        raise FileExistsError('Fusion results already exist; use a new output directory.')
    config = json.loads(Path(args.config).read_text())
    spec, seeds = config['fusion'], config['seeds']
    if spec['candidate_budget'] != 500 or not spec['preserve_baseline_top10'] or not spec['require_cold_and_tail_each_not_worse']:
        raise ValueError('This experiment requires the declared fixed-500 and protected-Top10 policy.')
    quotas = spec['extra_channel_quotas']
    if 0 not in quotas:
        raise ValueError('The zero-quota control is mandatory.')
    cache = Path(config['cache'])
    counts = np.load(cache / 'train_counts.npy', mmap_mode='r')
    present = np.load(cache / 'mm_present.npy', mmap_mode='r')
    metadata = json.loads((cache / 'metadata.json').read_text())
    extra_prefix = Path(args.extra_prefix)
    extra_meta = json.loads(extra_prefix.with_suffix('.json').read_text())
    with np.load(extra_prefix.with_suffix('.npz'), allow_pickle=False) as source:
        extra = {key: source[key].copy() for key in ['rows', 'targets', 'seen', 'predictions']}
    summaries = {quota: [] for quota in quotas}
    candidates, details, baselines, unions, inputs = {}, {}, {}, [], []
    for seed in seeds:
        prefix = Path(args.baseline_pattern.format(seed=seed))
        meta = json.loads(prefix.with_suffix('.json').read_text())
        with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as source:
            base = {key: source[key].copy() for key in ['rows', 'targets', 'seen', 'predictions']}
        aligned_inputs(base, extra, meta, extra_meta, len(counts) - 1)
        if meta['cache_signature'] != metadata['preparation_config']:
            raise ValueError('Cache signature differs from candidate artifacts.')
        if base['predictions'].shape[1] < 500 or extra['predictions'].shape[1] < 500:
            raise ValueError('Both diagnostic inputs must expose Top-500 candidate slots.')
        baselines[seed] = base
        inputs.append(dict(seed=seed, baseline=str(prefix.resolve()),
                           json_sha256=file_digest(prefix.with_suffix('.json')),
                           npz_sha256=file_digest(prefix.with_suffix('.npz')),
                           checkpoint_sha256=meta['checkpoint_sha256']))
        for quota in quotas:
            found, added = quota_candidates(base['predictions'], extra['predictions'][:, :500], quota)
            groups, per_user = coverage_summary(found, base['targets'], base['rows'], counts,
                                                 content_present=present[base['targets']])
            if not np.array_equal(found[:, :10], base['predictions'][:, :10]):
                raise AssertionError('Fusion changed the protected Top-10.')
            summaries[quota].append(groups)
            candidates[seed, quota] = found
            details[seed, quota] = dict(per_user=per_user, extra_new_candidates=added)
        unions.append(dict(seed=seed, **candidate_union(base['predictions'], extra['predictions'], base['targets'], counts)))
    rows = [dict(quota=quota, groups=mean_groups(summaries[quota]), per_seed=[dict(seed=seed, groups=groups)
                  for seed, groups in zip(seeds, summaries[quota])]) for quota in quotas]
    selection = select_quota(rows, spec['minimum_overall_recall500_retention'])
    chosen = selection['selected_quota']
    mask_frequency = counts[extra['targets']]
    masks = dict(all=np.ones(len(mask_frequency), dtype=bool), cold_0=mask_frequency == 0,
                 tail_1_5=(mask_frequency >= 1) & (mask_frequency <= 5),
                 mid_6_50=(mask_frequency >= 6) & (mask_frequency <= 50), head_gt50=mask_frequency > 50)
    differences = np.stack([details[seed, chosen]['per_user']['hits_at_500'].astype(float)
                            - details[seed, 0]['per_user']['hits_at_500'] for seed in seeds])
    intervals = {name: dict(users=int(mask.sum()), difference=float(differences[:, mask].mean()),
                            **confidence_intervals(differences[:, mask])) for name, mask in masks.items() if mask.any()}
    union_mean = {}
    for group in GROUPS:
        values = [u['groups'][group]['union_candidate_coverage'] for u in unions]
        union_mean[group] = dict(users=unions[0]['groups'][group]['users'],
                                 union_candidate_coverage=float(np.mean(values)) if values[0] is not None else None)
    result = dict(split='val', seeds=seeds, candidate_budget=500, quota_sweep=rows,
                  selection=selection, selected_paired_recall500_intervals=intervals,
                  unranked_union=dict(maximum_union_budget=1000, is_fixed_500_comparison=False,
                                      mean_groups=union_mean, per_seed=unions),
                  protected_top10_unchanged=True, top10_improvement_claimed=False,
                  input_artifacts=inputs, extra_artifact=str(extra_prefix.resolve()),
                  extra_json_sha256=file_digest(extra_prefix.with_suffix('.json')),
                  extra_npz_sha256=file_digest(extra_prefix.with_suffix('.npz')),
                  cache_signature=metadata['preparation_config'],
                  evaluation_cohort_sha256=extra_meta['evaluation_cohort_sha256'],
                  registered_configuration=spec, test_targets_read=False,
                  selection_caveat='Chosen on the same reused validation users; paired intervals are descriptive, not confirmatory.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        base = baselines[seed]
        np.savez_compressed(output / f'selected_seed{seed}.npz', predictions=candidates[seed, chosen],
                            rows=base['rows'], targets=base['targets'], seen=base['seen'],
                            extra_new_candidates=details[seed, chosen]['extra_new_candidates'],
                            **details[seed, chosen]['per_user'])
    selection['selected_at_utc'] = datetime.now(timezone.utc).isoformat()
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    write_report(result, report_path)
    print(json.dumps(dict(selected_quota=chosen, rows=rows, intervals=intervals), indent=2), flush=True)
    return result


def write_report(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = result['selection']['selected_quota']
    lines = ['# 固定 500 个候选的内容配额融合', '',
             f'验证集选中的内容配额为 **{chosen}**。这是候选覆盖对照，基线前十的广告及顺序全部保留，**没有训练排序模型，也没有改善 Top-10 的声明**。', '',
             '## 固定规则', '',
             '- 使用点击 InfoNCE-64 三个种子的精确 Top-500，以及同一用户的 raw81 文本均值余弦 Top-500。',
             '- 总预算恒为 500：先保留基线前 `500−quota` 个，再从内容通道按原顺序补最多 quota 个新广告；不足时用基线剩余候选补齐。',
             '- 跳过内容通道的非正 ID 与重复广告；保留基线 Top-10，过滤全部已见历史。两路分数来自不同空间，不直接相加。',
             '- 固定 quota 为 0、50、100、250，零配额是完整对照。仅验证集选择，不读取测试目标。',
             '- 准入要求：总体 Recall@500 不低于基线的 95%，冷启动与长尾 Recall@500 各自不低于基线；准入后最大化两组等权平均，平分选较小 quota。', '',
             '## 三种子平均 Recall@500', '',
             '| 内容配额 | 总体 | 冷启动 | 长尾 1–5 | 中频 6–50 | 头部 >50 | 通过约束 |', '|---:|---:|---:|---:|---:|---:|---|']
    decisions = {row['quota']: row for row in result['selection']['rows']}
    for row in result['quota_sweep']:
        values = ' | '.join(f"{row['groups'][group]['Recall@500']:.4%}" for group in GROUPS)
        lines.append(f"| {row['quota']} | {values} | {'是' if decisions[row['quota']]['eligible'] else '否'} |")
    lines += ['', '## 所选配置的配对差值', '',
              '以下是“所选固定500 − 基线固定500”的 Recall@500 差值，按用户与训练种子配对重采样。**参数就在本验证队列上选择，区间属于开发性描述，不是独立确认实验。**', '',
              '| 分组 | 平均差值 | 用户与种子配对 95% 区间 |', '|---|---:|---|']
    for name in GROUPS:
        row = result['selected_paired_recall500_intervals'].get(name)
        if row:
            lo, hi = row['paired_seed_and_user_95pct']
            lines.append(f"| {name} | {row['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    if chosen == 0:
        lines += ['', '本次所选零配额与基线是完全相同的候选，因此差值和区间恒为0；这只是恒等比较，不是统计确定的优化收益。']
    lines += ['', '本实验只评估候选池，所有配额都保护基线 Top-10，不会直接改变最终 Top-10 推荐。所选零配额还保持整个候选池不变。若后续增加排序，需另建仅使用训练前缀的候选训练样本，并独立验证效果。' if chosen == 0 else
              '本实验只评估候选池，所有配额都保护基线 Top-10，不会直接改变最终 Top-10 推荐。若后续增加排序，需另建仅使用训练前缀的候选训练样本，并独立验证效果。', '',
              '## 两路无序并集：不同预算的覆盖上界', '',
              '下面将基线前500与内容前500取集合并集，实际每人最多1000个候选。它是未排序集合的覆盖诊断，**不是固定500的结果，不能称作 Recall@500**。', '',
              '| 分组 | ≤1000 无序并集覆盖率 |', '|---|---:|']
    for name in GROUPS:
        row = result['unranked_union']['mean_groups'][name]
        lines.append(f"| {name} | {row['union_candidate_coverage']:.4%} |")
    lines += ['', '## 来源与复现', '',
              '- [固定模型诊断](diagnostics.md)、[完整配置](../../configs/cold_start.json)。',
              '- [融合实现](../../taac_retrieval/candidate_fusion.py)、[测试](../../tests/test_candidate_fusion.py)。',
              '- [指标与输入摘要](../../artifacts/cold_start/fusion/metrics.json)、[固定选择](../../artifacts/cold_start/fusion/selection.json)。',
              '- `artifacts/cold_start/fusion/selected_seed*.npz` 保存所选配额的候选、原始用户下标、目标、历史和逐用户指标。', '',
              '```bash', '.project_venv/bin/python -m taac_retrieval.candidate_fusion \\',
              '  --config configs/cold_start.json \\',
              '  --output artifacts/cold_start/fusion_reproduction \\',
              '  --report reports/cold_start/fusion_reproduction.md', '```', '',
              '本实验不替换原模型与索引，也不把开发集上的配额选择作为线上或独立测试收益。', '']
    path.write_text('\n'.join(lines))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', default='configs/cold_start.json')
    result.add_argument('--baseline-pattern', default='artifacts/cold_start/diagnostics/click_seed{seed}')
    result.add_argument('--extra-prefix', default='artifacts/cold_start/diagnostics/content_mean81')
    result.add_argument('--output', default='artifacts/cold_start/fusion')
    result.add_argument('--report', default='reports/cold_start/fusion.md')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
