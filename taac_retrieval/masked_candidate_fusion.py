"""Exploratory protected fusion with the matched ID-mask70 learned channel.

Reuses the already-tested rare-preserving merge and unchanged selection gate.
No training, reranking, test evaluation, or overwrite of previous experiments.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from .candidate_fusion import GROUPS, aligned_inputs, mean_groups, select_quota
from .coverage import coverage_summary, file_digest
from .protected_fusion import protected_candidates
from .objective_report import confidence_intervals


def validate_checkpoint_pair(seed, baseline_meta, extra_meta):
    """Match training settings and bind both saved pools to real checkpoints."""
    a, b = baseline_meta['training'], extra_meta['training']
    if a['seed'] != seed or b['seed'] != seed:
        raise ValueError('Baseline and augmented candidates must use the matching training seed.')
    if a.get('item_id_dropout', 0) != 0 or b.get('item_id_dropout') != .7:
        raise ValueError('Require the zero-mask baseline and ID-mask70 extra channel.')
    fields = ['variant', 'objective', 'num_negatives', 'temperature', 'epochs', 'batch_size',
              'hidden', 'blocks', 'heads', 'dropout', 'lr', 'max_steps', 'skip_test']
    for field in fields:
        if a[field] != b[field]:
            raise ValueError(f'The two candidate models differ in {field}.')
    if a['max_steps'] != 0 or not a['skip_test']:
        raise ValueError('Require complete training that did not evaluate test labels.')
    for metadata in [baseline_meta, extra_meta]:
        if file_digest(metadata['checkpoint']) != metadata['checkpoint_sha256']:
            raise ValueError('A candidate pool does not match its recorded checkpoint.')


def run(args):
    output = Path(args.output)
    if (output / 'metrics.json').exists() or (output / 'selection.json').exists():
        raise FileExistsError('Masked fusion results already exist; use a new output directory.')
    plan_path, config_path = Path(args.plan), Path(args.config)
    plan, config = json.loads(plan_path.read_text()), json.loads(config_path.read_text())
    if (plan.get('split') != 'val' or plan.get('test_targets_read') is not False
            or plan.get('extra_arm') != 'id_mask_70' or plan['quotas'] != [0, 10, 25, 50]
            or plan['candidate_budget'] != 500 or plan['protected_training_frequency_max'] != 5):
        raise ValueError('Require the separately registered validation-only mask70 fusion amendment.')
    plan_sha = file_digest(plan_path)
    preserved = {}
    for name in ['fusion', 'fusion_protected']:
        for filename in ['metrics.json', 'selection.json']:
            path = output.parent / name / filename
            if path.exists():
                preserved[str(path)] = file_digest(path)
    cache, seeds, quotas = Path(config['cache']), config['seeds'], plan['quotas']
    metadata = json.loads((cache / 'metadata.json').read_text())
    counts = np.load(cache / 'train_counts.npy', mmap_mode='r')
    present = np.load(cache / 'mm_present.npy', mmap_mode='r')
    summaries, replacements = {q: [] for q in quotas}, {q: [] for q in quotas}
    candidates, details, baselines, inputs = {}, {}, {}, []
    for seed in seeds:
        prefixes = dict(baseline=Path(args.baseline_pattern.format(seed=seed)),
                        extra=Path(args.extra_pattern.format(seed=seed)))
        files, metas, identity = {}, {}, {'seed': seed}
        for arm, prefix in prefixes.items():
            metas[arm] = json.loads(prefix.with_suffix('.json').read_text())
            with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as source:
                files[arm] = {key: source[key].copy() for key in ['rows', 'targets', 'seen', 'predictions']}
            identity[arm] = dict(prefix=str(prefix.resolve()), json_sha256=file_digest(prefix.with_suffix('.json')),
                                 npz_sha256=file_digest(prefix.with_suffix('.npz')),
                                 checkpoint_sha256=metas[arm]['checkpoint_sha256'])
        base, extra = files['baseline'], files['extra']
        aligned_inputs(base, extra, metas['baseline'], metas['extra'], len(counts) - 1)
        validate_checkpoint_pair(seed, metas['baseline'], metas['extra'])
        if metas['baseline']['cache_signature'] != metadata['preparation_config']:
            raise ValueError('Candidate cache signatures differ from the selected feature cache.')
        if base['predictions'].shape[1] < 500 or extra['predictions'].shape[1] < 500:
            raise ValueError('Both candidate inputs must expose the original Top-500 budget.')
        baselines[seed] = base
        inputs.append(identity)
        for quota in quotas:
            found, replaced = protected_candidates(base['predictions'], extra['predictions'][:, :500], counts, quota)
            groups, per_user = coverage_summary(found, base['targets'], base['rows'], counts,
                                                 content_present=present[base['targets']])
            np.testing.assert_array_equal(found[:, :10], base['predictions'][:, :10])
            summaries[quota].append(groups)
            replacements[quota].append(dict(seed=seed, mean=float(replaced.mean()), max=int(replaced.max()),
                                             users_with_replacements=int((replaced > 0).sum())))
            candidates[seed, quota] = found
            details[seed, quota] = dict(per_user=per_user, replaced=replaced)
        print(json.dumps(dict(seed=seed, quota_evaluations_complete=len(quotas))), flush=True)
    rows = [dict(quota=q, groups=mean_groups(summaries[q]),
                  per_seed=[dict(seed=seed, groups=g) for seed, g in zip(seeds, summaries[q])],
                  mean_replaced_candidates=float(np.mean([r['mean'] for r in replacements[q]])),
                  replacement_statistics=replacements[q]) for q in quotas]
    selection = select_quota(rows, config['fusion']['minimum_overall_recall500_retention'])
    selection.update(extra_arm='id_mask_70', validation_driven_followup=True, independently_confirmed=False)
    chosen = selection['selected_quota']
    reference = baselines[seeds[0]]
    frequency = counts[reference['targets']]
    masks = dict(all=np.ones(len(frequency), dtype=bool), cold_0=frequency == 0,
                 tail_1_5=(frequency >= 1) & (frequency <= 5),
                 mid_6_50=(frequency >= 6) & (frequency <= 50), head_gt50=frequency > 50)
    difference = np.stack([details[seed, chosen]['per_user']['hits_at_500'].astype(float)
                           - details[seed, 0]['per_user']['hits_at_500'] for seed in seeds])
    intervals = {name: dict(users=int(mask.sum()), difference=float(difference[:, mask].mean()),
                            **confidence_intervals(difference[:, mask])) for name, mask in masks.items() if mask.any()}
    for path, expected in preserved.items():
        if file_digest(path) != expected:
            raise AssertionError('An earlier fusion result changed while evaluating the amendment.')
    if file_digest(plan_path) != plan_sha:
        raise AssertionError('The registered amendment changed during evaluation.')
    result = dict(split='val', seeds=seeds, candidate_budget=500, quota_sweep=rows, selection=selection,
                  selected_paired_recall500_intervals=intervals, protected_top10_unchanged=True,
                  baseline_rare_candidates_preserved=True, protected_training_frequency_max=5,
                  extra_arm='id_mask_70', followup_plan=plan, followup_plan_sha256=plan_sha,
                  validation_driven_followup=True, independently_confirmed=False,
                  top10_improvement_claimed=False, test_targets_read=False,
                  input_artifacts=inputs, cache_signature=metadata['preparation_config'],
                  evaluation_cohort_sha256=metas['baseline']['evaluation_cohort_sha256'],
                  preserved_previous_fusion_sha256=preserved,
                  selection_caveat='Quotas/hypothesis and selection reuse validation after observing earlier tradeoffs; no independent confirmation.')
    output.mkdir(parents=True, exist_ok=True)
    result['source_sha256'] = {}
    for filename in ['masked_candidate_fusion.py', 'protected_fusion.py', 'candidate_fusion.py', 'coverage.py', 'objective_report.py']:
        source = Path(__file__).with_name(filename)
        snapshot = output / 'source_snapshot' / filename
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
        result['source_sha256'][filename] = file_digest(source)
    for seed in seeds:
        base = baselines[seed]
        np.savez_compressed(output / f'selected_seed{seed}.npz', predictions=candidates[seed, chosen],
                            rows=base['rows'], targets=base['targets'], seen=base['seen'],
                            replaced_candidates=details[seed, chosen]['replaced'], **details[seed, chosen]['per_user'])
    selection['selected_at_utc'] = datetime.now(timezone.utc).isoformat()
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    write_report(result, Path(args.report))
    print(json.dumps(dict(selection=selection, rows=rows, intervals=intervals), indent=2), flush=True)
    return result


def write_report(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = result['selection']['selected_quota']
    lines = ['# 探索性修正：基线与 ID-mask70 的保护式候选融合', '',
             '**这是观察到文本融合失败、ID-mask70 单模型存在冷/尾取舍之后提出的验证驱动 amendment。配额、选择和区间重复使用验证用户，不属于独立确认。** 没有读取测试目标。', '',
             '## 固定规则与身份检查', '',
             '- 两路均为相同训练种子的 learned Top-500：原零遮蔽模型与 ID-mask70 模型，严格核对用户行号、目标、全部历史、缓存/队列签名和 checkpoint SHA。',
             '- 每人仍为500候选；保护原Top-10及所有频次≤5的原有候选，只追加 mask70 通道中频次≤5、且不在整个原Top-500中的新广告。',
             '- 只从基线尾部的非稀有候选移除等量广告，保留其他基线顺序。两路内积分数属于不同训练模型，不直接相加。',
             '- 配额固定为0/10/25/50：总体Recall@500至少保留95%，冷启动和长尾各自不退步，再最大化两组等权平均，平分选更小配额。',
             '- 原Top-10逐项不变；本次仅分析候选池，不声称Top-10改善，也未训练排序。第二模型/索引的在线代价尚未测试。', '',
             f'所选配额：**{chosen}**。', '',
             '| 配额 | 总体 Recall@500 | 冷启动 | 长尾1–5 | 中频6–50 | 头部>50 | 平均替换数 | 通过约束 |',
             '|---:|---:|---:|---:|---:|---:|---:|---|']
    decisions = {r['quota']: r['eligible'] for r in result['selection']['rows']}
    for row in result['quota_sweep']:
        values = ' | '.join(f"{row['groups'][g]['Recall@500']:.4%}" for g in GROUPS)
        lines.append(f"| {row['quota']} | {values} | {row['mean_replaced_candidates']:.2f} | {'是' if decisions[row['quota']] else '否'} |")
    lines += ['', '## 所选固定500与基线的配对差值', '',
              '区间按用户与三个训练种子配对重采样，仅为重复使用验证集后的开发性描述。', '',
              '| 分组 | Recall@500差值 | 用户与种子配对95%区间 |', '|---|---:|---|']
    for group in GROUPS:
        row = result['selected_paired_recall500_intervals'].get(group)
        if row:
            lo, hi = row['paired_seed_and_user_95pct']
            lines.append(f"| {group} | {row['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    if chosen == 0:
        lines += ['', '零配额就是原基线，差值与区间恒为0属于恒等比较，不代表确定的优化收益。本轮不再扩展配额搜索。']
    lines += ['', '## 来源', '',
              '- [原固定模型诊断](diagnostics.md)、[首次融合](fusion.md)、[保护式文本融合](fusion_protected.md)。',
              '- [先登记的 amendment](../../artifacts/cold_start/masked_fusion_plan.json)。',
              '- [实现](../../taac_retrieval/masked_candidate_fusion.py)、[复用的保护合并](../../taac_retrieval/protected_fusion.py)。',
              '- [完整指标](../../artifacts/cold_start/fusion_masked/metrics.json)、[选择](../../artifacts/cold_start/fusion_masked/selection.json)。',
              '- 所选候选保存在 `artifacts/cold_start/fusion_masked/selected_seed*.npz`；原有模型、索引和前两次融合结果均保留。', '',
              '```bash', '.project_venv/bin/python -m taac_retrieval.masked_candidate_fusion \\',
              '  --output artifacts/cold_start/fusion_masked_reproduction \\',
              '  --report reports/cold_start/fusion_masked_reproduction.md', '```', '']
    path.write_text('\n'.join(lines))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', default='configs/cold_start.json')
    result.add_argument('--plan', default='artifacts/cold_start/masked_fusion_plan.json')
    result.add_argument('--baseline-pattern', default='artifacts/cold_start/diagnostics/click_seed{seed}')
    result.add_argument('--extra-pattern', default='artifacts/cold_start/diagnostics/id_mask_70_seed{seed}')
    result.add_argument('--output', default='artifacts/cold_start/fusion_masked')
    result.add_argument('--report', default='reports/cold_start/fusion_masked.md')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
