"""Validation-only evidence and guarded selection for item-ID masking studies."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .coverage import coverage_summary
from .data import PreparedData
from .week_report import confidence_intervals


GROUPS = ['all', 'cold_0', 'tail_1_5', 'mid_6_50', 'head_gt50']
KS = [10, 100, 500]
EPSILON = 1e-12


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def metric_mean(row, group, metric):
    value = row['groups'][group][metric]
    return float(value['mean'] if isinstance(value, dict) else value)


def select_configuration(rows, settings):
    """Enforce overall quality and each rare bucket before maximizing the macro."""
    if len({row['name'] for row in rows}) != len(rows):
        raise ValueError('Duplicate arm names.')
    controls = [row for row in rows if row['probability'] == 0]
    if len(controls) != 1:
        raise ValueError('Require exactly one zero-mask control.')
    control = controls[0]
    baseline = [metric_mean(control, group, 'Recall@500') for group in ['cold_0', 'tail_1_5']]
    baseline_macro = float(np.mean(baseline))
    baseline_ndcg = metric_mean(control, 'all', 'NDCG@10')
    evaluated = []
    for row in rows:
        cold, tail = [metric_mean(row, group, 'Recall@500') for group in ['cold_0', 'tail_1_5']]
        ndcg = metric_mean(row, 'all', 'NDCG@10')
        if not np.isfinite([cold, tail, ndcg]).all() or min(cold, tail, ndcg) < 0:
            raise ValueError('Selection metrics must be finite and nonnegative.')
        failures = []
        if ndcg + EPSILON < baseline_ndcg * settings['minimum_overall_ndcg_retention']:
            failures.append('overall_ndcg_below_retention_gate')
        if settings['require_cold_and_tail_each_not_worse']:
            if cold + EPSILON < baseline[0]:
                failures.append('cold_recall500_decreased')
            if tail + EPSILON < baseline[1]:
                failures.append('tail_recall500_decreased')
        macro = float(np.mean([cold, tail]))
        improved = macro > baseline_macro + EPSILON
        evaluated.append(dict(name=row['name'], probability=row['probability'],
                               overall_ndcg10=ndcg, cold_recall500=cold, tail_recall500=tail,
                               cold_tail_macro_recall500=macro, rejection_reasons=failures,
                               eligible=not failures, strictly_improves_control=improved))
    candidates = [row for row in evaluated if row['probability'] > 0
                  and row['eligible'] and row['strictly_improves_control']]
    selected = (min(candidates, key=lambda row: (-row['cold_tail_macro_recall500'], row['probability']))
                if candidates else next(row for row in evaluated if row['probability'] == 0))
    return dict(selected_arm=selected['name'], selected_probability=selected['probability'],
                baseline_retained=selected['probability'] == 0, rows=evaluated,
                selection_split='val', test_targets_read=False, deployed=False,
                criteria=settings, baseline_macro_recall500=baseline_macro,
                selected_macro_recall500=selected['cold_tail_macro_recall500'],
                reason=('no eligible strict cold/tail macro improvement; retain control' if not candidates
                        else 'largest eligible cold/tail macro Recall@500; smaller masking probability on ties'))


def aggregate_groups(diagnostics):
    result = {}
    for group in GROUPS:
        counts = [item['groups'][group]['users'] for item in diagnostics]
        if len(set(counts)) != 1 or not counts[0]:
            raise ValueError(f'Need the same nonempty {group} evaluation cohort for every seed.')
        row = result[group] = dict(users=counts[0])
        for k in KS:
            for metric in [f'Recall@{k}', f'NDCG@{k}']:
                values = [item['groups'][group][metric] for item in diagnostics]
                row[metric] = dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)), values=values)
        values = [item['groups'][group]['conditional_Recall@10_given_Top500'] for item in diagnostics]
        finite = [value for value in values if value is not None]
        row['conditional_Recall@10_given_Top500'] = dict(mean=float(np.mean(finite)) if finite else None, values=values)
    return result


def load_diagnostic(prefix, data, expected_rows, checkpoint, config, history):
    description = json.loads(prefix.with_suffix('.json').read_text())
    if (description['split'] != 'val' or description['test_targets_read']
            or description['ks'] != KS or not description['filter_seen_items']):
        raise AssertionError('Require validation-only, history-filtered Top-10/100/500 diagnostics.')
    if (description['candidate_items'] != data.num_items or Path(description['cache']).resolve() != data.root.resolve()
            or Path(description['checkpoint']).resolve() != checkpoint.resolve()
            or description['checkpoint_sha256'] != digest(checkpoint)
            or description['cache_signature'] != data.metadata['preparation_config']):
        raise AssertionError('Diagnostic cache, catalog or checkpoint identity differs.')
    if description['training'].get('item_id_dropout', 0.) != config.get('item_id_dropout', 0.):
        raise AssertionError('Diagnostic model uses a different masking policy.')
    best = max(history, key=lambda row: (row['validation']['NDCG@10'], row['validation']['HR@10'], -row['epoch']))
    if description['selected_epoch'] != best['epoch']:
        raise AssertionError('Checkpoint was not chosen by the locked validation criterion.')
    with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as source:
        saved = {name: source[name].copy() for name in source.files}
    np.testing.assert_array_equal(saved['rows'], expected_rows)
    targets = data.items[expected_rows, data.val_pos[expected_rows]]
    np.testing.assert_array_equal(saved['targets'], targets)
    if (saved['seen'] == saved['targets'][:, None]).any():
        raise AssertionError('Validation target leaked into visible context.')
    computed, per_user = coverage_summary(saved['predictions'], saved['targets'], saved['rows'], data.train_counts,
                                           KS, saved['target_content_present'])
    for group in GROUPS:
        for k in KS:
            for metric in [f'Recall@{k}', f'NDCG@{k}']:
                np.testing.assert_allclose(computed[group][metric], description['groups'][group][metric], atol=1e-12)
    for name, expected in per_user.items():
        np.testing.assert_array_equal(saved[name], expected)
    cohort_sha = hashlib.sha256(np.asarray(expected_rows, dtype=np.int64).tobytes()).hexdigest()
    if description['evaluation_cohort_sha256'] != cohort_sha or config['evaluation_cohort_sha256'] != cohort_sha:
        raise AssertionError('Frozen cohort checksum differs.')
    return description, saved


def load_evidence(root, plan):
    specification = plan['specification']
    repo = Path(__file__).resolve().parents[1]
    cache = (repo / specification['cache']).resolve()
    data = PreparedData(cache)
    expected_rows = np.load(repo / specification['evaluation_indices'], allow_pickle=False)
    arms = dict(click=dict(name='click', probability=0., runs=[], diagnostics=[], predictions=[]))
    for probability in specification['item_id_dropout_probabilities']:
        name = f'id_mask_{round(probability * 100):02d}'
        arms[name] = dict(name=name, probability=probability, runs=[], diagnostics=[], predictions=[])
    jobs = {Path(job['folder']).name: job for job in plan['jobs']}
    for reference in plan['references']:
        seed = reference['seed']
        control_folder = Path(reference['folder'])
        reference_config = json.loads((control_folder / 'config.json').read_text())
        for name, expected in reference['files'].items():
            if digest(control_folder / name) != expected:
                raise AssertionError('Reference artifact changed after the plan was frozen.')
        for arm, records in arms.items():
            folder = control_folder if arm == 'click' else Path(jobs[f'{arm}_seed{seed}']['folder'])
            config = json.loads((folder / 'config.json').read_text())
            trained = json.loads((folder / 'result.json').read_text())
            history = json.loads((folder / 'history.json').read_text())
            if (len(history) != specification['epochs'] or config['max_steps'] or not config['skip_test']
                    or trained['test'] is not None or config.get('item_id_dropout', 0.) != records['probability']):
                raise AssertionError('Require complete matched training without test evaluation.')
            for field in ['parameters', 'shared_initialization_sha256', 'evaluation_cohort_sha256',
                          'variant', 'seed', 'epochs', 'batch_size', 'eval_batch_size', 'candidate_chunk',
                          'workers', 'hidden', 'blocks', 'heads', 'dropout', 'lr', 'objective',
                          'num_negatives', 'temperature', 'exposure_weight', 'torch_version', 'numpy_version']:
                if config[field] != reference_config[field]:
                    raise AssertionError(f'{folder.name}: unmatched {field}.')
            if Path(config['cache']).resolve() != cache:
                raise AssertionError('Training uses a different cache.')
            prefix = root / 'diagnostics' / f'{arm}_seed{seed}'
            description, predictions = load_diagnostic(prefix, data, expected_rows, folder / 'best.pt', config, history)
            if trained['best_epoch'] != description['selected_epoch']:
                raise AssertionError('Training summary and selected diagnostic epoch differ.')
            records['diagnostics'].append(description)
            records['predictions'].append(predictions)
            records['runs'].append(dict(seed=seed, folder=str(folder), best_epoch=trained['best_epoch'],
                                        checkpoint_sha256=description['checkpoint_sha256'],
                                        shared_initialization_sha256=config['shared_initialization_sha256'],
                                        global_steps=trained['global_steps'],
                                        supervised_positions_per_epoch=[r['supervised_positions'] for r in history],
                                        training_seconds=float(sum(r['training_seconds'] for r in history)),
                                        elapsed_seconds=trained['elapsed_seconds'], peak_gpu_memory_mb=trained['peak_gpu_memory_mb'],
                                        coverage_seconds=description['elapsed_seconds'],
                                        history=[dict(epoch=r['epoch'], train_loss=r['train_loss'],
                                                      validation=r['validation']) for r in history]))
    reference = arms['click']['predictions'][0]
    for records in arms.values():
        for prediction in records['predictions']:
            for field in ['rows', 'targets', 'seen', 'target_train_frequency', 'target_content_present']:
                np.testing.assert_array_equal(prediction[field], reference[field])
        records['groups'] = aggregate_groups(records['diagnostics'])
    return data, arms


def paired_differences(arms):
    result = {}
    reference = arms['click']['predictions']
    frequencies = reference[0]['target_train_frequency']
    masks = dict(all=np.ones(len(frequencies), dtype=bool), cold_0=frequencies == 0,
                 tail_1_5=(frequencies >= 1) & (frequencies <= 5),
                 mid_6_50=(frequencies >= 6) & (frequencies <= 50), head_gt50=frequencies > 50)
    for arm, records in arms.items():
        if arm == 'click':
            continue
        result[arm] = {}
        for group, mask in masks.items():
            result[arm][group] = {}
            for field, metric in [('hits_at_10', 'Recall@10'), ('hits_at_100', 'Recall@100'),
                                  ('hits_at_500', 'Recall@500'), ('ndcg_at_10', 'NDCG@10')]:
                differences = np.stack([a[field][mask].astype(float) - b[field][mask].astype(float)
                                        for a, b in zip(records['predictions'], reference)])
                result[arm][group][metric] = dict(direction='masking minus unmasked control',
                                                  mean=float(differences.mean()), per_seed=differences.mean(1).tolist(),
                                                  **confidence_intervals(differences))
    return result


def optional_json(path):
    return dict(status='available', path=str(path), data=json.loads(path.read_text())) if path.exists() else dict(status='pending', path=str(path))


def freeze_selection(path, selection):
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != selection:
            raise ValueError('The locked masking selection changed; use a new study output directory.')
    else:
        path.write_text(json.dumps(selection, indent=2) + '\n')


def make_figures(reports, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = reports / 'figures'
    folder.mkdir(exist_ok=True)
    names = list(result['arms'])
    labels = ['ID mask 0%', 'ID mask 30%', 'ID mask 70%']
    colors = ['#356da6', '#db8a4c', '#4a9779']
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, group in zip(axes, ['cold_0', 'tail_1_5', 'all']):
        means = [metric_mean(result['arms'][name], group, 'Recall@500') * 100 for name in names]
        errors = [result['arms'][name]['groups'][group]['Recall@500']['sample_std'] * 100 for name in names]
        axis.bar(np.arange(len(names)), means, yerr=errors, capsize=4, color=colors[:len(names)])
        axis.set_xticks(np.arange(len(names)), labels[:len(names)], rotation=15)
        axis.set_title(group)
        axis.set_ylabel('Validation Recall@500 (%)')
        axis.grid(axis='y', alpha=.2)
    figure.tight_layout()
    figure.savefig(folder / 'masking_coverage.png', dpi=160)
    plt.close(figure)
    seeds = result['specification']['seeds']
    figure, axes = plt.subplots(1, len(seeds), figsize=(4 * len(seeds), 3.5), squeeze=False)
    for index, seed in enumerate(seeds):
        for name, label, color in zip(names, labels, colors):
            history = result['arms'][name]['runs'][index]['history']
            axes[0, index].plot([row['epoch'] for row in history],
                                [row['validation']['NDCG@10'] for row in history], marker='o', label=label, color=color)
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


def write_readme(reports, result):
    selection = result['model_selection']
    names = list(result['arms'])
    lines = ['# 冷启动与长尾：候选覆盖诊断、ID 遮蔽与内容召回', '',
             f"ID 遮蔽选择结果：`{selection['selected_arm']}`。选择只依据重复使用的验证队列，是开发结果，尚无新测试确认。", '',
             '## 问题与评估口径', '',
             f"固定 {result['training_users']:,} 个训练用户、{result['candidate_items']:,} 个候选广告与 {result['evaluation_users']:,} 名验证用户。每人只有一个目标，Recall@K 与单目标 HR@K 相同。", '',
             '频次为所有用户训练前缀里的广告事件数，包含曝光、点击和未知行为；不是点击数。cold=0、tail=1–5、mid=6–50、head>50。所有搜索过滤当前已见广告，精确内积搜索整个固定目录。',
             '原先测试分桶人数不能直接搬来使用：本报告验证分桶为 cold 659、tail 1326、mid 2966、head 5049。', '',
             '## 先诊断原模型：Top-10 与 Top-500 是不同问题', '',
             '| 目标分组 | 验证用户 | Recall@10 | Recall@100 | Recall@500 |', '|---|---:|---:|---:|---:|']
    for group in GROUPS:
        row = result['arms']['click']['groups'][group]
        lines.append(f"| {group} | {row['users']} | " + ' | '.join(percent(row[f'Recall@{k}']['mean']) for k in KS) + ' |')
    lines += ['', '原模型冷启动和长尾中仍有大量目标在前 500 之外。对这些用户，重排同一批候选无法找回根本没有被召回的答案。全体进入 Top-500 的目标比例也只是候选覆盖上限，不是一个已实现排序模型的效果。', '',
              '## 训练点击信号与内容特征诊断', '']
    audit = result['frequency_audit']
    if audit['status'] == 'available':
        data = audit['data']
        catalog = data['catalog_groups']['tail_1_5']
        validation = data['validation_target_groups']['tail_1_5']
        lines += [f"目录中 {catalog['items']:,} 个长尾广告里，{catalog['zero_training_click_fraction']:.2%} 没有训练点击；验证长尾目标用户中，这一比例为 {validation['zero_training_click_fraction']:.2%}。两个分母不同，不能把“约 82%”写成所有验证用户的比例。", '',
                  '点击 InfoNCE 的这些广告缺少自身点击正目标；它们仍可能作为历史上下文或负例影响梯度。属性与 81 号文本向量虽然能提供表示，不代表已经学到足够的冷/尾召回能力。', '',
                  '完整特征覆盖、向量范数和高基数字段检查见 `frequency_audit.json`。']
    else:
        lines += ['频次与特征审计尚未生成，当前不推测未点击广告比例。']
    lines += ['', '## 控制实验：训练时随机遮蔽广告 ID', '',
              '比较 0%、30%、70% ID 遮蔽；每个 optimizer batch 为训练可见广告抽一次独立掩码，在历史、正样本和负样本间共享。只把被遮广告的 ID 查表索引改成 OOV=0，属性和文本保留；不做倒置缩放，验证时关闭遮蔽。独立随机生成器避免消耗普通 dropout、打乱和负采样的随机流。',
              '同数据、监督位置、每目标 64 个负例、初值、模型大小、学习率、batch size 与 5 轮预算。每种遮蔽率使用三个种子；每个 checkpoint 仍按整体验证 NDCG@10 选取。', '',
              '| 方案 | 全体 NDCG@10 | 全体 Recall@500 | 冷启动 Recall@500 | 长尾 Recall@500 |', '|---|---:|---:|---:|---:|']
    for name in names:
        row = result['arms'][name]
        lines.append(f"| {name} | {metric_mean(row, 'all', 'NDCG@10'):.6f} | {percent(metric_mean(row, 'all', 'Recall@500'))} | {percent(metric_mean(row, 'cold_0', 'Recall@500'))} | {percent(metric_mean(row, 'tail_1_5', 'Recall@500'))} |")
    lines += ['', '![候选覆盖](figures/masking_coverage.png)', '', '![逐种子验证曲线](figures/validation_curves.png)', '',
              '### 按训练前约定的规则选择', '',
              '整体 NDCG@10 至少保留对照的 95%；冷启动和长尾 Recall@500 各自都不得下降；通过后最大化两组 Recall@500 的等权均值，并要求严格改善，相同则选更小遮蔽率。没有合格改善就保留 0% 对照。', '',
              '| 方案 | 合格 | 冷/尾宏平均 Recall@500 | 未通过原因 |', '|---|---|---:|---|']
    for row in selection['rows']:
        lines.append(f"| {row['name']} | {'是' if row['eligible'] else '否'} | {percent(row['cold_tail_macro_recall500'])} | {', '.join(row['rejection_reasons']) or '无'} |")
    lines += ['', '### 配对不确定性', '',
              '以下为相对 0% 对照的差值。用户和三个训练种子一起配对 bootstrap；仅三个种子不能代表所有初始化。候选选择按预先给定的均值与门槛执行，区间用于解释证据强弱。', '',
              '| 方案 | 分组/指标 | 平均差值 | 95% 用户与种子配对区间 |', '|---|---|---:|---|']
    for name, groups in result['paired_differences'].items():
        for group, metric in [('all', 'NDCG@10'), ('cold_0', 'Recall@500'), ('tail_1_5', 'Recall@500')]:
            row = groups[group][metric]
            lo, hi = row['paired_seed_and_user_95pct']
            lines.append(f"| {name} | {group}/{metric} | {row['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', '## 内容召回与固定候选预算融合', '']
    fusion = result['fusion']
    if fusion['metrics']['status'] == 'available' and fusion['selection']['status'] == 'available':
        data, chosen = fusion['metrics']['data'], fusion['selection']['data']
        lines += [f"固定候选预算是 {data['candidate_budget']}，验证选出的额外内容通道配额为 {chosen['selected_quota']}。保留原 Top-10，替换较低名次候选；没有训练独立排序器，也没有据此声称 Top-10 改善。", '',
                  '| 内容配额 | 全体 Recall@500 | 冷启动 Recall@500 | 长尾 Recall@500 |', '|---:|---:|---:|---:|']
        for row in data['quota_sweep']:
            lines.append(f"| {row['quota']} | {percent(row['groups']['all']['Recall@500'])} | {percent(row['groups']['cold_0']['Recall@500'])} | {percent(row['groups']['tail_1_5']['Recall@500'])} |")
        lines += ['', '另有 learned Top-500 与 content Top-500 的未排序并集诊断，其候选数最多为 1000。该并集命中必须称为至多 1000 个候选的覆盖率，不能作为固定 500 融合的 Recall@500。详细结果见[融合报告](fusion.md)与 `fusion_metrics.json`。']
        if chosen['selected_quota'] == 0:
            lines += ['', '原始固定配额融合没有通过门槛，保留配额 0。此时与对照的差值区间 [0,0] 是同一结果与自身相比，不代表统计确定的改进。']
    else:
        lines += ['内容通道或融合结果尚未生成，当前报告将其标记为 pending，不填补结果。']
    lines += ['', '### 基于失败诊断的探索修正：保护已召回的稀有候选', '']
    protected = result['fusion_protected']
    if protected['metrics']['status'] == 'available' and protected['selection']['status'] == 'available':
        data, chosen = protected['metrics']['data'], protected['selection']['data']
        lines += ['这是查看前一版验证失败后单独登记的 follow-up，不能包装成最初预注册成功或独立确认。原失败结果保持原样。',
                  '保护 learned Top-10 和其中所有训练频次不超过 5 的候选，只从原列表尾部移除非稀有广告，再加入内容通道提供、且不在原 Top-500 的新稀有广告；始终维持 500 个候选。', '',
                  f"验证选择的额外通道配额为 {chosen['selected_quota']}。", '',
                  '| 内容配额 | 全体 Recall@500 | 冷启动 Recall@500 | 长尾 Recall@500 |', '|---:|---:|---:|---:|']
        for row in data['quota_sweep']:
            lines.append(f"| {row['quota']} | {percent(row['groups']['all']['Recall@500'])} | {percent(row['groups']['cold_0']['Recall@500'])} | {percent(row['groups']['tail_1_5']['Recall@500'])} |")
        lines += ['', '这条候选构造规则保持原始前十，因此当前 Top-10 没有提升；覆盖改善只表示给未来排序增加了候选空间。完整门槛、配对区间与保护不变量见[探索修正报告](fusion_protected.md)及 `fusion_protected_metrics.json`。']
        if chosen['selected_quota'] == 0:
            lines += ['本次修正也未选择额外候选，继续保留原通道。']
    else:
        lines += ['保护稀有候选的 follow-up 尚未生成，标记为 pending，不推测是否有效。']
    lines += ['', '### 基于 ID 遮蔽取舍的探索：把 mask70 作为互补通道', '']
    masked = result['fusion_masked']
    if masked['metrics']['status'] == 'available' and masked['selection']['status'] == 'available':
        data, chosen = masked['metrics']['data'], masked['selection']['data']
        lines += ['该实验是在看到 mask70 提高长尾覆盖但降低冷启动覆盖后另行登记的验证驱动 follow-up，不属于最初预注册，也没有独立测试确认。',
                  '每个基线种子只与相同种子的 mask70 模型配对。保护基线 Top-10 及原 Top-500 中所有频次不超过 5 的候选，只用 mask70 提供的新稀有广告替换尾部非稀有广告，候选总数保持 500。', '',
                  f"验证选择的互补候选配额为 {chosen['selected_quota']}；该选择没有把 mask70 替换为主模型，也没有更新原索引。", '',
                  '| mask70 配额 | 全体 Recall@500 | 冷启动 Recall@500 | 长尾 Recall@500 |', '|---:|---:|---:|---:|']
        for row in data['quota_sweep']:
            lines.append(f"| {row['quota']} | {percent(row['groups']['all']['Recall@500'])} | {percent(row['groups']['cold_0']['Recall@500'])} | {percent(row['groups']['tail_1_5']['Recall@500'])} |")
        if chosen['selected_quota'] == 0:
            lines += ['', '这次互补通道也保留 0 配额；与基线完全相同时，零差值区间不表示优化收益。']
        else:
            lines += ['', '候选池在验证选择规则下有所取舍，仍需结合配对区间判断不确定性。原 Top-10 按规则保持不变，当前不能声称最终推荐命中改善。']
        lines += ['', '| 相对基线的选定方案 | Recall@500 差值 | 95% 用户与种子配对区间 |', '|---|---:|---|']
        for group in ['all', 'cold_0', 'tail_1_5']:
            interval = data['selected_paired_recall500_intervals'][group]
            lo, hi = interval['paired_seed_and_user_95pct']
            lines.append(f"| {group} | {interval['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
        lines += ['', '详细来源摘要、seed 配对和保护不变量见[互补通道报告](fusion_masked.md)、`fusion_masked_metrics.json` 与登记的 `masked_fusion_plan.json`。']
    else:
        lines += ['mask70 互补候选 follow-up 尚未生成，标记为 pending，不填补覆盖收益。']
    lines += ['', '## 是否现在接独立排序', '',
              '不把 MLP 排序作为本轮默认下一步。冷启动、长尾仍有大量答案未进入候选，排序只能处理已入选广告；先以固定预算的候选覆盖改善为主。若后续覆盖足够，再固定候选和时间协议训练轻量排序，并评估条件命中与端到端收益。',
              '当前内容通道与 ID 遮蔽都是验证开发实验。不能把验证选出的收益当成独立测试确认，也不自动替换现有主模型、广告向量或持久化 HNSW 索引。', '',
              '## 成本与可复现证据', '',
              '| 方案 | 种子 | 最佳 epoch | 训练秒数 | 峰值显存 MiB |', '|---|---:|---:|---:|---:|']
    for name in names:
        for row in result['arms'][name]['runs']:
            memory = f"{row['peak_gpu_memory_mb']:.1f}" if row['peak_gpu_memory_mb'] is not None else 'N/A'
            lines.append(f"| {name} | {row['seed']} | {row['best_epoch']} | {row['training_seconds']:.1f} | {memory} |")
    lines += ['', '训练秒数为各 epoch 训练部分之和，不含验证。参考模型来自此前运行，共享机器的负载不同，这不是严格性能基准。完整逐种子数据与曲线保存在 `metrics.json`。', '',
              '- [协议配置](../../configs/cold_start.json)、[编排脚本](../../scripts/run_cold_start.py)、[ID 遮蔽模型](../../taac_retrieval/model.py)。',
              '- [覆盖诊断](diagnostics.md)、[覆盖实现](../../taac_retrieval/coverage.py)、[频次审计](../../scripts/audit_cold_start_features.py)。',
              '- `model_selection.json`：预定门槛、全部候选及保留对照的原因。',
              '- `verification.json`：初值、checkpoint、队列、已见历史和逐用户指标的核对。', '',
              '```bash', '.project_venv/bin/python scripts/run_cold_start_pipeline.py \\',
              '  --root artifacts/cold_start_reproduction \\',
              '  --report-dir reports/cold_start_reproduction --gpus 0,2,3,4,5,6', '```', '',
              '该完整入口会补齐参考诊断、特征审计、内容通道与融合，再运行 ID 遮蔽实验。新实验用新 root；已完成训练的目录在相同 wrapper 命令后追加 `--stage finish` 继续诊断与报告，`--prerequisites-only` 只复核前置证据而不训练。当前研究只以验证目标评估与选参，未构造测试 Dataset；数组缓存本身仍包含完整公开序列及测试位置元数据。', '']
    (reports / 'README.md').write_text('\n'.join(lines))


def report(root, reports):
    root, reports = Path(root).resolve(), Path(reports).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    data, evidence = load_evidence(root, plan)
    summaries = {name: {key: row[key] for key in ['name', 'probability', 'groups', 'runs']}
                 for name, row in evidence.items()}
    selection = select_configuration(list(summaries.values()), plan['specification']['variant_selection'])
    selection['repeated_validation_development'] = True
    selection['input_checkpoint_sha256'] = {name: [row['checkpoint_sha256'] for row in records['runs']]
                                            for name, records in evidence.items()}
    freeze_selection(root / 'model_selection.json', selection)
    frequency_audit = optional_json(root / 'frequency_audit.json')
    fusion = dict(metrics=optional_json(root / 'fusion/metrics.json'), selection=optional_json(root / 'fusion/selection.json'))
    protected = dict(metrics=optional_json(root / 'fusion_protected/metrics.json'),
                     selection=optional_json(root / 'fusion_protected/selection.json'),
                     amendment=optional_json(root / 'fusion_protected_plan.json'))
    masked = dict(metrics=optional_json(root / 'fusion_masked/metrics.json'),
                  selection=optional_json(root / 'fusion_masked/selection.json'),
                  amendment=optional_json(root / 'masked_fusion_plan.json'))
    if masked['metrics']['status'] == 'available' and masked['amendment']['status'] != 'available':
        raise FileNotFoundError('Masked-fusion results require their registered masked_fusion_plan.json.')
    cohort_sha = evidence['click']['diagnostics'][0]['evaluation_cohort_sha256']
    for item in [frequency_audit, fusion['metrics'], fusion['selection'], protected['metrics'], protected['selection'],
                 masked['metrics'], masked['selection'], masked['amendment']]:
        if item['status'] == 'available' and item['data'].get('test_targets_read', False):
            raise AssertionError('Auxiliary evidence must not use test targets.')
        if item['status'] == 'available' and item['data'].get('selection_split', item['data'].get('split', 'val')) != 'val':
            raise AssertionError('Auxiliary evidence must use validation selection.')
        if item['status'] == 'available' and item['data'].get('evaluation_cohort_sha256', cohort_sha) != cohort_sha:
            raise AssertionError('Auxiliary evidence uses a different cohort.')
    result = dict(specification=plan['specification'], training_users=data.num_users, candidate_items=data.num_items,
                  evaluation_users=len(evidence['click']['predictions'][0]['rows']), arms=summaries,
                  paired_differences=paired_differences(evidence), model_selection=selection,
                  frequency_audit=frequency_audit, fusion=fusion, fusion_protected=protected,
                  fusion_masked=masked, test_targets_read=False,
                  repeated_validation_development=True, original_index_unchanged=True)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    (reports / 'model_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    if frequency_audit['status'] == 'available':
        (reports / 'frequency_audit.json').write_text(json.dumps(frequency_audit['data'], indent=2) + '\n')
    if fusion['metrics']['status'] == 'available':
        (reports / 'fusion_metrics.json').write_text(json.dumps(fusion['metrics']['data'], indent=2) + '\n')
    if protected['metrics']['status'] == 'available':
        (reports / 'fusion_protected_metrics.json').write_text(json.dumps(protected['metrics']['data'], indent=2) + '\n')
    if masked['metrics']['status'] == 'available':
        (reports / 'fusion_masked_metrics.json').write_text(json.dumps(masked['metrics']['data'], indent=2) + '\n')
        (reports / 'masked_fusion_plan.json').write_text(json.dumps(masked['amendment']['data'], indent=2) + '\n')
    verification = dict(new_training_runs=sum(len(row['runs']) for name, row in evidence.items() if name != 'click'),
                        reused_control_runs=len(evidence['click']['runs']), full_training=True,
                        matched_shared_initialization=True, canonical_cache_paths_matched=True,
                        rows_targets_seen_and_frequency_matched=True, checkpoint_sha256_matched=True,
                        per_user_metrics_recomputed=True, validation_only=True, test_targets_read=False,
                        old_model_or_index_deployed=False, report_source_sha256=digest(Path(__file__)),
                        frequency_audit_status=frequency_audit['status'], fusion_status=fusion['metrics']['status'],
                        protected_fusion_status=protected['metrics']['status'],
                        masked_fusion_status=masked['metrics']['status'])
    (reports / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    make_figures(reports, result)
    write_readme(reports, result)
    print(json.dumps(dict(report=str(reports / 'README.md'), selection=selection), indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/cold_start')
    parser.add_argument('--report-dir', default='reports/cold_start')
    args = parser.parse_args()
    report(args.root, args.report_dir)
