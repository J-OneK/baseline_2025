"""Build the day-4/7 evidence, case analysis, figures and interview materials."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .data import PreparedData, load_array
from .metrics import recommendation_metrics
from .study import file_hash


def confidence_intervals(differences, samples=2000, seed=771):
    """Paired users; additionally resample the small, fixed training-seed set."""
    rng = np.random.default_rng(seed)
    per_user = differences.mean(0)
    conditional, two_way = [], []
    for start in range(0, samples, 100):
        count = min(100, samples - start)
        users = rng.integers(differences.shape[1], size=(count, differences.shape[1]))
        seeds = rng.integers(differences.shape[0], size=(count, differences.shape[0]))
        conditional.extend(per_user[users].mean(1))
        two_way.extend(differences[seeds[:, :, None], users[:, None, :]].mean((1, 2)))
    return dict(paired_user_95pct=np.quantile(conditional, [.025, .975]).tolist(),
                paired_seed_and_user_95pct=np.quantile(two_way, [.025, .975]).tolist(),
                note='Only three training seeds: resampling them does not establish broad seed-population generalization.')


def load_evidence(root, reports, config):
    selection = json.loads((reports / 'model_selection.json').read_text())
    for path, expected in selection['history_sha256'].items():
        if file_hash(path) != expected:
            raise AssertionError('A development result changed after model selection was locked.')
    runs, predictions = {}, {}
    for name in ['baseline', 'selected']:
        runs[name], predictions[name] = [], []
        for seed in config['confirmation_seeds']:
            train_dir = root / 'confirmation' / f'{name}_seed{seed}'
            trained = json.loads((train_dir / 'result.json').read_text())
            training_config = json.loads((train_dir / 'config.json').read_text())
            history = json.loads((train_dir / 'history.json').read_text())
            evaluated = json.loads((root / 'evaluation' / f'{name}_seed{seed}.json').read_text())
            if trained['test'] is not None or not training_config['skip_test']:
                raise AssertionError('Confirmation training consumed test labels before the final evaluation stage.')
            if len(history) != config['epochs'] or training_config['max_steps'] != 0:
                raise AssertionError('Incomplete/smoke training is not valid evidence.')
            best = max(history, key=lambda r: (r['validation']['NDCG@10'], r['validation']['HR@10'], -r['epoch']))
            if best['epoch'] != evaluated['selected_epoch']:
                raise AssertionError('Checkpoint was not selected by validation.')
            with np.load(root / 'evaluation' / f'{name}_seed{seed}.npz') as source:
                prediction = {key: source[key].copy() for key in source.files}
            hits, ndcg = recommendation_metrics(prediction['predictions'], prediction['targets'])
            np.testing.assert_allclose([hits.mean(), ndcg.mean()],
                                       [evaluated['metrics']['HR@10'], evaluated['metrics']['NDCG@10']])
            runs[name].append(dict(seed=seed, train=trained, evaluation=evaluated,
                                   config=training_config, history=history))
            predictions[name].append(prediction)
    reference = predictions['baseline'][0]
    for name in predictions:
        for prediction in predictions[name]:
            for field in ['rows', 'targets']:
                np.testing.assert_array_equal(prediction[field], reference[field])
    for j in range(len(config['confirmation_seeds'])):
        a, b = runs['baseline'][j], runs['selected'][j]
        if a['config']['shared_initialization_sha256'] != b['config']['shared_initialization_sha256']:
            raise AssertionError('Confirmation models did not share initial weights.')
        for field in ['epochs', 'batch_size', 'hidden', 'blocks', 'heads', 'dropout', 'lr', 'evaluation_cohort_sha256']:
            if a['config'][field] != b['config'][field]:
                raise AssertionError(f'Unequal confirmation setting: {field}')
    return selection, runs, predictions


def aggregates(runs, predictions):
    summary, deltas = {}, {}
    for name in runs:
        summary[name] = {}
        for metric in ['HR@10', 'NDCG@10']:
            values = [r['evaluation']['metrics'][metric] for r in runs[name]]
            summary[name][metric] = dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)), values=values)
    for metric, key in [('HR@10', 'hits'), ('NDCG@10', 'ndcg')]:
        difference = np.stack([b[key] - a[key] for a, b in zip(predictions['baseline'], predictions['selected'])])
        mean = float(difference.mean())
        baseline = summary['baseline'][metric]['mean']
        deltas[metric] = dict(absolute=mean, relative=mean / baseline if baseline else None,
                             per_seed=difference.mean(1).tolist(), **confidence_intervals(difference))
    return summary, deltas


def group_analysis(data, predictions):
    reference = predictions['baseline'][0]
    targets, rows = reference['targets'], reference['rows']
    counts = data.train_counts[targets]
    present = load_array(data.root, 'mm_present')[targets]
    groups = {
        'cold_0': counts == 0, 'tail_1_5': (counts >= 1) & (counts <= 5),
        'mid_6_50': (counts >= 6) & (counts <= 50), 'head_gt50': counts > 50,
        'history_le50': data.test_cut[rows] <= 50, 'history_gt50': data.test_cut[rows] > 50,
        'content_present': present, 'content_missing': ~present,
    }
    result = {}
    for group, selected in groups.items():
        result[group] = dict(users=int(selected.sum()))
        for name in predictions:
            result[group][name] = {metric: float(np.mean([p[key][selected].mean() for p in predictions[name]]))
                                   if selected.any() else None
                                   for metric, key in [('HR@10', 'hits'), ('NDCG@10', 'ndcg')]}
    return result


def cases(data, predictions):
    baseline, selected = predictions['baseline'][0], predictions['selected'][0]
    counts = data.train_counts[baseline['targets']]
    masks = {
        'selected_hit_baseline_miss': (selected['hits'] == 1) & (baseline['hits'] == 0),
        'baseline_hit_selected_miss': (baseline['hits'] == 1) & (selected['hits'] == 0),
        'cold_miss': (counts == 0) & (selected['hits'] == 0),
        'warm_shared_miss': (counts > 0) & (selected['hits'] == 0) & (baseline['hits'] == 0),
    }
    present = load_array(data.root, 'mm_present')
    result = {}
    for name, mask in masks.items():
        examples = []
        for position in np.flatnonzero(mask)[:5]:
            row, target = int(baseline['rows'][position]), int(baseline['targets'][position])
            cut = int(data.test_cut[row])
            examples.append(dict(user_rid=int(data.user_rids[row]), target_oid=int(data.catalog_oids[target]),
                                 target_training_events=int(data.train_counts[target]), target_has_content=bool(present[target]),
                                 history_events=cut,
                                 recent_history=[dict(item_oid=int(data.catalog_oids[item]), action=int(action), timestamp=int(stamp))
                                                 for item, action, stamp in zip(data.items[row, max(0, cut-10):cut],
                                                                              data.actions[row, max(0, cut-10):cut],
                                                                              data.timestamps[row, max(0, cut-10):cut])],
                                 baseline_top10_oids=data.catalog_oids[baseline['predictions'][position]].tolist(),
                                 selected_top10_oids=data.catalog_oids[selected['predictions'][position]].tolist()))
        result[name] = dict(total_users=int(mask.sum()), examples=examples)
    return result


def figures(reports, runs, summary, groups, benchmark):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    output = reports / 'figures'
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for name, label, color in [('baseline', 'Next-event BCE', '#356a9a'), ('selected', 'Click InfoNCE-64', '#cf6c2b')]:
        for axis, metric in zip(axes, ['NDCG@10', 'HR@10']):
            values = np.array([[row['validation'][metric] for row in run['history']] for run in runs[name]])
            epochs = np.arange(1, values.shape[1] + 1)
            axis.plot(epochs, values.mean(0), 'o-', color=color, label=label)
            axis.fill_between(epochs, values.mean(0)-values.std(0), values.mean(0)+values.std(0), color=color, alpha=.15)
            axis.set(xlabel='Training epoch', ylabel=f'Validation {metric}')
            axis.grid(alpha=.2)
    axes[0].legend()
    for extension in ['png', 'svg']:
        fig.savefig(output / f'validation_curves.{extension}', dpi=180)
    plt.close(fig)
    buckets = ['cold_0', 'tail_1_5', 'mid_6_50', 'head_gt50']
    x = np.arange(len(buckets))
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for name, offset, label, color in [('baseline', -.18, 'Next-event BCE', '#356a9a'),
                                       ('selected', .18, 'Click InfoNCE-64', '#cf6c2b')]:
        axis.bar(x+offset, [groups[b][name]['HR@10']*100 if groups[b][name]['HR@10'] is not None else 0 for b in buckets],
                 width=.36, label=label, color=color)
    axis.set_xticks(x, [f'{label}\nn={groups[b]["users"]:,}' for b, label in
                       zip(buckets, ['Cold (0)', 'Tail (1-5)', 'Mid (6-50)', 'Head (>50)'])])
    axis.set(ylabel='Test HR@10 (%)', xlabel='Target frequency in training prefixes')
    axis.legend()
    axis.grid(axis='y', alpha=.2)
    for extension in ['png', 'svg']:
        fig.savefig(output / f'frequency_buckets.{extension}', dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for row in benchmark['validation_configurations']:
        axis.scatter(row['single_query_p95_ms'], row['ANNRecall@10']*100, color='#cf6c2b', s=60)
        axis.annotate(f"ef={row['efSearch']}", (row['single_query_p95_ms'], row['ANNRecall@10']*100), xytext=(5,-12), textcoords='offset points')
    axis.scatter(benchmark['flat_validation_latency']['single_query_p95_ms'], 100, color='#356a9a', marker='s', s=65, label='FlatIP exact')
    chosen = benchmark['selected']
    axis.scatter(chosen['single_query_p95_ms'], chosen['ANNRecall@10']*100, marker='*', s=190, color='#244d2f', label='Selected on validation')
    axis.axhline(benchmark['configuration']['minimum_validation_ann_recall']*100, color='gray', linestyle='--', linewidth=1)
    axis.set(xlabel='Warmed CPU single-query p95 (ms, 1 thread)', ylabel='Validation ANN Recall@10 (%)', xscale='log')
    axis.set_ylim(min(row['ANNRecall@10'] for row in benchmark['validation_configurations'])*100 - 1.5, 100.8)
    axis.grid(alpha=.2)
    axis.legend()
    for extension in ['png', 'svg']:
        fig.savefig(output / f'ann_tradeoff.{extension}', dpi=180)
    plt.close(fig)


def report(args):
    root, reports = Path(args.root).resolve(), Path(args.output).resolve()
    reports.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    selection, runs, predictions = load_evidence(root, reports, config)
    summary, deltas = aggregates(runs, predictions)
    data = PreparedData(root / 'scale_data')
    cohort = np.load(root / 'confirm_eval_indices.npy')
    np.testing.assert_array_equal(predictions['baseline'][0]['rows'], cohort)
    previous = PreparedData(args.baseline_root + '/data')
    overlap = np.intersect1d(data.user_rids[cohort], previous.user_rids[previous.eval_indices])
    if len(overlap):
        raise AssertionError('Previously inspected test users leaked into the confirmation cohort.')
    groups = group_analysis(data, predictions)
    examples = cases(data, predictions)
    benchmark = json.loads((root / 'ann/benchmark.json').read_text())
    if benchmark['export']['checkpoint_sha256'] != runs['selected'][0]['evaluation']['checkpoint_sha256']:
        raise AssertionError('ANN benchmark does not use the predeclared primary-seed model.')
    with np.load(root / 'ann/test_search.npz') as searched:
        positions = np.searchsorted(predictions['selected'][0]['rows'], searched['rows'])
        exact_torch = predictions['selected'][0]['predictions'][positions]
        vectors = np.load(root / 'export/item_vectors.npy', mmap_mode='r')
        with np.load(root / 'export/test_queries.npz') as queries:
            query_vectors = queries['queries'][searched['test_query_indices']].astype(np.float64)
        faiss_scores = np.einsum('qd,qkd->qk', query_vectors, vectors[searched['exact']].astype(np.float64))
        torch_scores = np.einsum('qd,qkd->qk', query_vectors, vectors[exact_torch].astype(np.float64))
        if not np.allclose(faiss_scores, torch_scores, rtol=2e-5, atol=2e-5):
            raise AssertionError('Faiss exact top-k does not agree with the original Torch evaluation.')
        flat_torch_recall = float((searched['exact'][:, :, None] == exact_torch[:, None, :]).any(-1).mean())
    speedup = benchmark['test_flat_latency']['single_query_p95_ms'] / benchmark['test_hnsw_latency']['single_query_p95_ms']
    result = dict(development_selection=selection, training_users=data.num_users, candidate_items=data.num_items,
                  evaluation_users=len(cohort), previous_test_user_overlap=0, confirmation_seeds=config['confirmation_seeds'],
                  aggregates=summary, deltas=deltas, frequency_and_context_groups=groups,
                  runs={name: [dict(seed=r['seed'], epoch=r['evaluation']['selected_epoch'],
                                         metrics=r['evaluation']['metrics'], seconds=r['train']['elapsed_seconds'],
                                         peak_gpu_memory_mb=r['train']['peak_gpu_memory_mb']) for r in values]
                        for name, values in runs.items()},
                  popularity=runs['baseline'][0]['evaluation']['popularity'],
                  ann=dict(selected_ef=benchmark['selected']['efSearch'], test=benchmark['test'],
                           geometry=benchmark['configuration'].get('hnsw_geometry', 'raw_ip'),
                           flat_vs_torch_exact_id_recall=flat_torch_recall,
                           flat_vs_torch_exact_scores_verified=True,
                           cpu_single_query_p95_speedup=speedup,
                           flat_test_latency=benchmark['test_flat_latency'], hnsw_test_latency=benchmark['test_hnsw_latency'],
                           build=benchmark['build']))
    (reports / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    (reports / 'failure_cases.json').write_text(json.dumps(examples, indent=2) + '\n')
    figures(reports, runs, summary, groups, benchmark)
    write_documents(reports, root, result, benchmark)
    source_root = Path(__file__).resolve().parents[1]
    paths = list((source_root / 'taac_retrieval').glob('*.py')) + list((source_root / 'tests').glob('*.py'))
    paths += list((source_root / 'scripts').glob('*.py')) + list((source_root / 'configs').glob('*.json'))
    (reports / 'code_manifest.json').write_text(json.dumps({str(p.relative_to(source_root)): file_hash(p) for p in sorted(paths)}, indent=2) + '\n')
    import torch, pyarrow, faiss, matplotlib
    environment = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                       torch_cuda=torch.version.cuda, numpy=np.__version__, pyarrow=pyarrow.__version__,
                       faiss=faiss.__version__, matplotlib=matplotlib.__version__,
                       gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       cpu_model=benchmark['cpu_model'])
    (reports / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    print(json.dumps(dict(aggregates=summary, deltas=deltas, ann=result['ann']), indent=2))
    return result


def write_documents(reports, root, result, benchmark):
    a, b = result['aggregates']['baseline'], result['aggregates']['selected']
    lines = ['# 第 4–7 天：完整项目结果', '',
             f"扩大实验：{result['training_users']:,} 名有效用户、{result['candidate_items']:,} 个固定候选广告；",
             f"验证、测试各 {result['evaluation_users']:,} 名用户，排除之前查看过测试结果的用户；种子 {result['confirmation_seeds']}。", '',
             '## 第四天：训练目标消融（只看验证集）', '',
             '| 配置 | 验证 NDCG@10（两种子均值） | 验证 HR@10 |', '|---|---:|---:|']
    for row in result['development_selection']['ranking']:
        lines.append(f"| {row['name']} | {row['validation_ndcg']:.6f} | {row['validation_hr']:.4%} |")
    lines += ['', '依据预先约定的平均验证 NDCG@10 选择 `'+result['development_selection']['selected_variant']+'`。',
              '点击 BCE-64 是负样本数量对照；它与 InfoNCE-64 使用相同点击监督、负样本数和采样策略。',
              '所有配置保留；开发阶段没有评估新模型的测试集。选择记录见 `model_selection.json`。', '',
              '## 第五天：扩大规模、三个种子复核', '',
              '| 模型 | 测试 HR@10 | 测试 NDCG@10 |', '|---|---:|---:|',
              f"| 热门广告 | {result['popularity']['HR@10']:.4%} | {result['popularity']['NDCG@10']:.6f} |",
              f"| 原目标 BCE 基线 | {a['HR@10']['mean']:.4%} | {a['NDCG@10']['mean']:.6f} |",
              f"| 验证选定方案 | {b['HR@10']['mean']:.4%} | {b['NDCG@10']['mean']:.6f} |", '',
              '| 种子 | 模型 | 最优 epoch | 测试 HR@10 | 测试 NDCG@10 |', '|---|---|---:|---:|---:|']
    for name in ['baseline', 'selected']:
        for run in result['runs'][name]:
            lines.append(f"| {run['seed']} | {name} | {run['epoch']} | {run['metrics']['HR@10']:.4%} | {run['metrics']['NDCG@10']:.6f} |")
    lines += ['', '两个模型在同一种子下共享初始化、用户、词表、候选、网络和学习率，均训练 5 轮；',
              '点击监督位置数和负样本数不同，所以这不是等 FLOPs 对照。epoch 只由验证集选择。', '']
    for metric, delta in result['deltas'].items():
        low, high = delta['paired_seed_and_user_95pct']
        qualifier = '区间跨零，证据不足以确认稳定收益。' if low <= 0 <= high else (
            '本次复核观察到一致的正向差异。' if low > 0 else '本次复核观察到负向差异。')
        relative = f"{delta['relative']:+.2%}" if delta['relative'] is not None else '不可计算'
        lines.append(f"- {metric}：绝对差 {delta['absolute']:+.6f}，相对差 {relative}；种子+用户配对 bootstrap 95% 区间 [{low:+.6f}, {high:+.6f}]。{qualifier}")
    lines += ['', '三个种子仍然有限；区间与结论只针对本次训练和本地评估协议，不代表线上 A/B 或官方成绩。', '',
              '### 分桶及失败案例', '', '| 目标训练频次 | 用户数 | 基线 HR@10 | 选定方案 HR@10 |', '|---|---:|---:|---:|']
    for key in ['cold_0', 'tail_1_5', 'mid_6_50', 'head_gt50']:
        row = result['frequency_and_context_groups'][key]
        def pct(value):
            return f'{value:.4%}' if value is not None else 'N/A'
        lines.append(f"| {key} | {row['users']:,} | {pct(row['baseline']['HR@10'])} | {pct(row['selected']['HR@10'])} |")
    lines += ['', '本次总体收益主要来自高频目标；1–5 次训练频次桶仍未命中，中频桶也没有同步改善。冷启动桶仅有极少命中，不能据此声称解决了冷启动或长尾问题。', '',
              '![Frequency buckets](figures/frequency_buckets.png)', '',
              '失败分析还覆盖历史长短和内容是否缺失，见 `metrics.json`。`failure_cases.json` 按固定顺序各取最多 5 个改善、退化、冷启动未命中和共同未命中样例。',
              '匿名数据只有 ID 和向量；这些样例支持频次/覆盖率分析，不支持凭空解释广告主题或用户偏好。', '',
              '![Validation curves](figures/validation_curves.png)', '',
              '## 第六天：Faiss 精度—延迟—索引体积', '',
              '同一 seed=2026 选定模型、同一候选向量、同一历史过滤；原始内积，没有改成余弦。',
              '原始内积 HNSW 未达到验证召回门槛，失败配置与结果保留在 `ann_initial_failure.json`。',
              '后续仅根据验证查询改用等价的 MIPS→L2 升维构图：广告变成 `(v, sqrt(R²-||v||²))`，查询变成 `(q, 0)`；',
              '`R²` 为广告最大平方范数，距离满足 `D²=||q||²+R²-2q·v`，因此精确排序仍是原始内积。',
              '返回的距离转换回内积分数。它与直接归一化成余弦不同；相关单元测试、独立打分和原 Torch 精确结果交叉检查均保留。',
              '索引参数仅根据验证查询的 ANNRecall@10 门槛和 p95 延迟选取，然后固定到测试查询。', '',
              '| 验证 efSearch | ANN Recall@10 | 单查询 p95 (ms) |', '|---:|---:|---:|']
    for row in benchmark['validation_configurations']:
        lines.append(f"| {row['efSearch']} | {row['ANNRecall@10']:.4%} | {row['single_query_p95_ms']:.4f} |")
    flat, ann = result['ann']['flat_test_latency'], result['ann']['hnsw_test_latency']
    lines += ['', f"固定 efSearch={result['ann']['selected_ef']} 后，在 {benchmark['test']['queries']} 个测试查询上：", '',
              f"- ANN Recall@10：{benchmark['test']['ANNRecall@10']:.4%}（相对精确检索的 ID 重合率）。",
              f"- 单线程 warmed p95：FlatIP {flat['single_query_p95_ms']:.4f} ms，HNSW {ann['single_query_p95_ms']:.4f} ms，约 {result['ann']['cpu_single_query_p95_speedup']:.2f} 倍。",
              f"- 8 线程批量吞吐：FlatIP {flat['batch_qps_8_threads']:.1f} QPS，HNSW {ann['batch_qps_8_threads']:.1f} QPS。",
              f"- 持久化索引体积：FlatIP {benchmark['build']['flat']['serialized_index_bytes']/1024**2:.1f} MiB，HNSW {benchmark['build']['hnsw']['serialized_index_bytes']/1024**2:.1f} MiB。", '',
              '![ANN trade-off](figures/ann_tradeoff.png)', '',
              '计时包含 CPU 检索和 Python 历史过滤，不含编码器、网络、进程启动与模型/索引加载。',
              '单查询使用 1 个 CPU 线程并预热；批量吞吐是另一个 8 线程负载。持久化体积不是峰值 RSS。机器是共享工作站，不代表生产 SLA。',
              'ANN Recall 与真实点击 HR 是两个指标；测试查询子集的点击指标也保存在原始 benchmark JSON 中，不与全体 10,000 用户的模型指标混用。', '',
              '## 第七天：复现和面试材料', '',
              '- [项目说明与运行命令](../../WEEK_PROJECT.md)',
              '- [面试讲稿与简历写法](INTERVIEW.md)',
              '- `metrics.json`：所有种子、配对差值、分桶指标。',
              '- `failure_cases.json`：公开匿名 ID 的成功/失败案例。',
              '- `environment.json`、`code_manifest.json`：实际环境、源文件摘要。', '',
              '### 范围与限制', '',
              '- 本地逐用户首次出现广告点击留出，排除重复广告点击，不等价于官方私有测试标签。',
              '- 用户/广告静态特征来自公开快照，不保证跨用户的全局时间隔离。',
              '- 仅使用预提取的 81 号文本向量；未实现文图融合或语义 ID 自回归生成。',
              '- 冷启动是否有效以分桶为准；内容存在不等于冷启动问题已解决。',
              '- 不声称参与比赛获奖、线上 CTR/CVR 提升、生产流量验证。', '']
    (reports / 'README.md').write_text('\n'.join(lines))
    write_interview(reports, result, benchmark)


def write_interview(reports, result, benchmark):
    a, b = result['aggregates']['baseline'], result['aggregates']['selected']
    text = f'''# 面试讲稿与简历材料

## 项目定位

基于腾讯广告公开日志的序列召回与向量检索项目。核心是下一次新广告点击预测：将用户历史编码为向量，从固定广告目录检索 Top-10。

## 约两分钟的介绍

我基于 TencentGR-1M 做了一个可复现的广告序列召回项目。起点是开源 SASRec 风格基线。我先修复原始 ID、内容向量连接、候选字段和测试输入的问题，建立逐用户严格时间截断的留出评估，避免目标广告提前出现在历史里。

随后我发现原基线把下一次曝光和点击都当作序列预测正样本，所以围绕训练目标与点击任务是否对齐做了消融。我比较了曝光降权、只监督点击、多负样本 BCE、InfoNCE，以及历史行为 Embedding，并且只依据验证集选方案。选择的是 `{result['development_selection']['selected_variant']}`。

我把数据扩大到 {result['training_users']:,} 名用户和 {result['candidate_items']:,} 个候选，排除之前查看过测试结果的用户，固定 10,000 名验证/测试用户，用三个种子复核。测试 HR@10 从 {a['HR@10']['mean']:.4%} 到 {b['HR@10']['mean']:.4%}，NDCG@10 从 {a['NDCG@10']['mean']:.6f} 到 {b['NDCG@10']['mean']:.6f}，完整保留每个种子、置信区间和冷启动分桶。

最后我用同一模型的向量比较 Faiss 精确内积检索和 HNSW。原始内积图未过召回门槛，我保留失败结果，并根据验证查询采用保持内积排序的升维 L2 构图，再锁定 efSearch 测测试查询。测试 ANN Recall@10 为 {benchmark['test']['ANNRecall@10']:.4%}。检索速度与索引体积的取舍都有实测，GPU 编码与 CPU 索引耗时分开报告。

这个项目使用公开静态目录和本地代理任务，没有线上 A/B，也不能当作比赛排名。

## 简历示例（数字对应本次实验）

**腾讯广告序列召回与 Faiss 检索优化｜Python / PyTorch / Faiss**

- 基于 TencentGR-1M 构建数据预处理、时间留出、序列训练、精确评估及向量召回闭环；修复 OID/RID 类型不匹配导致内容特征未命中的问题，完成缺失行为与冷启动词表处理。
- 通过目标/损失/负样本数量消融，依据验证集选定点击对比学习方案；在 {result['training_users']:,} 用户、{result['candidate_items']:,} 候选的本地任务上，三个种子的测试 HR@10 为 {b['HR@10']['mean']:.4%}、NDCG@10 为 {b['NDCG@10']['mean']:.6f}，同时报告对照和不确定性。
- 构建持久化 HNSW 召回索引，在验证集约束 ANN Recall 后锁定参数，测试 ANN Recall@10 达 {benchmark['test']['ANNRecall@10']:.4%}；对比单线程 p95、批量吞吐与索引体积，并提供模型与索引一致性检查和单用户推断命令。

## 高频追问

1. **业务目标是什么，为什么做召回？** 给定历史预测下一次新广告点击，从百万级目录取 Top-10。它对应召回能力；CTR 概率校准、竞价和转化建模没有在本项目中实现。
2. **为什么不直接照着官方 eval 跑？** 公共数据没有官方未来点击答案，原 eval 只导出预测。我的指标来自明确标注的本地留出协议。
3. **OID、RID、Faiss ID 怎么对应？** OID 是原始广告 ID，RID 用于原数据索引；预处理再建立局部目录 ID。Faiss 只索引非填充向量，Faiss 行号加 1 得局部 ID，再还原 OID。不是任意再查一次 ID 数组。
4. **怎样防止泄漏？** 每个用户的训练止于验证目标之前，历史时间严格早于目标；目标必须从未在该历史出现；负采样、词表和热门计数只用训练前缀。扩大复核排除之前已查看测试结果的用户。公开特征快照和跨用户全局时间仍是限制。
5. **曝光为什么不是直接的负样本？** 曝光意味着广告进入了观察序列，并不等同于明确拒绝。原目标预测下一事件，把曝光也作序列正样本；随机广告才是 BCE 负样本。我通过对照衡量目标对齐，而不是把所有曝光强行标负。
6. **只训练点击为什么不一定好？** 曝光也有协同和上下文信息，点击更稀疏。开发实验里的点击 BCE 表现更差，说明不能仅凭直觉删掉监督。
7. **InfoNCE 做了什么？** 对每个训练点击，用当前历史状态给真实下一广告和 64 个随机候选打内积分数，按正样本类别做 softmax 交叉熵，温度为 1。它不是对全目录精确 softmax，也不是 CTR 概率。
8. **怎么知道不是负样本更多造成的？** 我加入点击 BCE-64 对照：相同点击位置、64 个负样本和采样策略，只改变损失形式。多负样本 BCE 的负项取均值，避免简单放大 64 倍梯度。
9. **负采样有什么问题？** 从训练可见广告均匀有放回采样，排除该用户训练前缀中的广告。这样能避免一部分假负例；仍可能采到用户潜在喜欢但未观察的广告，也未实现困难负样本挖掘。
10. **历史行为 Embedding 一定有用吗？** 没有。前期不稳定，InfoNCE 的行为分支也没有在验证集胜过不加行为的版本，最终采用验证选中的配置。行为类型仍用于监督目标选择。
11. **内容向量是谁训练的？** 81 号是公开预提取文本向量；我训练的是它的投影和推荐模型，没有训练 BERT，也没有使用原始图像。修复内容读取不代表已经证明内容带来增益。
12. **为什么需要冷启动分桶？** 总体效果可能由热门广告主导。未在训练前缀出现的广告 ID 使用 OOV=0，靠属性和内容参与编码；实际能否命中必须看 cold 分桶，不能只说模型“支持冷启动”。
13. **对照公平在哪里？** 相同目录、切分、初始共享参数、网络、学习率和 5 个 epoch，三种子配对。不同目标监督量、负样本数不同，所以我不声称等计算量；运行耗时也保存了。
14. **ANN Recall 和 HR 的区别？** ANN Recall 测近似索引能否复现精确 Top-10；HR 测是否包含真实留出点击。模型没把目标排进精确前十，单纯把索引做得更准也不能解决。
15. **为什么用内积，不归一化？** 训练打分是原始内积，归一化会改变模型排序。本项目没有把它悄悄替换成余弦。
16. **HNSW 怎么调，怎么处理历史过滤？** 原始 IP 图召回未过门槛；仅增加 efSearch 也收效有限。我根据验证查询改用 MIPS→L2 升维，广告附加 sqrt(R²-||v||²)，查询附加 0，精确排序仍是内积。M 和 efConstruction 固定，再验证扫描 efSearch，满足召回门槛后取较小 p95。检索先多取历史长度加 10 个结果，再过滤历史。
17. **延迟数字能代表线上吗？** 不能。基准是 warmed CPU 索引搜索和 Python 过滤，单查询一个线程；批量吞吐另用八线程。单用户演示另报 GPU 编码+传输、索引、总请求耗时；进程启动与加载不算在 warmed 请求里。
18. **下一步怎么做？** 优先做全局时间回放和特征可用性检查，扩大训练/评估并复核种子，再研究长尾与冷启动、内容消融、困难负样本和图像特征。每次依据验证集决策。

## 面试前自己实际操作

- 从 `prepare.clean_and_split` 跟踪一个用户到 `SequenceDataset`，手画输入和下一广告标签的错位。
- 在 `objectives.py` 写出 BCE 与 InfoNCE 的式子，解释梯度方向、权重和负样本数。
- 打开 `failure_cases.json`，各讲一个改善、退化和冷启动未命中案例；不虚构 ID 的语义。
- 跑一次单用户 `recommend`，确认输出原始 OID，并解释模型—索引 SHA 检查及各段计时。
'''
    (reports / 'INTERVIEW.md').write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/objective_study')
    parser.add_argument('--output', default='reports/objective_study')
    parser.add_argument('--baseline-root', default='artifacts/baseline')
    parser.add_argument('--config', default='configs/objective_study.json')
    report(parser.parse_args())
