"""Summarize matched seeds without hiding negative or uncertain results."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def paired_interval(values, seed=4102, samples=5000):
    """User bootstrap conditional on the fixed set of trained seed pairs."""
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(0, samples, 250):
        draws = rng.integers(len(values), size=(min(250, samples - len(means)), len(values)))
        means.extend(values[draws].mean(axis=1))
    return np.quantile(means, [.025, .975]).tolist()


def report(root, output, seeds):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results, predictions = {}, {}
    for variant in ['baseline', 'actions']:
        results[variant], predictions[variant] = [], []
        for seed in seeds:
            folder = root / f'{variant}_seed{seed}'
            result = json.loads((folder / 'result.json').read_text())
            config = json.loads((folder / 'config.json').read_text())
            if config['max_steps'] != 0:
                raise ValueError('Smoke runs must not appear in the experiment report.')
            results[variant].append(result)
            with np.load(folder / 'test_predictions.npz') as data:
                predictions[variant].append({key: data[key].copy() for key in data.files})
    for j, seed in enumerate(seeds):
        a, b = results['baseline'][j], results['actions'][j]
        if a['shared_initialization_sha256'] != b['shared_initialization_sha256']:
            raise AssertionError(f'Unmatched initialization for seed {seed}')
        ca = json.loads((root / f'baseline_seed{seed}/config.json').read_text())
        cb = json.loads((root / f'actions_seed{seed}/config.json').read_text())
        for key in ['cache', 'epochs', 'batch_size', 'eval_batch_size', 'candidate_chunk', 'hidden',
                    'blocks', 'heads', 'dropout', 'lr', 'max_steps', 'protocol', 'device_name']:
            if ca[key] != cb[key]:
                raise AssertionError(f'Unequal comparison setting: {key}')
    reference = predictions['baseline'][0]
    for variant in predictions.values():
        for prediction in variant:
            for key in ['rows', 'targets']:
                np.testing.assert_array_equal(reference[key], prediction[key])
    aggregates = {}
    for variant, runs in results.items():
        aggregates[variant] = {}
        for metric in ['HR@10', 'NDCG@10']:
            values = [run['test'][metric] for run in runs]
            aggregates[variant][metric] = dict(mean=float(np.mean(values)),
                                               sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                               per_seed=values)
    deltas = {}
    for metric, key in [('HR@10', 'hits'), ('NDCG@10', 'ndcg')]:
        differences = np.mean([b[key] - a[key] for a, b in
                               zip(predictions['baseline'], predictions['actions'])], axis=0)
        baseline = aggregates['baseline'][metric]['mean']
        deltas[metric] = dict(absolute=float(differences.mean()),
                             relative=float(differences.mean() / baseline) if baseline else None,
                             paired_user_bootstrap_95pct=paired_interval(differences))
    summary = dict(seeds=seeds, comparison='corrected array-backed SASRec baseline vs + observed-history action embedding',
                   evaluation_users=len(reference['rows']), candidate_items=results['baseline'][0]['candidates'],
                   training_users=results['baseline'][0]['training_users'], epochs=ca['epochs'],
                   matched_initialization_and_budget=True, test_used_for_checkpoint_selection=False,
                   aggregates=aggregates, deltas=deltas, runs=results,
                   popularity=results['baseline'][0]['popularity_test'],
                   uncertainty_note='Bootstrap resamples paired users after averaging these training seeds; '
                                    'it is conditional on these trained models, not a population interval over all training seeds.',
                   scope='Local first-appearance-click holdout on a sampled-user cohort; not an official leaderboard score.')
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    source_root = Path(__file__).resolve().parents[1]
    paths = sorted((source_root / 'taac_retrieval').glob('*.py')) + [source_root / 'model.py', source_root / 'dataset.py']
    manifest = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (output / 'code_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    metadata = json.loads((root / 'data/metadata.json').read_text())
    lines = ['# 前三天实验结果', '',
             f"从 {metadata['audit']['sampled_users']:,} 名随机用户中保留 {metadata['users']:,} 名；固定候选池 {summary['candidate_items']:,} 个广告。",
             f"验证、测试分别评估相同的 {summary['evaluation_users']:,} 名用户；两种模型使用相同配置和种子 {seeds}。",
             f"每个运行训练 {summary['epochs']} 个完整 epoch，按验证 NDCG@10（再按 HR@10）选 checkpoint，再评估一次测试集。", '',
             '| 模型 | 测试 HR@10（均值） | 测试 NDCG@10（均值） |', '|---|---:|---:|',
             f"| 训练前缀热门广告 | {summary['popularity']['HR@10']:.4%} | {summary['popularity']['NDCG@10']:.6f} |"]
    for variant, label in [('baseline', '修正后的 SASRec 基线'), ('actions', '+ 历史行为 Embedding')]:
        row = aggregates[variant]
        lines.append(f"| {label} | {row['HR@10']['mean']:.4%} | {row['NDCG@10']['mean']:.6f} |")
    lines += ['', '| 种子 | 模型 | 最优 epoch | 测试 HR@10 | 测试 NDCG@10 |', '|---|---|---:|---:|---:|']
    for j, seed in enumerate(seeds):
        for variant in ['baseline', 'actions']:
            row = results[variant][j]
            lines.append(f"| {seed} | {variant} | {row['best_epoch']} | {row['test']['HR@10']:.4%} | {row['test']['NDCG@10']:.6f} |")
    lines += ['', '## 改进是否成立', '']
    for metric, delta in deltas.items():
        low, high = delta['paired_user_bootstrap_95pct']
        interpretation = '区间跨过 0，当前证据不足以确认稳定改善。' if low <= 0 <= high else (
            '在本次固定模型和用户样本下观察到正向差异。' if low > 0 else '在本次固定模型和用户样本下观察到负向差异。')
        lines.append(f"- {metric}：行为模型减基线 = {delta['absolute']:+.6f}；配对用户 bootstrap 95% 区间 [{low:+.6f}, {high:+.6f}]。{interpretation}")
    lines += ['', f'区间在 {len(seeds)} 个固定训练种子上先求用户级平均，再重采样用户；不能据此声称覆盖任意训练种子的泛化区间。', '',
              '## 评估协议与边界', '',
              '- 每个用户预留最后两次此前从未出现过的广告点击；历史时间严格早于目标，训练截止于验证目标之前。',
              '- 这一限制主动排除已曝光目标，避免无法可靠关联曝光和点击时的目标泄漏；它属于本地新广告点击代理任务。',
              '- 候选为所选公开序列中全部不同广告，不按模型或测试得分调整，也没有向随机小池单独补目标。',
              '- 目录是公开静态目录，允许候选内容可见；按用户时间截断，不保证跨用户的全局时间隔离。',
              '- ID、广告属性词表和热门计数只从训练可见广告拟合；未见 ID / 属性映射到 0。',
              '- 曝光、点击、未知行为分别编码为 1、2、3；未知行为只作上下文，不作监督目标。',
              '- 只启用 81 号预提取文本向量。报告不是文图融合、语义 ID 生成或官方榜单结果。',
              '- 所有模型使用精确内积 Top-10，并过滤历史广告；这里没有混入 ANN 近似误差。',
              '- 测试集只在每个运行选好 checkpoint 后评估；后续模型调整应继续依据验证集。', '',
              '完整机器可读结果见 `metrics.json`；全量缓存检查见 `data_audit.json`。',
              f'训练日志、逐用户预测、配置与 checkpoint 位于 `{root}`。', '']
    (output / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(aggregates=aggregates, deltas=deltas), indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/baseline')
    parser.add_argument('--output', default='reports/baseline')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027])
    args = parser.parse_args()
    report(args.root, args.output, args.seeds)
