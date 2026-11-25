"""Bounded, training-free zero-click-tail retrieval and independent confirmation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch.nn import functional as F

from run_click_support import REPO, array_sha, freeze, read, sha

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def summarize(main, fused, targets, counts, clicks):
    before = (main == targets[:, None]).any(1)
    after = (fused == targets[:, None]).any(1)
    freq, support = counts[targets], clicks[targets]
    masks = dict(all=np.ones(len(targets), bool), cold_0=freq == 0,
                 tail_1_5=(freq >= 1) & (freq <= 5),
                 zero_click_tail=(freq >= 1) & (freq <= 5) & (support == 0),
                 clicked_tail=(freq >= 1) & (freq <= 5) & (support > 0),
                 mid_6_50=(freq >= 6) & (freq <= 50), head_gt50=freq > 50)
    return {name: dict(users=int(mask.sum()), baseline_hits=int(before[mask].sum()),
                      hits=int(after[mask].sum()), gained=int((~before & after & mask).sum()),
                      lost=int((before & ~after & mask).sum()),
                      baseline_recall=float(before[mask].mean()), recall=float(after[mask].mean()))
            for name, mask in masks.items()}


def worker(args):
    from taac_retrieval.click_support import evaluate_click_support, load_click_support
    from taac_retrieval.tail_channel import (eligible_tail_ids, encode_tail_vectors,
                                           fuse_tail, retrieve_tail)
    from taac_retrieval.train import seed_everything
    root = Path(args.root).resolve()
    plan, identity = read(root / 'plan.json'), read(root / 'identity.json')
    output = root / f'{args.phase}/seed{args.seed}'
    if output.with_suffix('.json').exists():
        raise FileExistsError('Worker outputs are immutable.')
    output.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    rows = np.load(root / f'{args.phase}.npy', allow_pickle=False)
    checkpoint = identity['checkpoints'][str(args.seed)]
    assert sha(checkpoint['path']) == checkpoint['sha256']
    model, data, saved = load_click_support(checkpoint['path'], identity['cache'],
        identity['counts_file'], args.device, eval_indices=rows)
    model.requires_grad_(False)
    initial_hash = parameter_hash(model)
    if args.phase == 'development':
        reference = identity['references'][str(args.seed)]
        assert sha(reference['path']) == reference['sha256']
        with np.load(reference['path'], allow_pickle=False) as a:
            baseline = {key: a[key] for key in ['predictions', 'queries', 'seen', 'rows', 'targets']}
        np.testing.assert_array_equal(baseline['rows'], rows)
        np.testing.assert_array_equal(baseline['targets'], data.items[rows, data.val_pos[rows]])
    else:
        lock = read(root / 'confirmation_lock.json')
        assert lock['cohort_sha256'] == array_sha(rows)
        assert lock['checkpoints'] == identity['checkpoints']
        _, baseline = evaluate_click_support(model, data, args.device, 128, 32768, k=500)
    counts = np.asarray(data.train_counts)
    clicks = np.load(identity['counts_file'], allow_pickle=False)
    ids = eligible_tail_ids(counts, clicks)
    raw_vectors = encode_tail_vectors(model, ids, 'raw_ip')
    norms = raw_vectors.norm(dim=-1).cpu().numpy()
    modes = plan['score_modes'] if args.phase == 'development' else [lock['score_mode']]
    quotas = plan['quotas'] if args.phase == 'development' else [lock['quota']]
    arrays = {key: baseline[key] for key in ['predictions', 'queries', 'seen', 'rows', 'targets']}
    combinations, timing = {}, {}
    for mode in modes:
        vectors = raw_vectors if mode == 'raw_ip' else F.normalize(raw_vectors, dim=-1)
        proposals, elapsed = [], []
        for start in range(0, len(rows), 128):
            stop = min(start + 128, len(rows))
            queries = torch.as_tensor(baseline['queries'][start:stop], device=args.device)
            if mode == 'cosine':
                queries = F.normalize(queries, dim=-1)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            began = time.monotonic()
            proposals.append(retrieve_tail(queries, vectors, ids, baseline['seen'][start:stop],
                baseline['predictions'][start:stop], k=plan['proposal_k']))
            elapsed.append(time.monotonic() - began)
        proposed = np.concatenate(proposals)
        assert np.isin(proposed[proposed > 0], ids).all()
        arrays[f'proposals_{mode}'] = proposed.astype(np.int32)
        timing[mode] = dict(batch_size=128, batches=len(elapsed),
            batch_search_seconds=sum(elapsed),
            amortized_search_ms_per_query=1000 * sum(elapsed) / len(rows),
            note='Exact GPU subset search including exclusions and CPU output copy; not single-query serving p95.')
        for quota in quotas:
            key = f'{mode}_q{quota}'
            began = time.monotonic()
            fused, applied = fuse_tail(baseline['predictions'], proposed, counts,
                                      quota, plan['protected_top'])
            np.testing.assert_array_equal(fused[:, :10], baseline['predictions'][:, :10])
            arrays[key] = fused.astype(np.int32)
            combinations[key] = dict(score_mode=mode, quota=quota,
                groups=summarize(baseline['predictions'], fused, baseline['targets'], counts, clicks),
                mean_replacements=float(applied.mean()), fusion_seconds=time.monotonic() - began)
    assert parameter_hash(model) == initial_hash
    result = dict(seed=args.seed, phase=args.phase, checkpoint=checkpoint, model_parameters_unchanged=True,
        shared_parameter_sha256=initial_hash, auxiliary_pool_items=len(ids),
        vector_norm_quantiles=np.quantile(norms, [0, .1, .5, .9, 1]).tolist(),
        vector_zero_norm_count=int((norms == 0).sum()), combinations=combinations, timing=timing,
        eligible_ids_sha256=array_sha(ids), source_manifest=read(root / 'source_manifest.json'))
    np.savez_compressed(output.with_suffix('.npz'), **arrays)
    freeze(output.with_suffix('.json'), result)
    print(json.dumps(dict(event='worker_finished', phase=args.phase, seed=args.seed)), flush=True)


class Study:
    def __init__(self, args):
        self.args, self.root = args, Path(args.root).resolve()
        self.reports, self.plan = Path(args.report_dir).resolve(), read(args.config)
        self.prior = REPO / 'artifacts/click_support_v1'
        self.prior_identity = read(self.prior / 'cohorts/identity.json')
        if (self.plan['score_modes'] != ['raw_ip', 'cosine'] or self.plan['quotas'] != [1, 5]
                or self.plan['budget'] != 500 or self.plan['seeds'] != [2026, 2027, 2028]
                or self.plan['protected_top'] != 10 or self.plan['proposal_k'] != 25
                or self.plan['guards'] != dict(overall_retention=.99, minimum_positive_seeds=2)
                or self.plan['eligibility'] != dict(min_events=1, max_events=5, training_clicks=0)):
            raise ValueError('This runner implements only the four registered combinations.')

    def prepare(self):
        if any((REPO / f'artifacts/{name}/confirmation').exists() for name in
               ['click_support_v1', 'candidate_regularization_v1']):
            raise ValueError('The reused confirmation cohort has already been opened.')
        freeze(self.root / 'plan.json', self.plan)
        original = Path('/home/jyk/project/baseline_2025/artifacts')
        identity = dict(self.prior_identity, checkpoints={}, references={})
        identity['counts_file'] = identity['training_click_counts_file']
        for seed in self.plan['seeds']:
            path = original / f'objective_study/confirmation/selected_seed{seed}/best.pt'
            reference = original / f'cold_start/diagnostics/click_seed{seed}.npz'
            metadata = read(reference.with_suffix('.json'))
            expected = identity['initial_checkpoints_sha256'][str(seed)]
            assert sha(path) == metadata['checkpoint_sha256'] == expected
            identity['checkpoints'][str(seed)] = dict(path=str(path), sha256=expected)
            identity['references'][str(seed)] = dict(path=str(reference), sha256=sha(reference))
        freeze(self.root / 'identity.json', identity)
        for name in ['development', 'confirmation']:
            values = np.load(self.prior / f'cohorts/{name}.npy', allow_pickle=False)
            path = self.root / f'{name}.npy'
            if path.exists():
                np.testing.assert_array_equal(np.load(path, allow_pickle=False), values)
            else:
                np.save(path, values)
        files = [Path(__file__), Path(self.args.config).resolve(), REPO / 'model.py',
                 REPO / 'scripts/run_click_support.py',
                 *[REPO / 'taac_retrieval' / name for name in ['tail_channel.py', 'click_support.py',
                     'model.py', 'data.py', 'metrics.py', 'coverage.py', 'train.py', 'objectives.py',
                     'evaluate_checkpoint.py']]]
        freeze(self.root / 'source_manifest.json', {str(p.relative_to(REPO)): sha(p) for p in files})
        self.verify()

    def verify(self):
        assert read(self.root / 'plan.json') == self.plan
        for path, expected in read(self.root / 'source_manifest.json').items():
            assert sha(REPO / path) == expected
        identity = read(self.root / 'identity.json')
        assert sha(identity['counts_file']) == identity['training_click_counts_sha256']
        assert sha(Path(identity['cache']) / 'metadata.json') == identity['cache_metadata_sha256']
        for name in ['development', 'confirmation']:
            assert array_sha(np.load(self.root / f'{name}.npy', allow_pickle=False)) == identity[name + '_indices_sha256']

    def evaluate(self, phase):
        self.verify()
        devices = self.args.gpus.split(',')
        def run(pair):
            index, seed = pair
            if (self.root / f'{phase}/seed{seed}.json').exists():
                return
            log = self.root / f'{phase}/seed{seed}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as handle:
                subprocess.run([sys.executable, str(Path(__file__)), '--worker', '--root', str(self.root),
                    '--phase', phase, '--seed', str(seed), '--device', 'cuda:' + devices[index % len(devices)]],
                    cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)
        with ThreadPoolExecutor(max_workers=min(3, len(devices))) as pool:
            list(pool.map(run, enumerate(self.plan['seeds'])))
        return [read(self.root / f'{phase}/seed{seed}.json') for seed in self.plan['seeds']]

    def aggregate(self, rows, key):
        combinations = [row['combinations'][key] for row in rows]
        groups = {group: {field: float(np.mean([r['groups'][group][field] for r in combinations]))
                          for field in ['baseline_recall', 'recall', 'baseline_hits', 'hits', 'gained', 'lost']}
                  for group in combinations[0]['groups']}
        reasons = []
        if groups['all']['recall'] < self.plan['guards']['overall_retention'] * groups['all']['baseline_recall']:
            reasons.append('overall_retention')
        for group in ['cold_0', 'clicked_tail']:
            if groups[group]['recall'] < groups[group]['baseline_recall']:
                reasons.append(group + '_decreased')
        positive = sum(r['groups']['zero_click_tail']['hits'] > r['groups']['zero_click_tail']['baseline_hits'] for r in combinations)
        if groups['zero_click_tail']['recall'] <= groups['zero_click_tail']['baseline_recall']:
            reasons.append('no_zero_click_tail_gain')
        if positive < self.plan['guards']['minimum_positive_seeds']:
            reasons.append('insufficient_positive_seed_models')
        return dict(key=key, score_mode=combinations[0]['score_mode'], quota=combinations[0]['quota'],
                    groups=groups, passed=not reasons, reasons=reasons, positive_seed_models=positive)

    def intervals(self, key):
        values, targets = [], None
        identity = read(self.root / 'identity.json')
        counts = np.load(Path(identity['cache']) / 'train_counts.npy', allow_pickle=False)
        clicks = np.load(identity['counts_file'], allow_pickle=False)
        for seed in self.plan['seeds']:
            with np.load(self.root / f'confirmation/seed{seed}.npz', allow_pickle=False) as a:
                if targets is not None:
                    np.testing.assert_array_equal(targets, a['targets'])
                targets = a['targets']
                values.append((a[key] == targets[:, None]).any(1).astype(float)
                              - (a['predictions'] == targets[:, None]).any(1))
        delta = np.stack(values)
        freq = counts[targets]
        masks = dict(all=np.ones(len(targets), bool), cold_0=freq == 0,
                     tail_1_5=(freq >= 1) & (freq <= 5),
                     zero_click_tail=(freq >= 1) & (freq <= 5) & (clicks[targets] == 0))
        rng, result = np.random.default_rng(self.plan['bootstrap']['seed']), {}
        for group, mask in masks.items():
            selected = delta[:, mask]
            sampled = [float(selected[rng.integers(0, 3, 3)][:, rng.integers(0, mask.sum(), mask.sum())].mean())
                       for _ in range(self.plan['bootstrap']['replicates'])]
            result[group] = dict(mean=float(selected.mean()), ci95=np.quantile(sampled, [.025, .975]).tolist(),
                                 unique_users_with_gain=int((selected > 0).any(0).sum()), users=int(mask.sum()))
        return result

    def report(self, status, development, selected, confirmation=None, intervals=None):
        self.reports.mkdir(parents=True, exist_ok=True)
        result = dict(status=status, plan=self.plan, development=development, selected=selected,
                      confirmation=confirmation, paired_intervals=intervals,
                      confirmation_targets_evaluated=confirmation is not None,
                      original_main_replaced=False, source_manifest=read(self.root / 'source_manifest.json'))
        freeze(self.reports / 'metrics.json', result)
        lines = ['# 原序列召回＋零点击长尾补充通道', '', f'状态：`{status}`。没有训练或替换主模型。', '',
                 '三份原主模型分别复用自己的用户查询；辅助候选去掉显式广告ID，保留原属性与81。',
                 '先排除历史和原500候选，再从专用长尾目录检索；保护原Top-10及已有冷尾候选，总预算仍500。', '',
                 '## 三模型开发复核', '',
                 '| 评分/配额 | 整体Recall500 | 冷启动Recall500 | 长尾Recall500 | 零点击长尾Recall500 | 合格 |',
                 '|---|---:|---:|---:|---:|---|']
        for row in development:
            g = row['groups']
            lines.append(f'| {row["key"]} | {g["all"]["recall"]*100:.4f}% | {g["cold_0"]["recall"]*100:.4f}% | {g["tail_1_5"]["recall"]*100:.4f}% | {g["zero_click_tail"]["recall"]*100:.4f}% | {row["passed"]} |')
        if confirmation is not None:
            lines += ['', '## 一次新留出行为确认', '', f'冻结方案：`{selected["key"]}`；点估计门槛通过：{confirmation["passed"]}。', '',
                      '| 分组 | 主通道Recall500 | 融合Recall500 | 平均新增/丢失目标数 |', '|---|---:|---:|---:|']
            for group, g in confirmation['groups'].items():
                lines.append(f'| {group} | {g["baseline_recall"]*100:.4f}% | {g["recall"]*100:.4f}% | {g["gained"]:.2f}/{g["lost"]:.2f} |')
            lines += ['', '配对区间、逐模型结果和不同用户新增规模见metrics.json及本地产物；区间涉及三个固定模型和当前用户，不保证线上效果。']
        lines += ['', '## 边界', '', '- 主Top-10和已有冷尾候选保持来自保护规则，不能称为这些指标的学习收益。',
                  '- 只比较已登记raw_ip/cosine、配额1/5；确认后不改选。点估计门槛通过不等于统计稳定收益。',
                  '- 仍是公共静态目录、逐用户截断，新确认用户的训练历史参加训练。',
                  '- 辅助使用学得的属性/81表示，包含原高基数字段，不能称为纯文本语义模型。',
                  f'- 本地产物：`{self.root}`。模型、完整候选和逐用户结果不进入Git。', '']
        (self.reports / 'README.md').write_text('\n'.join(lines))
        print(json.dumps(dict(event='finished', status=status, report=str(self.reports / 'README.md'))), flush=True)

    def run(self):
        self.prepare()
        dev = self.evaluate('development')
        rows = [self.aggregate(dev, f'{mode}_q{quota}') for mode in self.plan['score_modes'] for quota in self.plan['quotas']]
        eligible = [r for r in rows if r['passed']]
        freeze(self.root / 'development_selection.json', rows)
        if not eligible:
            self.report('stopped_after_development', rows, None)
            return
        selected = max(eligible, key=lambda r: (r['groups']['zero_click_tail']['recall'],
            r['groups']['all']['recall'], -r['quota'], r['score_mode'] == 'raw_ip'))
        lock = dict(score_mode=selected['score_mode'], quota=selected['quota'],
                    cohort_sha256=array_sha(np.load(self.root / 'confirmation.npy', allow_pickle=False)),
                    checkpoints=read(self.root / 'identity.json')['checkpoints'])
        freeze(self.root / 'confirmation_lock.json', lock)
        print(json.dumps(dict(event='locked', **lock)), flush=True)
        confirmation = self.aggregate(self.evaluate('confirmation'), selected['key'])
        self.report('completed_confirmation', rows, selected, confirmation, self.intervals(selected['key']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/supplemental_tail_v1')
    parser.add_argument('--report-dir', default='reports/supplemental_tail_study')
    parser.add_argument('--config', default=str(REPO / 'configs/supplemental_tail.json'))
    parser.add_argument('--gpus', default='0,2,3')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--phase', choices=['development', 'confirmation'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    worker(args) if args.worker else Study(args).run()
