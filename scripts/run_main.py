"""Run the selected retrieval model from preparation to an offline Top-10 demo.

Paths are resolved relative to this repository, including when invoked elsewhere.
The default run writes only artifacts/showcase; historical reports are preserved.
Use --dry-run to inspect any stage without reading data or starting computation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
STAGES = ('prepare', 'train', 'evaluate', 'export', 'index', 'recommend')


def resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config):
    prep, model, ann = (config[key] for key in ('preparation', 'model', 'faiss'))
    if model['variant'] != 'baseline' or model['objective'] != 'click_infonce':
        raise ValueError('The showcase config must use the selected click-only InfoNCE model.')
    if model['num_negatives'] != 64 or model['blocks'] != 1 or model['hidden'] != 32 or model['heads'] != 1:
        raise ValueError('The selected model has 64 negatives, one block, hidden=32 and one head.')
    if model['seed'] != 2026 or model['epochs'] != 5:
        raise ValueError('The showcase reproduces seed 2026 for five complete epochs.')
    expected = dict(reference_max_users=100000, reference_eval_users=2000, max_users=300000,
                    eval_users=10000, seed=2026, max_history=100, min_history=5, cohort_seed=3107)
    if prep != expected:
        raise ValueError('Preparation must preserve the historical sampling and fresh-cohort protocol.')
    if ann['hnsw_geometry'] != 'mips_l2' or ann['efSearch'] != [2048]:
        raise ValueError('The selected index uses mips_l2 and fixed efSearch=2048.')
    if ann['metric'] != 'inner_product_without_normalization' or ann['minimum_validation_ann_recall'] != 0.98:
        raise ValueError('Use raw inner-product ranking and the unchanged 98% validation ANN recall gate.')


class Workflow:
    def __init__(self, args):
        self.args = args
        self.config_path = resolve(args.config)
        self.config = read_json(self.config_path)
        validate_config(self.config)
        self.root = resolve(args.root)
        self.data_dir = resolve(args.data_dir)
        self.cache = self.root / 'data'
        self.reference_cache = self.root / 'reference_data'
        self.cohort = self.root / 'cohort.npy'
        self.model = self.root / 'model'
        self.checkpoint = self.model / 'best.pt'
        self.export_dir = self.root / 'export'
        self.ann = self.root / 'ann'
        self.reports = self.root / 'reports'
        if not args.dry_run:
            self.reports.mkdir(parents=True, exist_ok=True)
            snapshot = self.root / 'main_config.json'
            if snapshot.exists() and read_json(snapshot) != self.config:
                raise ValueError('This output directory belongs to a different config. Use a new --root.')
            if not snapshot.exists():
                snapshot.write_text(json.dumps(self.config, indent=2) + '\n')

    def run(self, arguments, stdout_path=None):
        command = [sys.executable, *map(str, arguments)]
        display = shlex.join(command)
        if stdout_path is not None:
            display += ' > ' + shlex.quote(str(stdout_path))
        print(display, flush=True)
        if self.args.dry_run:
            return
        with (self.root / 'commands.jsonl').open('a') as log:
            log.write(json.dumps(dict(stage=self.stage, command=command,
                                      stdout=str(stdout_path) if stdout_path else None)) + '\n')
        if stdout_path is None:
            subprocess.run(command, cwd=REPO, check=True)
        else:
            with Path(stdout_path).open('w') as handle:
                subprocess.run(command, cwd=REPO, check=True, stdout=handle)
            print(Path(stdout_path).read_text(), end='', flush=True)

    def require_prepared(self):
        if self.args.dry_run:
            return
        if not (self.cache / 'metadata.json').exists() or not self.cohort.exists():
            raise FileNotFoundError('Run --stage prepare first with this --root.')
        prep = self.config['preparation']
        actual = read_json(self.cache / 'metadata.json')['preparation_config']
        for key in ('max_users', 'eval_users', 'seed', 'max_history', 'min_history'):
            if actual[key] != prep[key]:
                raise ValueError(f'Prepared cache has a different {key}. Use a new --root.')
        cohort = read_json(self.cohort.with_suffix('.json'))
        if (cohort['users'] != prep['eval_users'] or cohort['seed'] != prep['cohort_seed']
                or cohort['excluded_previous_test_users'] != prep['reference_eval_users']
                or cohort['overlap_with_previous_test_users'] != 0):
            raise ValueError('The frozen evaluation cohort does not match the showcase protocol.')

    def training_arguments(self):
        return dict(cache=str(self.cache), output=str(self.model), device=self.args.device,
                    **self.config['model'], skip_test=True, eval_indices=str(self.cohort))

    def require_model(self):
        if self.args.dry_run:
            return
        self.require_prepared()
        if not self.checkpoint.exists() or not (self.model / 'result.json').exists():
            raise FileNotFoundError('Run --stage train first with this --root.')
        actual = read_json(self.model / 'config.json')
        for key, value in self.training_arguments().items():
            if key != 'device' and actual.get(key) != value:
                raise ValueError(f'Existing model has a different {key}. Use a new --root.')
        if actual['protocol'] != read_json(self.cache / 'metadata.json'):
            raise ValueError('The trained model belongs to a different prepared cache.')
        cohort = read_json(self.cohort.with_suffix('.json'))
        if actual['evaluation_cohort_sha256'] != cohort['indices_sha256']:
            raise ValueError('The trained model belongs to a different evaluation cohort.')

    def prepare(self):
        prep = self.config['preparation']
        # Recreate the early 100k sample to exclude its 2k previously inspected
        # users by raw RID, just as in the historical 300k confirmation study.
        for cache, users, evaluation in ((self.reference_cache, prep['reference_max_users'], prep['reference_eval_users']),
                                         (self.cache, prep['max_users'], prep['eval_users'])):
            self.run(['-m', 'taac_retrieval.prepare', '--data-dir', self.data_dir, '--output', cache,
                      '--max-users', users, '--eval-users', evaluation, '--seed', prep['seed'],
                      '--max-history', prep['max_history'], '--min-history', prep['min_history']])
        self.run(['-m', 'taac_retrieval.cohorts', '--cache', self.cache,
                  '--previous-cache', self.reference_cache, '--output', self.cohort,
                  '--users', prep['eval_users'], '--seed', prep['cohort_seed']])
        self.run(['-m', 'taac_retrieval.audit', '--cache', self.cache,
                  '--output', self.reports / 'data_audit.json'])

    def train(self):
        self.require_prepared()
        if not self.args.dry_run and (self.model / 'result.json').exists():
            self.require_model()
            print(f'Reusing completed model: {self.checkpoint}', flush=True)
            return
        if not self.args.dry_run and (self.model / 'config.json').exists():
            raise FileExistsError('An incomplete training run exists. Use a new --root; training is not resumable.')
        command = ['-m', 'taac_retrieval.train']
        for key, value in self.training_arguments().items():
            command.append('--' + key.replace('_', '-'))
            if not isinstance(value, bool):
                command.append(value)
        self.run(command)

    def evaluate(self):
        self.require_model()
        output = self.reports / 'evaluation'
        if not self.args.dry_run and output.with_suffix('.json').exists():
            actual = read_json(output.with_suffix('.json'))
            if (actual['checkpoint_sha256'] != sha256(self.checkpoint) or actual['split'] != 'test'
                    or actual['evaluation_cohort_sha256'] != read_json(self.cohort.with_suffix('.json'))['indices_sha256']):
                raise ValueError('Existing evaluation belongs to another model/cohort.')
            print(f'Reusing completed test evaluation: {output.with_suffix(".json")}', flush=True)
            return
        self.run(['-m', 'taac_retrieval.evaluate_checkpoint', '--checkpoint', self.checkpoint,
                  '--cache', self.cache, '--output', output, '--split', 'test', '--device', self.args.device])

    def export(self):
        self.require_model()
        self.run(['-m', 'taac_retrieval.export', '--checkpoint', self.checkpoint,
                  '--cache', self.cache, '--output', self.export_dir, '--device', self.args.device])

    def index(self):
        self.require_model()
        if not self.args.dry_run:
            metadata_path = self.export_dir / 'metadata.json'
            if not metadata_path.exists():
                raise FileNotFoundError('Run --stage export first with this --root.')
            exported = read_json(metadata_path)
            if exported['checkpoint_sha256'] != sha256(self.checkpoint):
                raise ValueError('The vector export belongs to another checkpoint.')
            if (self.ann / 'benchmark.json').exists():
                completed = read_json(self.ann / 'benchmark.json')
                if completed['configuration'] != self.config['faiss'] or completed['export'] != exported:
                    raise ValueError('Existing ANN benchmark belongs to another config/export.')
                print(f'Reusing completed ANN benchmark: {self.ann / "benchmark.json"}', flush=True)
                return
        self.run(['-m', 'taac_retrieval.benchmark_ann', '--export-dir', self.export_dir,
                  '--output', self.ann, '--config', self.config_path])

    def recommend(self):
        self.require_model()
        if not self.args.dry_run and not (self.ann / 'benchmark.json').exists():
            raise FileNotFoundError('Run --stage index first with this --root.')
        self.run(['-m', 'taac_retrieval.recommend', '--checkpoint', self.checkpoint, '--cache', self.cache,
                  '--export-dir', self.export_dir, '--index-dir', self.ann, '--device', self.args.device,
                  '--split', self.args.split, '--index', self.args.index],
                 stdout_path=self.reports / 'recommend_example.json')

    def write_report(self):
        evaluation = self.reports / 'evaluation.json'
        if self.args.dry_run or not evaluation.exists():
            return
        result = read_json(evaluation)
        metadata = read_json(self.cache / 'metadata.json')
        lines = ['# Showcase reproduction', '',
                 'Single seed 2026, click-only InfoNCE with 64 negatives; this is not a three-seed mean.', '',
                 f"Sampled users: {self.config['preparation']['max_users']:,}; retained: {metadata['users']:,}.",
                 f"Catalog items: {metadata['catalog_items']:,}; frozen evaluation users: {self.config['preparation']['eval_users']:,}.",
                 f"Validation-selected epoch: {result['selected_epoch']}; checkpoint SHA256: `{result['checkpoint_sha256']}`.", '',
                 '| Model | Test HR@10 | Test NDCG@10 |', '| --- | ---: | ---: |']
        for name, scores in [('Selected model', result['metrics']), ('Popularity', result['popularity'])]:
            lines.append(f"| {name} | {scores['HR@10']:.6f} | {scores['NDCG@10']:.6f} |")
        if (self.ann / 'benchmark.json').exists():
            ann = read_json(self.ann / 'benchmark.json')
            lines += ['', 'HNSW uses the MIPS-to-L2 lift with raw inner-product ranking.',
                      f"Fixed efSearch: {ann['selected']['efSearch']}; test ANNRecall@10: {ann['test']['ANNRecall@10']:.6f}.",
                      'ANNRecall measures overlap with exact search, not user-click recall.',
                      'Latency in ann/benchmark.json is warmed CPU index timing, not an online-service SLA.']
        lines += ['', 'Full artifacts: model/, export/, ann/, reports/evaluation.json and reports/data_audit.json.',
                  'Run scripts/run_objective_study.py separately to reproduce controlled multi-seed evidence.', '']
        (self.reports / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=[*STAGES, 'all'], default='all',
                        help='Stages depend on earlier artifacts; all runs the full sequence')
    parser.add_argument('--config', default='configs/main.json')
    parser.add_argument('--data-dir', default='TencentGR-1M')
    parser.add_argument('--root', default='artifacts/showcase')
    parser.add_argument('--device', default='cuda:0', help='PyTorch device, for example cpu or cuda:0')
    parser.add_argument('--split', choices=['val', 'test'], default='val', help='Offline demo split only')
    parser.add_argument('--index', type=int, default=0, help='Offline demo position in the frozen cohort; Top-10 output')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without modifying files or running stages')
    args = parser.parse_args()
    if args.index < 0:
        parser.error('--index must be nonnegative')
    workflow = Workflow(args)
    if args.index >= workflow.config['preparation']['eval_users']:
        parser.error('--index must be smaller than the frozen cohort size')
    for stage in STAGES if args.stage == 'all' else (args.stage,):
        workflow.stage = stage
        getattr(workflow, stage)()
    workflow.write_report()


if __name__ == '__main__':
    main()
