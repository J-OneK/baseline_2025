"""Compare one, two and four layers with verified nested initial parameters."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from taac_retrieval.initialization import prepare_depth_initialization
from taac_retrieval.study import run_training_jobs


MATCHED_FIELDS = ['variant', 'seed', 'epochs', 'batch_size', 'eval_batch_size', 'candidate_chunk',
                  'workers', 'hidden', 'heads', 'dropout', 'lr', 'max_steps', 'objective',
                  'num_negatives', 'exposure_weight', 'temperature', 'skip_test']


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def freeze_plan(root, specification):
    reference_root = (REPO / specification['reference_root']).resolve()
    jobs, references, initializations = [], [], []
    for seed in specification['seeds']:
        folder = reference_root / 'confirmation' / f'selected_seed{seed}'
        config = json.loads((folder / 'config.json').read_text())
        for key in ['epochs', 'hidden', 'heads', 'dropout', 'objective', 'num_negatives', 'temperature', 'batch_size', 'lr']:
            if config[key] != specification[key]:
                raise ValueError(f'Reference {seed} differs in {key}.')
        if config['blocks'] != 1 or config['max_steps'] or not config['skip_test'] or config.get('item_id_dropout', 0):
            raise ValueError('Require complete unmasked one-layer control without train-time test evaluation.')
        reference = dict(seed=seed, folder=str(folder),
                         files={name: digest(folder / name) for name in ['config.json', 'history.json', 'result.json', 'best.pt']})
        references.append(reference)
        weights = root / 'initialization' / f'seed{seed}.pt'
        manifest = prepare_depth_initialization(REPO / specification['cache'], folder / 'config.json', weights,
                                                max_blocks=max(specification['depths']))
        initializations.append(dict(seed=seed, file=str(weights), manifest=str(weights.with_suffix('.json')),
                                    manifest_sha256=digest(weights.with_suffix('.json'))))
        for depth in specification['depths']:
            if depth == 1:
                continue
            arguments = {key: config[key] for key in MATCHED_FIELDS}
            arguments.update(cache=str((REPO / specification['cache']).resolve()),
                             eval_indices=str((REPO / specification['evaluation_indices']).resolve()),
                             blocks=depth, initial_weights=str(weights))
            jobs.append((root / 'training' / f'depth_{depth}_seed{seed}', arguments))
    sources = ['model.py', 'taac_retrieval/model.py', 'taac_retrieval/train.py', 'taac_retrieval/data.py',
               'taac_retrieval/objectives.py', 'taac_retrieval/metrics.py', 'taac_retrieval/initialization.py',
               'taac_retrieval/evaluate_checkpoint.py', 'taac_retrieval/coverage.py', 'taac_retrieval/study.py',
               'scripts/run_depth_study.py']
    plan = dict(specification=specification, references=references, initializations=initializations,
                jobs=[dict(folder=str(folder), arguments=arguments) for folder, arguments in jobs],
                source_sha256={name: digest(REPO / name) for name in sources})
    path = root / 'plan.json'
    if path.exists():
        existing = json.loads(path.read_text())
        existing.pop('registered_at_utc')
        if existing != plan:
            raise ValueError('Depth plan, initial state, reference or source changed; use a new --root.')
    else:
        root.mkdir(parents=True, exist_ok=True)
        plan['registered_at_utc'] = datetime.now(timezone.utc).isoformat()
        for name in sources:
            destination = root / 'source_snapshot' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO / name).read_bytes())
        path.write_text(json.dumps(plan, indent=2) + '\n')
    return jobs


def check_runs(root, specification):
    for depth in [value for value in specification['depths'] if value != 1]:
        for seed in specification['seeds']:
            folder = root / 'training' / f'depth_{depth}_seed{seed}'
            config = json.loads((folder / 'config.json').read_text())
            result = json.loads((folder / 'result.json').read_text())
            history = json.loads((folder / 'history.json').read_text())
            reference = json.loads((REPO / specification['reference_root'] / 'confirmation'
                                    / f'selected_seed{seed}' / 'config.json').read_text())
            if (len(history) != specification['epochs'] or result['test'] is not None
                    or config['blocks'] != depth or config.get('item_id_dropout', 0) != 0):
                raise AssertionError('Incomplete training or wrong depth/masking/test configuration.')
            for field in MATCHED_FIELDS + ['evaluation_cohort_sha256']:
                if config[field] != reference[field]:
                    raise AssertionError(f'{folder.name}: unmatched {field}.')
            if config['common_initialization_sha256'] != reference['shared_initialization_sha256']:
                raise AssertionError('Common initial weights are not identical to historical one-layer initial weights.')


def diagnose(root, specification, gpus):
    check_runs(root, specification)
    diagnostics = root / 'diagnostics'
    diagnostics.mkdir(parents=True, exist_ok=True)
    pending = []
    for depth in specification['depths']:
        for seed in specification['seeds']:
            prefix = diagnostics / f'depth_{depth}_seed{seed}'
            if depth == 1:
                reference = REPO / specification['reference_root'] / 'confirmation' / f'selected_seed{seed}' / 'best.pt'
                existing = REPO / 'artifacts/cold_start/diagnostics' / f'click_seed{seed}'
                if existing.with_suffix('.json').exists() and not prefix.with_suffix('.json').exists():
                    description = json.loads(existing.with_suffix('.json').read_text())
                    if (description['checkpoint_sha256'] != digest(reference) or description['split'] != 'val'
                            or description['ks'] != specification['retrieval_budgets']):
                        raise ValueError('Reusable control coverage has a different identity.')
                    for suffix in ['.json', '.npz']:
                        shutil.copyfile(existing.with_suffix(suffix), prefix.with_suffix(suffix))
                checkpoint = reference
            else:
                checkpoint = root / 'training' / f'depth_{depth}_seed{seed}' / 'best.pt'
            if prefix.with_suffix('.json').exists():
                saved = json.loads(prefix.with_suffix('.json').read_text())
                if saved['checkpoint_sha256'] != digest(checkpoint) or not prefix.with_suffix('.npz').exists():
                    raise ValueError('Existing depth coverage has a different identity.')
                continue
            pending.append((checkpoint, prefix))
    for start in range(0, len(pending), len(gpus)):
        processes = []
        for gpu, (checkpoint, prefix) in zip(gpus, pending[start:start + len(gpus)]):
            command = [sys.executable, '-m', 'taac_retrieval.coverage', '--checkpoint', str(checkpoint),
                       '--cache', str((REPO / specification['cache']).resolve()), '--output', str(prefix),
                       '--device', f'cuda:{gpu}']
            with prefix.with_suffix('.log').open('w') as log:
                processes.append(subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT))
        if any(code != 0 for code in [process.wait() for process in processes]):
            raise RuntimeError('Depth coverage failed; inspect diagnostic logs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/depth_study.json')
    parser.add_argument('--root', default='artifacts/depth_study')
    parser.add_argument('--report-dir', default='reports/depth_study')
    parser.add_argument('--gpus', default='0,2,3,4,5,6')
    parser.add_argument('--stage', choices=['all', 'prepare', 'train', 'finish'], default='all')
    args = parser.parse_args()
    root, reports = (REPO / args.root).resolve(), (REPO / args.report_dir).resolve()
    specification = json.loads((REPO / args.config).read_text())
    gpus = [int(value) for value in args.gpus.split(',')]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('Provide unique GPU indices.')
    jobs = freeze_plan(root, specification)
    if args.stage in ['all', 'train']:
        run_training_jobs(jobs, gpus, REPO)
        check_runs(root, specification)
    if args.stage in ['all', 'finish']:
        diagnose(root, specification, gpus)
        from taac_retrieval.depth_report import report
        report(root, reports)
    print(f'Completed {args.stage}: {root}', flush=True)


if __name__ == '__main__':
    main()
