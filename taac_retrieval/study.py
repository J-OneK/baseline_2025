"""Run predeclared development ablations and validation-locked confirmation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

DEVELOPMENT_VARIANTS = {
    'weighted_bce': dict(objective='weighted_bce', num_negatives=1, variant='baseline'),
    'click_bce': dict(objective='click_bce', num_negatives=1, variant='baseline'),
    'click_bce64': dict(objective='click_bce', num_negatives=64, variant='baseline'),
    'click_infonce64': dict(objective='click_infonce', num_negatives=64, variant='baseline'),
    'click_infonce64_actions': dict(objective='click_infonce', num_negatives=64, variant='actions'),
}
BASELINE = dict(objective='all_bce', num_negatives=1, variant='baseline')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_training_jobs(jobs, gpus, repo):
    """Each job owns one GPU; completion is checked before scheduling another."""
    pending = list(jobs)
    running, failed = {}, []
    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            output, arguments = pending.pop(0)
            output = Path(output)
            if (output / 'result.json').exists():
                actual = json.loads((output / 'config.json').read_text())
                for key, value in arguments.items():
                    if key == 'cache':
                        value = str(Path(value).resolve())
                    if actual.get(key) != value:
                        raise ValueError(f'Existing run {output} has a different {key}.')
                print(f'Reusing completed run {output}', flush=True)
                continue
            command = [sys.executable, '-m', 'taac_retrieval.train', '--output', str(output), '--device', f'cuda:{gpu}']
            for key, value in arguments.items():
                option = '--' + key.replace('_', '-')
                if isinstance(value, bool):
                    if value:
                        command.append(option)
                else:
                    command.extend([option, str(value)])
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.with_suffix('.log').open('w') as log:
                process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = process, output
            print(f'Started GPU {gpu}: {output.name}, PID {process.pid}', flush=True)
        for gpu, (process, output) in list(running.items()):
            code = process.poll()
            if code is not None:
                print(f'Finished GPU {gpu}: {output.name}, exit={code}', flush=True)
                if code:
                    failed.append(str(output))
                del running[gpu]
        if running:
            time.sleep(2)
    if failed:
        raise RuntimeError(f'Training jobs failed; inspect their logs: {failed}')


def best_validation(folder):
    history = json.loads((Path(folder) / 'history.json').read_text())
    row = max(history, key=lambda r: (r['validation']['NDCG@10'], r['validation']['HR@10'], -r['epoch']))
    return dict(epoch=row['epoch'], metrics=row['validation'])


def lock_selection(root, baseline_root, config, report_dir):
    """This function reads validation history only, including for old baselines."""
    rows, fingerprints = [], {}
    for name in ['baseline'] + list(DEVELOPMENT_VARIANTS):
        runs = []
        for seed in config['development_seeds']:
            folder = Path(baseline_root) / f'baseline_seed{seed}' if name == 'baseline' else root / 'development' / f'{name}_seed{seed}'
            best = best_validation(folder)
            actual = json.loads((folder / 'config.json').read_text())
            reference = json.loads((Path(baseline_root) / f'baseline_seed{seed}/config.json').read_text())
            if actual['shared_initialization_sha256'] != reference['shared_initialization_sha256']:
                raise AssertionError('Development ablations do not have matched initialization.')
            for key in ['epochs', 'batch_size', 'hidden', 'blocks', 'heads', 'dropout', 'lr']:
                if actual[key] != reference[key]:
                    raise AssertionError(f'Development budget mismatch: {key}')
            runs.append(dict(seed=seed, output=str(folder), **best))
            fingerprints[str(folder / 'history.json')] = file_hash(folder / 'history.json')
        rows.append(dict(name=name, runs=runs,
                         validation_ndcg=float(np.mean([r['metrics']['NDCG@10'] for r in runs])),
                         validation_hr=float(np.mean([r['metrics']['HR@10'] for r in runs]))))
    ranking = sorted(rows, key=lambda r: (-r['validation_ndcg'], -r['validation_hr'], r['name']))
    selected = ranking[0]['name']
    candidate = selected if selected != 'baseline' else next(r['name'] for r in ranking if r['name'] != 'baseline')
    settings = DEVELOPMENT_VARIANTS[candidate]
    result = dict(selected_variant=selected, confirmation_candidate=candidate, confirmation_settings=settings,
                  criterion=config['selection'], test_used_for_selection=False, ranking=ranking,
                  history_sha256=fingerprints,
                  scope='Development uses the old 100k-user cache; confirmation uses 300k sampled users and fresh test users.')
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / 'model_selection.json'
    if path.exists():
        old = json.loads(path.read_text())
        for key in ['selected_variant', 'confirmation_candidate', 'confirmation_settings', 'history_sha256']:
            if old[key] != result[key]:
                raise ValueError('The locked model selection has changed. Start a new study.')
        return old
    result['locked_at_utc'] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Locked using validation only: {selected}; confirmation comparison: {candidate}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/objective_study')
    parser.add_argument('--baseline-root', default='artifacts/baseline')
    parser.add_argument('--config', default='configs/objective_study.json')
    parser.add_argument('--report-dir', default='reports/objective_study')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    parser.add_argument('--stage', choices=['development', 'confirmation', 'models', 'test'], default='models')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root, baseline_root, reports = Path(args.root).resolve(), Path(args.baseline_root).resolve(), Path(args.report_dir).resolve()
    config = json.loads(Path(args.config).read_text())
    gpus = [int(value) for value in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('Unique GPU indices are required.')
    common = {key: config[key] for key in ['epochs', 'batch_size', 'hidden', 'blocks', 'heads', 'dropout', 'lr', 'exposure_weight', 'temperature']}
    common.update(workers=2, skip_test=True)
    root.mkdir(parents=True, exist_ok=True)
    plan = dict(config=config, variants=DEVELOPMENT_VARIANTS,
                negative_count_control='click_bce64 isolates the 64-negative budget from the contrastive objective',
                confirmation_eval_indices=str(root / 'confirm_eval_indices.npy'))
    plan_path = root / 'study_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Study plan changed; use a new output directory.')
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    if args.stage in ['development', 'models']:
        jobs = []
        for name, settings in DEVELOPMENT_VARIANTS.items():
            for seed in config['development_seeds']:
                arguments = dict(common, cache=str(baseline_root / 'data'), seed=seed, **settings)
                jobs.append((root / 'development' / f'{name}_seed{seed}', arguments))
        run_training_jobs(jobs, gpus, repo)
    selection = lock_selection(root, baseline_root, config, reports)
    if args.stage in ['confirmation', 'models']:
        if not (root / 'scale_data/metadata.json').exists() or not (root / 'confirm_eval_indices.npy').exists():
            raise FileNotFoundError('Prepare the scale cache and freeze the fresh evaluation cohort first.')
        jobs = []
        for name, settings in [('baseline', BASELINE), ('selected', selection['confirmation_settings'])]:
            for seed in config['confirmation_seeds']:
                arguments = dict(common, cache=str(root / 'scale_data'), seed=seed,
                                 eval_indices=str(root / 'confirm_eval_indices.npy'), **settings)
                jobs.append((root / 'confirmation' / f'{name}_seed{seed}', arguments))
        run_training_jobs(jobs, gpus, repo)
    if args.stage == 'test':
        evaluation_jobs = []
        for name in ['baseline', 'selected']:
            for seed in config['confirmation_seeds']:
                folder = root / 'confirmation' / f'{name}_seed{seed}'
                result = json.loads((folder / 'result.json').read_text())
                if result['test'] is not None:
                    raise AssertionError('Confirmation training must not have already consumed test labels.')
                output = root / 'evaluation' / f'{name}_seed{seed}'
                if output.with_suffix('.json').exists():
                    continue
                command = [sys.executable, '-m', 'taac_retrieval.evaluate_checkpoint', '--checkpoint', str(folder / 'best.pt'),
                           '--cache', str(root / 'scale_data'), '--output', str(output)]
                evaluation_jobs.append((command, output))
        for offset in range(0, len(evaluation_jobs), len(gpus)):
            jobs = []
            for gpu, (command, output) in zip(gpus, evaluation_jobs[offset:offset+len(gpus)]):
                command = command + ['--device', f'cuda:{gpu}']
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.with_suffix('.log').open('w') as log:
                    process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
                jobs.append(process)
            codes = [p.wait() for p in jobs]
            if any(codes):
                raise RuntimeError(f'Test evaluation failed: {codes}')
    print(f'Completed stage {args.stage}', flush=True)


if __name__ == '__main__':
    main()
