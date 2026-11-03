"""Match all-event InfoNCE-64 against the three existing click-only runs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from taac_retrieval.study import run_training_jobs


MATCHED_FIELDS = [
    'cache', 'variant', 'seed', 'epochs', 'batch_size', 'eval_batch_size',
    'candidate_chunk', 'workers', 'hidden', 'blocks', 'heads', 'dropout', 'lr',
    'max_steps', 'num_negatives', 'exposure_weight', 'temperature', 'skip_test',
    'eval_indices',
]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def freeze_plan(root, specification):
    """Freeze reference identities and training settings before new results."""
    reference_root = (REPO / specification['reference_root']).resolve()
    jobs, references = [], []
    for seed in specification['seeds']:
        folder = reference_root / 'confirmation' / f'selected_seed{seed}'
        actual = json.loads((folder / 'config.json').read_text())
        if (actual['objective'] != specification['control_objective']
                or actual['num_negatives'] != specification['num_negatives']
                or actual['temperature'] != specification['temperature']
                or actual['epochs'] != specification['epochs']
                or actual['max_steps'] != 0 or not actual['skip_test']
                or actual['variant'] != 'baseline'):
            raise ValueError(f'Reference {folder} does not match the ablation.')
        arguments = {key: actual[key] for key in MATCHED_FIELDS}
        arguments['objective'] = specification['experimental_objective']
        jobs.append((root / 'training' / f'all_infonce_seed{seed}', arguments))
        references.append(dict(seed=seed, output=str(folder),
                               shared_initialization_sha256=actual['shared_initialization_sha256'],
                               files={name: digest(folder / name) for name in
                                      ['config.json', 'history.json', 'result.json', 'best.pt']}))
    sources = ['model.py', 'taac_retrieval/data.py', 'taac_retrieval/model.py',
               'taac_retrieval/objectives.py', 'taac_retrieval/train.py',
               'taac_retrieval/metrics.py', 'taac_retrieval/study.py',
               'taac_retrieval/evaluate_checkpoint.py', 'scripts/run_infonce_supervision.py']
    frozen = dict(specification=specification, references=references,
                  jobs=[dict(output=str(path), arguments=arguments) for path, arguments in jobs],
                  source_sha256={name: digest(REPO / name) for name in sources})
    plan_path = root / 'plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        previous.pop('registered_at_utc')
        if previous != frozen:
            raise ValueError('Settings, training code or reference artifacts changed; use a new --root.')
    else:
        frozen['registered_at_utc'] = datetime.now(timezone.utc).isoformat()
        root.mkdir(parents=True, exist_ok=True)
        for name in sources:
            destination = root / 'source_snapshot' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO / name).read_bytes())
        plan_path.write_text(json.dumps(frozen, indent=2) + '\n')
    return jobs


def check_completed_runs(root, specification):
    reference_root = (REPO / specification['reference_root']).resolve()
    for seed in specification['seeds']:
        reference = reference_root / 'confirmation' / f'selected_seed{seed}'
        candidate = root / 'training' / f'all_infonce_seed{seed}'
        a = json.loads((reference / 'config.json').read_text())
        b = json.loads((candidate / 'config.json').read_text())
        result = json.loads((candidate / 'result.json').read_text())
        history = json.loads((candidate / 'history.json').read_text())
        for key in MATCHED_FIELDS + ['shared_initialization_sha256', 'evaluation_cohort_sha256', 'parameters']:
            if a[key] != b[key]:
                raise AssertionError(f'Seed {seed}: unmatched {key}.')
        if b['objective'] != specification['experimental_objective']:
            raise AssertionError('Candidate has the wrong objective.')
        if len(history) != specification['epochs'] or result['test'] is not None:
            raise AssertionError('Require complete training without test evaluation.')
        best = max(history, key=lambda row: (row['validation']['NDCG@10'],
                                           row['validation']['HR@10'], -row['epoch']))
        if result['best_epoch'] != best['epoch']:
            raise AssertionError('Checkpoint was not selected using validation.')


def finish(root, reports, specification, gpus):
    check_completed_runs(root, specification)
    # Load and inspect test targets only after all new checkpoints are fixed.
    pending = []
    for seed in specification['seeds']:
        output = root / 'evaluation' / f'all_infonce_seed{seed}'
        if output.with_suffix('.json').exists():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        pending.append((seed, output))
    for start in range(0, len(pending), len(gpus)):
        processes = []
        for gpu, (seed, output) in zip(gpus, pending[start:start + len(gpus)]):
            command = [sys.executable, '-m', 'taac_retrieval.evaluate_checkpoint',
                       '--checkpoint', str(root / 'training' / f'all_infonce_seed{seed}' / 'best.pt'),
                       '--cache', str((REPO / specification['reference_root'] / 'scale_data').resolve()),
                       '--output', str(output), '--device', f'cuda:{gpu}']
            with output.with_suffix('.log').open('w') as handle:
                processes.append(subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT))
        if any(code != 0 for code in [process.wait() for process in processes]):
            raise RuntimeError('Supplementary evaluation failed; inspect evaluation logs.')
    from taac_retrieval.supervision_report import report
    report(root, reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/infonce_supervision.json')
    parser.add_argument('--root', default='artifacts/infonce_supervision')
    parser.add_argument('--report-dir', default='reports/infonce_supervision')
    parser.add_argument('--gpus', default='0,2,4')
    parser.add_argument('--stage', choices=['all', 'train', 'finish'], default='all')
    args = parser.parse_args()
    root, reports = (REPO / args.root).resolve(), (REPO / args.report_dir).resolve()
    specification = json.loads((REPO / args.config).read_text())
    gpus = [int(value) for value in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('Provide unique GPU indices.')
    jobs = freeze_plan(root, specification)
    if args.stage in ['all', 'train']:
        run_training_jobs(jobs, gpus, REPO)
        check_completed_runs(root, specification)
    if args.stage in ['all', 'finish']:
        finish(root, reports, specification, gpus)
    print(f'Completed {args.stage}: {root}', flush=True)


if __name__ == '__main__':
    main()
