"""Train controlled item-ID masking and diagnose validation candidate coverage."""

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


MATCHED_FIELDS = ['variant', 'seed', 'epochs', 'batch_size', 'eval_batch_size', 'candidate_chunk',
                  'workers', 'hidden', 'blocks', 'heads', 'dropout', 'lr', 'max_steps', 'objective',
                  'num_negatives', 'exposure_weight', 'temperature', 'skip_test']


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def arm_name(probability):
    return f'id_mask_{round(probability * 100):02d}'


def freeze_plan(root, specification):
    reference_root = (REPO / specification['reference_root']).resolve()
    jobs, references = [], []
    for seed in specification['seeds']:
        reference = reference_root / 'confirmation' / f'selected_seed{seed}'
        config = json.loads((reference / 'config.json').read_text())
        for key in ['epochs', 'objective', 'num_negatives', 'temperature']:
            if config[key] != specification[key]:
                raise ValueError(f'Reference {seed} does not match {key}.')
        if config['max_steps'] or not config['skip_test'] or config.get('item_id_dropout', 0):
            raise ValueError('Require complete, unmasked reference training without test selection.')
        if Path(config['cache']).resolve() != (REPO / specification['cache']).resolve():
            raise ValueError('Reference data cache differs.')
        references.append(dict(seed=seed, folder=str(reference),
                               shared_initialization_sha256=config['shared_initialization_sha256'],
                               files={name: digest(reference / name) for name in
                                      ['config.json', 'history.json', 'result.json', 'best.pt']}))
    for probability in specification['item_id_dropout_probabilities']:
        if not 0 < probability < 1:
            raise ValueError('Ablations require probabilities strictly between 0 and 1.')
        for reference in references:
            config = json.loads((Path(reference['folder']) / 'config.json').read_text())
            arguments = {key: config[key] for key in MATCHED_FIELDS}
            arguments.update(cache=str((REPO / specification['cache']).resolve()),
                             eval_indices=str((REPO / specification['evaluation_indices']).resolve()),
                             item_id_dropout=probability)
            jobs.append((root / 'training' / f"{arm_name(probability)}_seed{config['seed']}", arguments))
    sources = ['model.py', 'taac_retrieval/model.py', 'taac_retrieval/train.py',
               'taac_retrieval/objectives.py', 'taac_retrieval/data.py', 'taac_retrieval/metrics.py',
               'taac_retrieval/evaluate_checkpoint.py', 'taac_retrieval/coverage.py',
               'taac_retrieval/study.py', 'scripts/run_cold_start.py']
    plan = dict(specification=specification, references=references,
                jobs=[dict(folder=str(folder), arguments=arguments) for folder, arguments in jobs],
                source_sha256={name: digest(REPO / name) for name in sources})
    path = root / 'plan.json'
    if path.exists():
        previous = json.loads(path.read_text())
        previous.pop('registered_at_utc')
        if previous != plan:
            raise ValueError('Study settings/reference/training source changed; use a new --root.')
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
    reference_root = (REPO / specification['reference_root']).resolve()
    for probability in specification['item_id_dropout_probabilities']:
        for seed in specification['seeds']:
            reference = reference_root / 'confirmation' / f'selected_seed{seed}'
            candidate = root / 'training' / f'{arm_name(probability)}_seed{seed}'
            a = json.loads((reference / 'config.json').read_text())
            b = json.loads((candidate / 'config.json').read_text())
            result = json.loads((candidate / 'result.json').read_text())
            history = json.loads((candidate / 'history.json').read_text())
            for key in MATCHED_FIELDS + ['parameters', 'shared_initialization_sha256', 'evaluation_cohort_sha256']:
                if a[key] != b[key]:
                    raise AssertionError(f'{candidate.name}: unmatched {key}.')
            if (b['item_id_dropout'] != probability or result['test'] is not None
                    or len(history) != specification['epochs']):
                raise AssertionError('Incomplete training or wrong masking/test policy.')


def diagnose_runs(root, specification, gpus):
    check_runs(root, specification)
    pending = []
    for probability in specification['item_id_dropout_probabilities']:
        for seed in specification['seeds']:
            name = f'{arm_name(probability)}_seed{seed}'
            prefix = root / 'diagnostics' / name
            if prefix.with_suffix('.json').exists():
                saved = json.loads(prefix.with_suffix('.json').read_text())
                if saved['checkpoint_sha256'] != digest(root / 'training' / name / 'best.pt'):
                    raise ValueError('Existing coverage belongs to another checkpoint.')
                continue
            pending.append((name, prefix))
    for start in range(0, len(pending), len(gpus)):
        processes = []
        for gpu, (name, prefix) in zip(gpus, pending[start:start + len(gpus)]):
            prefix.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-m', 'taac_retrieval.coverage', '--checkpoint',
                       str(root / 'training' / name / 'best.pt'), '--cache',
                       str((REPO / specification['cache']).resolve()), '--output', str(prefix),
                       '--device', f'cuda:{gpu}']
            with prefix.with_suffix('.log').open('w') as log:
                processes.append(subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT))
        if any(code != 0 for code in [process.wait() for process in processes]):
            raise RuntimeError('Coverage diagnosis failed; inspect diagnostic logs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/cold_start.json')
    parser.add_argument('--root', default='artifacts/cold_start')
    parser.add_argument('--report-dir', default='reports/cold_start')
    parser.add_argument('--gpus', default='0,2,3,4,5,6')
    parser.add_argument('--stage', choices=['all', 'train', 'finish'], default='all')
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
        diagnose_runs(root, specification, gpus)
        from taac_retrieval.cold_start_report import report
        report(root, reports)
    print(f'Completed {args.stage}: {root}', flush=True)


if __name__ == '__main__':
    main()
