"""Reproduce preparation, audits, matched training seeds, and the report.

Example: python scripts/run_day1_3.py --gpus 0,1 --root artifacts/reproduction
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='New output directory; existing runs are not overwritten')
    parser.add_argument('--data-dir', default='TencentGR-1M')
    parser.add_argument('--config', default='configs/day1_3.json')
    parser.add_argument('--gpus', default='0,1', help='Unique CUDA device indices; one run per device')
    parser.add_argument('--report-dir', default=None)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / args.config).read_text())
    root = (repo / args.root).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else root / 'report'
    gpus = [int(value) for value in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('Provide one or more unique CUDA device indices.')
    jobs = [(variant, seed) for seed in config['training_seeds'] for variant in ['baseline', 'actions']]
    for variant, seed in jobs:
        if (root / f'{variant}_seed{seed}').exists():
            raise FileExistsError('A model output already exists. Use a new --root to reproduce without overwriting.')
    root.mkdir(parents=True, exist_ok=True)

    def run(arguments):
        subprocess.run([sys.executable] + arguments, cwd=repo, check=True)

    run(['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
    run(['-m', 'taac_retrieval.prepare', '--data-dir', args.data_dir, '--output', str(root / 'data'),
         '--max-users', str(config['max_users']), '--seed', str(config['data_seed']),
         '--max-history', str(config['max_history']), '--min-history', str(config['min_history']),
         '--eval-users', str(config['eval_users'])])
    run(['-m', 'taac_retrieval.audit', '--cache', str(root / 'data'),
         '--output', str(report_dir / 'data_audit.json')])
    manifest = []
    for offset in range(0, len(jobs), len(gpus)):
        running = []
        for gpu, (variant, seed) in zip(gpus, jobs[offset:offset + len(gpus)]):
            output = root / f'{variant}_seed{seed}'
            command = ['-m', 'taac_retrieval.train', '--cache', str(root / 'data'),
                       '--output', str(output), '--variant', variant, '--seed', str(seed),
                       '--device', f'cuda:{gpu}']
            for key in ['epochs', 'batch_size', 'workers', 'hidden', 'blocks', 'heads', 'dropout', 'lr',
                        'eval_batch_size', 'candidate_chunk']:
                command += ['--' + key.replace('_', '-'), str(config[key])]
            with output.with_suffix('.log').open('w') as log:
                process = subprocess.Popen([sys.executable] + command, cwd=repo,
                                           stdout=log, stderr=subprocess.STDOUT)
            manifest.append(dict(variant=variant, seed=seed, gpu=gpu, pid=process.pid,
                                 output=str(output), command=[sys.executable] + command))
            running.append(process)
            print(f'{variant} seed={seed} GPU={gpu} PID={process.pid}', flush=True)
        (root / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        codes = [process.wait() for process in running]
        if any(codes):
            raise RuntimeError(f'Training failed with exit codes {codes}; inspect the run logs.')
    run(['-m', 'taac_retrieval.report', '--root', str(root), '--output', str(report_dir),
         '--seeds'] + [str(seed) for seed in config['training_seeds']])
    print(f'Completed. Report: {report_dir / "README.md"}')


if __name__ == '__main__':
    main()
