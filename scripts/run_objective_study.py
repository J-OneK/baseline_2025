"""Reproduce the objective study, fresh-cohort confirmation, ANN benchmark and report."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/objective_study_reproduction')
    parser.add_argument('--baseline-root', default='artifacts/baseline_reproduction')
    parser.add_argument('--config', default='configs/objective_study.json')
    parser.add_argument('--report-dir', default='reports/objective_study_reproduction',
                        help='A separate report directory; historical reports are preserved')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    parser.add_argument('--stage', choices=['all', 'finish'], default='all')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    def resolve(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else repo / path).resolve()

    root, baseline = resolve(args.root), resolve(args.baseline_root)
    reports = resolve(args.report_dir) if args.report_dir else root / 'report'
    args.config = str(resolve(args.config))
    config = json.loads(Path(args.config).read_text())
    gpus = [int(value) for value in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError('Provide one or more unique nonnegative CUDA indices.')
    primary_gpu = gpus[0]
    root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    def run(arguments, stdout=None):
        print('Running:', ' '.join([sys.executable] + arguments), flush=True)
        subprocess.run([sys.executable] + arguments, cwd=repo, check=True, stdout=stdout)

    with (root / 'acceptance_tests.log').open('w') as test_log:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                       cwd=repo, stdout=test_log, stderr=subprocess.STDOUT, check=True)
    if args.stage == 'all':
        old_metadata = json.loads((baseline / 'data/metadata.json').read_text())
        prep = old_metadata['preparation_config']
        run(['-m', 'taac_retrieval.prepare', '--data-dir', old_metadata['source_data'],
             '--output', str(root / 'scale_data'), '--max-users', str(config['scale_sampled_users']),
             '--eval-users', str(config['confirmation_eval_users']), '--seed', str(prep['seed']),
             '--max-history', str(prep['max_history']), '--min-history', str(prep['min_history'])])
        run(['-m', 'taac_retrieval.cohorts', '--cache', str(root / 'scale_data'),
             '--previous-cache', str(baseline / 'data'), '--output', str(root / 'confirm_eval_indices.npy'),
             '--users', str(config['confirmation_eval_users']), '--seed', str(config['confirmation_cohort_seed'])])
        run(['-m', 'taac_retrieval.audit', '--cache', str(root / 'scale_data'),
             '--output', str(reports / 'scale_data_audit.json')])
        run(['-m', 'taac_retrieval.study', '--stage', 'models', '--root', str(root),
             '--baseline-root', str(baseline), '--config', args.config, '--report-dir', str(reports), '--gpus', args.gpus])
    for name in ['baseline', 'selected']:
        for seed in config['confirmation_seeds']:
            if not (root / 'confirmation' / f'{name}_seed{seed}' / 'result.json').exists():
                raise FileNotFoundError('Confirmation is incomplete. Run --stage all, or wait for active model jobs.')
    run(['-m', 'taac_retrieval.study', '--stage', 'test', '--root', str(root),
         '--baseline-root', str(baseline), '--config', args.config, '--report-dir', str(reports), '--gpus', args.gpus])
    checkpoint = root / 'confirmation' / f"selected_seed{config['faiss']['primary_seed']}" / 'best.pt'
    run(['-m', 'taac_retrieval.export', '--checkpoint', str(checkpoint), '--cache', str(root / 'scale_data'),
         '--output', str(root / 'export'), '--device', f'cuda:{primary_gpu}'])
    if not (root / 'ann/benchmark.json').exists():
        ann_config = args.config
        recovery = root / 'ann_recovery_config.json'
        if (root / 'ann_raw_ip_failed').exists():
            amended = json.loads(json.dumps(config))
            amended['faiss']['hnsw_geometry'] = 'mips_l2'
            amended['faiss']['efSearch'] = sorted(set(amended['faiss']['efSearch'] + [2048, 4096]))
            recovery.write_text(json.dumps(amended, indent=2) + '\n')
            ann_config = str(recovery)
        command = ['-m', 'taac_retrieval.benchmark_ann', '--export-dir', str(root / 'export'),
                   '--output', str(root / 'ann'), '--config', ann_config]
        try:
            run(command)
        except subprocess.CalledProcessError:
            sweep = root / 'ann/validation_sweep.json'
            if not sweep.exists() or ann_config != args.config:
                raise
            rows = json.loads(sweep.read_text())
            threshold = config['faiss']['minimum_validation_ann_recall']
            if any(row['ANNRecall@10'] >= threshold for row in rows):
                raise  # An unrelated error must not be hidden as a recall recovery.
            (reports / 'ann_initial_failure.json').write_text(json.dumps(dict(
                threshold=threshold, eligible=False, validation_configurations=rows,
                test_queries_evaluated=False, reason='Raw-IP HNSW did not meet the validation recall gate.'), indent=2) + '\n')
            (root / 'ann').rename(root / 'ann_raw_ip_failed')
            amended = json.loads(json.dumps(config))
            amended['faiss']['hnsw_geometry'] = 'mips_l2'
            amended['faiss']['efSearch'] = sorted(set(amended['faiss']['efSearch'] + [2048, 4096]))
            recovery.write_text(json.dumps(amended, indent=2) + '\n')
            (reports / 'ann_amendment.json').write_text(json.dumps(dict(
                registered_at_utc=datetime.now(timezone.utc).isoformat(),
                model_selection_unchanged=True, threshold_unchanged=threshold,
                test_ann_queries_unopened=True, candidate_geometry='mips_l2',
                reason='Validation recall failure; use an exact-ranking-preserving asymmetric lift, never cosine normalization.'), indent=2) + '\n')
            run(command[:-1] + [str(recovery)])
    with (reports / 'recommend_example.json').open('w') as prediction:
        run(['-m', 'taac_retrieval.recommend', '--checkpoint', str(checkpoint), '--cache', str(root / 'scale_data'),
             '--export-dir', str(root / 'export'), '--index-dir', str(root / 'ann'),
             '--device', f'cuda:{primary_gpu}', '--split', 'test', '--index', '0'], stdout=prediction)
    run(['-m', 'taac_retrieval.objective_report', '--root', str(root), '--output', str(reports),
         '--baseline-root', str(baseline), '--config', args.config])
    match = re.search(r'Ran (\d+) tests', (root / 'acceptance_tests.log').read_text())
    if match is None:
        raise RuntimeError('Acceptance test count was not found in the test log.')
    cohort = json.loads((root / 'confirm_eval_indices.json').read_text())
    development_runs = list((root / 'development').glob('*/result.json'))
    confirmation_runs = list((root / 'confirmation').glob('*/result.json'))
    verification = dict(unit_tests=int(match.group(1)), acceptance_tests_passed=True,
                        completed_development_runs=len(development_runs),
                        completed_confirmation_runs=len(confirmation_runs),
                        fresh_test_cohort_overlap_with_previous=cohort['overlap_with_previous_test_users'], validation_only_model_and_index_selection=True,
                        checkpoint_inference_and_persisted_ann_demo_passed=True,
                        source_data_modified=False, full_workflow_finished=True)
    (reports / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print('Completed all stages. Report:', reports / 'README.md', flush=True)


if __name__ == '__main__':
    main()
