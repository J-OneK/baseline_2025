"""Reproduce objective ablations, confirmation, vector export and ANN study."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/objective_study_reproduction', help='Fresh output directory')
    parser.add_argument('--baseline-root', default='artifacts/baseline')
    parser.add_argument('--config', default='configs/objective_study.json')
    parser.add_argument('--report-dir', default='reports/objective_study_reproduction')
    parser.add_argument('--gpus', default='0,2,4')
    parser.add_argument('--stage', choices=['all', 'finish'], default='all')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repo / 'scripts/run_day4_7.py')]
    for key, value in vars(args).items():
        command.extend(['--' + key.replace('_', '-'), value])
    subprocess.run(command, cwd=repo, check=True)


if __name__ == '__main__':
    main()
