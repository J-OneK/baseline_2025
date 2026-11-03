"""Reproduce data preparation and the matched baseline/action control."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/baseline_reproduction', help='Fresh output directory')
    parser.add_argument('--data-dir', default='TencentGR-1M')
    parser.add_argument('--config', default='configs/baseline.json')
    parser.add_argument('--gpus', default='0,2')
    parser.add_argument('--report-dir', default='reports/baseline_reproduction')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repo / 'scripts/run_day1_3.py')]
    for key, value in vars(args).items():
        command.extend(['--' + key.replace('_', '-'), value])
    subprocess.run(command, cwd=repo, check=True)


if __name__ == '__main__':
    main()
