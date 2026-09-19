"""Reproduce the numerical summaries and figures of the paper.

`paper` rebuilds reproduction/analysis/numbers.json from the archived experiments
under output/ and draws the five figures into reproduction/figures/. Nothing is
simulated or timed again, and the drivers that produced the archives are in studies/.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Fixed date for matplotlib PDF metadata so regenerated figures are byte-identical
# across runs (matplotlib honors SOURCE_DATE_EPOCH). Subprocesses inherit this.
os.environ.setdefault('SOURCE_DATE_EPOCH', '1704067200')  # 2024-01-01 UTC


def main():
    parser = argparse.ArgumentParser(
        prog='reproduce.py',
        description='Reproduce the SAC-GRW paper summaries and figures from the archived experiments.')
    parser.add_argument('target', choices=['paper'])
    parser.parse_args()
    scripts = ['build_numbers.py', 'make_figures.py']
    for script in scripts:
        subprocess.run([sys.executable, os.path.join(ROOT, 'reproduction', 'analysis', script)],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
