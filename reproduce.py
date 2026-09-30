"""Reproduce the numerical summaries and figures of the paper.

`paper` rebuilds reproduction/analysis/numbers.json from the archived experiments
under output/ and draws the six figures into reproduction/figures/. Nothing is
simulated or timed again. ``verify`` checks solver identities and recomputes central
archived results independently. The original experiment drivers are in studies/.
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
        description='Reproduce or verify the sign-switched coupling paper results.')
    parser.add_argument('target', nargs="?", choices=['paper', 'verify'])
    args = parser.parse_args()
    if args.target is None:
        parser.print_help()
        return
    if args.target == "verify":
        subprocess.run([sys.executable, os.path.join(ROOT, "checks", "verify_results.py")],
                       cwd=ROOT, check=True)
        return
    scripts = ['build_numbers.py', 'make_figures.py']
    for script in scripts:
        subprocess.run([sys.executable, os.path.join(ROOT, 'reproduction', 'analysis', script)],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
