"""Master runner for the 5 pre-publication studies (Tasks 1-5) + final manifest.

Usage:
    python run_prepublication_studies.py [--tasks 1 2 3 4 5] [--seeds 30] [--fast]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

MANIFEST_BASE = 'output/final_prepublication_tests/final_manifest'


def _banner(msg):
    print(f"\n{'#'*70}\n#  {msg}\n{'#'*70}")


def run_all(tasks, seeds, fast=False):
    os.makedirs(MANIFEST_BASE, exist_ok=True)
    timings = {}
    results = {}

    # ---- Task 1: GBMC dt-bias ----
    if 1 in tasks:
        _banner("Task 1: GBMC dt-bias study")
        from study_t1_gbmc_dt_bias import run_task1
        if fast:
            N_vals = [800]
            dt_seq = [0.02, 0.01, 0.005, 0.0025]
            S = 5
        else:
            N_vals = [6400]
            dt_seq = [0.01, 0.005, 0.0025, 0.00125, 0.000625]
            S = seeds
        t0 = time.perf_counter()
        r1 = run_task1(N_vals=N_vals, dt_seq=dt_seq, S=S)
        timings['task1'] = time.perf_counter() - t0
        results['task1'] = r1
        print(f"  Task 1 done in {timings['task1']:.1f}s")

    # ---- Task 2: Traveling-shock GBMC validation ----
    if 2 in tasks:
        _banner("Task 2: Traveling-shock validation")
        from study_t2_traveling_shock import run_task2
        if fast:
            N_seq = [100, 200, 400, 800]
            S = 5
        else:
            N_seq = [100, 200, 400, 800, 1600, 3200]
            S = seeds
        t0 = time.perf_counter()
        r2 = run_task2(N_seq=N_seq, S=S, output_times=[0.25, 0.5, 1.0])
        timings['task2'] = time.perf_counter() - t0
        results['task2'] = r2
        print(f"  Task 2 done in {timings['task2']:.1f}s")

    # ---- Task 3: Cole-Hopf plateau diagnosis ----
    if 3 in tasks:
        _banner("Task 3: Cole-Hopf plateau diagnosis")
        from study_t3_cole_hopf_plateau import run_task3
        S3 = 5 if fast else 10
        t0 = time.perf_counter()
        r3 = run_task3(S=S3)
        timings['task3'] = time.perf_counter() - t0
        results['task3'] = r3
        print(f"  Task 3 done in {timings['task3']:.1f}s")

    # ---- Task 4: Extended Heat study ----
    if 4 in tasks:
        _banner("Task 4: Extended Heat convergence")
        from study_t4_heat_extended import run_task4
        if fast:
            N_seq = [500, 1000, 2000, 5000]
            S = 5
        else:
            N_seq = [500, 1000, 2000, 5000, 10000, 20000, 50000]
            S = seeds
        t0 = time.perf_counter()
        r4 = run_task4(N_seq=N_seq, S=S)
        timings['task4'] = time.perf_counter() - t0
        results['task4'] = r4
        print(f"  Task 4 done in {timings['task4']:.1f}s")

    # ---- Task 5: Extended FHN study ----
    if 5 in tasks:
        _banner("Task 5: Extended FHN convergence")
        from study_t5_fhn_extended import run_task5
        if fast:
            N_seq = [100, 200, 500, 1000]
            S = 5
        else:
            N_seq = [100, 200, 500, 1000, 2000, 5000]
            S = seeds
        t0 = time.perf_counter()
        r5 = run_task5(N_seq=N_seq, S=S, T=5.0, dt=0.01, L=30.0, xc=15.0, a=0.25)
        timings['task5'] = time.perf_counter() - t0
        results['task5'] = r5
        print(f"  Task 5 done in {timings['task5']:.1f}s")

    # ---- Generate final manifest ----
    _banner("Generating final manifest")
    from generate_final_manifest import generate_manifest
    generate_manifest(results, timings, tasks)

    total = sum(timings.values())
    print(f"\n  Total wall time: {total:.1f}s")
    print(f"  Manifest written to: {MANIFEST_BASE}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--seeds', type=int, default=30)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    run_all(tasks=set(args.tasks), seeds=args.seeds, fast=args.fast)
