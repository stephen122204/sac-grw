"""Focused relaxation-speed sensitivity study for Paper 2.

The principal stationary study fixes ``a=2``.  This study varies only the
two-speed relaxation parameter while retaining its PDE, physical viscosity,
time step, final time, output points, and seed list.  Three particle counts
separate a parameter effect from behavior at one discretization.
"""

import contextlib
import csv
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from studies.study_gbmc_production_n_refinement import (
    A,
    BASE_SEED,
    DT,
    L,
    N_OUT,
    NU,
    T,
    XC,
    _run_one,
    _savez_deterministic,
    u_exact_fn,
)


OUT_BASE = os.path.join(
    'output', 'final_prepublication_tests',
    'gbmc_relaxation_speed_sensitivity',
)
A_SEQ = [1.5, 2.0, 3.0, 4.0]
N_SEQ = [400, 1600, 6400]
S = 50


def _case(a_rel, N):
    return f"a={a_rel:g},N={N}"


def _array_key(a_rel, N):
    return f"a{a_rel:g}".replace('.', 'p') + f"_N{N}"


def _spread_interval(profiles, dx, rng, n_boot=5000):
    """Percentile interval obtained by resampling realizations."""
    S_actual = len(profiles)
    values = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        sample = profiles[rng.integers(0, S_actual, size=S_actual)]
        sample_mean = sample.mean(axis=0)
        values[k] = np.sqrt(np.mean(
            np.sum((sample - sample_mean[None, :]) ** 2 * dx, axis=1)
        ))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def run_study():
    os.makedirs(OUT_BASE, exist_ok=True)
    seeds = [BASE_SEED + s for s in range(S)]
    x_ref = np.linspace(0.0, L, N_OUT)
    u_ref = u_exact_fn(x_ref)
    dx = float(x_ref[1] - x_ref[0])
    rng_boot = np.random.default_rng(20260820)

    summaries = []
    per_run = []
    stored_arrays = {'x': x_ref, 'u_exact': u_ref}

    print("Relaxation-speed sensitivity")
    print(f"a={A_SEQ}, N={N_SEQ}, S={S}, nu={NU}, dt={DT}, T={T}")

    for N in N_SEQ:
        for a_rel in A_SEQ:
            profiles = []
            records = []
            for seed in seeds:
                # The solver's per-run diagnostics are useful interactively but
                # obscure the study-level record, so this driver suppresses them.
                with contextlib.redirect_stdout(io.StringIO()):
                    result = _run_one(N, seed, a_rel=a_rel)
                if result is None or result.get('failed'):
                    reason = (result or {}).get('reason', 'unknown failure')
                    raise RuntimeError(
                        f"Sensitivity run failed for a={a_rel}, N={N}, "
                        f"seed={seed}: {reason}"
                    )
                profiles.append(result['u_out'])
                records.append(result)
                per_run.append({
                    'case': _case(a_rel, N),
                    'a': a_rel,
                    'N': N,
                    'seed': seed,
                    'l2': result['l2'],
                    'xc_fit': result['xc_fit'],
                    'nu_fit': result['nu_fit'],
                    'A_fit': result['A_fit'],
                    'runtime_s': result['runtime_s'],
                })

            u_arr = np.asarray(profiles)
            u_mean = u_arr.mean(axis=0)
            E_bias = float(np.sqrt(np.sum((u_mean - u_ref) ** 2 * dx)))
            E_spread = float(np.sqrt(np.mean(
                np.sum((u_arr - u_mean[None, :]) ** 2 * dx, axis=1)
            )))
            E_total = float(np.sqrt(np.mean(
                np.sum((u_arr - u_ref[None, :]) ** 2 * dx, axis=1)
            )))
            identity_error = abs(
                E_total ** 2 - E_bias ** 2 - E_spread ** 2
            )
            spread_lo, spread_hi = _spread_interval(
                u_arr, dx, rng_boot
            )
            nu_values = np.asarray([row['nu_fit'] for row in records])
            summary = {
                'case': _case(a_rel, N),
                'a': a_rel,
                'N': N,
                'S': S,
                'E_bias': E_bias,
                'E_spread': E_spread,
                'E_total': E_total,
                'spread_ci_lo': spread_lo,
                'spread_ci_hi': spread_hi,
                'finite_ensemble_scale': E_spread / np.sqrt(S - 1),
                'nu_mean': float(nu_values.mean()),
                'nu_std': float(nu_values.std()),
                'identity_error': identity_error,
                'n_failed': 0,
            }
            summaries.append(summary)
            stored_arrays[_array_key(a_rel, N)] = u_arr
            print(
                f"{summary['case']:14s} "
                f"bias={E_bias:.5f} spread={E_spread:.5f} "
                f"total={E_total:.5f} nu_fit={summary['nu_mean']:.5f}"
            )

    summary_path = os.path.join(OUT_BASE, 'summary.csv')
    with open(summary_path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    per_run_path = os.path.join(OUT_BASE, 'per_run.csv')
    with open(per_run_path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)

    metadata = {
        'purpose': (
            'Sensitivity of the stationary RB-GBMC calculation to the fixed '
            'two-speed relaxation parameter.'
        ),
        'a_seq': A_SEQ,
        'N_seq': N_SEQ,
        'S': S,
        'seeds': seeds,
        'parameters': {
            'A': A, 'nu': NU, 'dt': DT, 'T': T,
            'L': L, 'xc': XC, 'N_out': N_OUT,
        },
        'design': (
            'Only a and N vary. The output grid and physical/numerical '
            'parameters are held fixed. The same seed identifiers are reused '
            'at each a and N; comparisons across N are not particlewise '
            'common-random-number couplings.'
        ),
        'shared_stepper': 'advance_rbgbmc_particles in relaxation_gbmc.py',
        'command': 'python reproduce.py ta',
    }
    with open(os.path.join(OUT_BASE, 'summary.json'), 'w') as stream:
        json.dump({'metadata': metadata, 'results': summaries}, stream, indent=2)

    _savez_deterministic(
        os.path.join(OUT_BASE, 'profiles.npz'), **stored_arrays
    )
    print(f"Wrote {summary_path}, {per_run_path}, summary.json, and profiles.npz")
    return summaries


if __name__ == '__main__':
    run_study()
