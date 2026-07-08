#!/usr/bin/env python3
"""N-refinement convergence study for the Relaxation GBMC Burgers solver.

For each N, runs the stationary-shock IC, measures L2/Linf/rel-L2 error
against the exact solution, and fits the empirical log-log rate against the
O(1/sqrt(N)) Monte Carlo theory.  Writes n_refinement_results.json and
n_refinement_plot.png to --output-dir; see --help for options.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

if sys.platform.startswith("darwin"):
    matplotlib.use("MacOSX")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relaxation_gbmc import simulate_burgers_relaxation_gbmc
from config import SimulationConfig


def exact_stationary_shock(x, nu, x_center, amplitude):
    return -amplitude * np.tanh(amplitude * (x - x_center) / (2.0 * nu))


def generate_run_seeds(base_seed, repeats):
    """Return [base_seed, base_seed+1, ..., base_seed+repeats-1].

    Paired-seed design: repeat index i uses seed base_seed+i for every N value,
    so comparisons across particle counts use the same stochastic realisations.
    Does not use Python hash() so results are stable across processes.
    """
    return [base_seed + i for i in range(repeats)]


def _run_rbmc_one(N, nu, T, dt, L, amplitude, seed=None):
    """Run one RBMC experiment for the stationary-shock IC.

    Returns (x_out, u_out, u_exact, metrics_dict).  seed is forwarded to
    SimulationConfig.seed, which seeds the solver's private RNG
    (np.random.default_rng); the global NumPy RNG is NOT set here and the
    relaxation solver does not use it.
    """
    xc = L / 2.0
    x0 = np.linspace(0.0, L, N)
    u0 = exact_stationary_shock(x0, nu, xc, amplitude)

    ic = list(zip(x0.tolist(), u0.tolist()))
    cfg = SimulationConfig(
        equation_type='burgers',
        domain_type='Finite',
        domain_size=L,
        boundary_conditions={
            'LEFT':  {'type': 'Dirichlet', 'value': 0.0},
            'RIGHT': {'type': 'Dirichlet', 'value': 0.0},
        },
        diff_constant=nu,
        time_step=dt,
        total_time=T,
        num_points=N,
        initial_conditions=ic,
        reaction_term=False,
        burgers_mode='relaxation_gbmc',
        burgers_ic_type='stationary_shock',
        burgers_ic_amplitude=amplitude,
        relaxation_speed_a=2.0,
        seed=seed,
    )

    globs = [{'position': float(p), 'value': [float(v)]} for p, v in ic]
    result = simulate_burgers_relaxation_gbmc(globs, cfg)

    x_out = np.array([g['position'] for g in result])
    u_out = np.array([g['value'][0] for g in result])
    u_exact = exact_stationary_shock(x_out, nu, xc, amplitude)

    dx = float(x_out[1] - x_out[0])
    err = u_out - u_exact
    l2  = float(np.sqrt(np.sum(err ** 2) * dx))
    linf = float(np.max(np.abs(err)))
    l2_ref = float(np.sqrt(np.sum(u_exact ** 2) * dx))
    rel_l2 = l2 / l2_ref if l2_ref > 1e-12 else float('nan')

    return x_out, u_out, u_exact, {
        'N': N, 'l2': l2, 'linf': linf, 'rel_l2': rel_l2,
        'l2_ref': l2_ref, 'seed': seed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="N-refinement study for relaxation GBMC (Burgers stationary shock)",
    )
    parser.add_argument(
        "--n-seq", nargs="+", type=int,
        default=[50, 100, 200, 400, 800],
        metavar="N",
        help="Sequence of N values to test (default: 50 100 200 400 800)",
    )
    parser.add_argument("--nu", type=float, default=0.5, help="Viscosity (default: 0.5)")
    parser.add_argument("--T", type=float, default=0.5, help="Final time (default: 0.5)")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step (default: 0.005)")
    parser.add_argument("--L", type=float, default=4.0, help="Domain size (default: 4.0)")
    parser.add_argument("--amplitude", type=float, default=1.0,
                        help="Shock amplitude A (default: 1.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base RNG seed (default: random); each repeat gets seed+rep")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of runs per N to average over (default: 1)")
    parser.add_argument("--output-dir", default="output/n_refinement_rbmc",
                        help="Output directory (default: output/n_refinement_rbmc)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_seq = sorted(args.n_seq)
    nu, T, dt, L, A = args.nu, args.T, args.dt, args.L, args.amplitude

    print(f"\n{'='*62}")
    print(f"  Relaxation GBMC -- N-refinement convergence study")
    print(f"  IC: stationary shock  A={A:.4g}, nu={nu:.4g}")
    print(f"  T={T}, dt={dt}  ({int(T/dt)} steps), domain [0, {L}]")
    print(f"  N values: {n_seq}")
    print(f"  Repeats per N: {args.repeats}")
    if args.seed is not None:
        _example_seeds = generate_run_seeds(args.seed, args.repeats)
        print(f"  Base seed: {args.seed}")
        print(f"  Repeat seeds: {', '.join(str(s) for s in _example_seeds)}")
        print(f"  Seed pairing across N: enabled  (repeat i => seed {args.seed}+i)")
    else:
        print(f"  Seed: None  (non-reproducible runs)")
    print(f"{'='*62}\n")

    results = []
    base_seed = args.seed

    for N in n_seq:
        l2_runs, linf_runs, rel_l2_runs = [], [], []
        run_seeds = (generate_run_seeds(base_seed, args.repeats)
                     if base_seed is not None else [None] * args.repeats)
        for rep in range(args.repeats):
            run_seed = run_seeds[rep]
            print(f"  N={N:5d}, repeat={rep+1}/{args.repeats}, seed={run_seed} ...",
                  flush=True)
            _, _, _, m = _run_rbmc_one(N, nu, T, dt, L, A, seed=run_seed)
            l2_runs.append(m['l2'])
            linf_runs.append(m['linf'])
            rel_l2_runs.append(m['rel_l2'])

        row = {
            'N': N,
            'base_seed':   base_seed,
            'run_seeds':   run_seeds,
            'l2_mean':     float(np.mean(l2_runs)),
            'l2_std':      float(np.std(l2_runs)),
            'linf_mean':   float(np.mean(linf_runs)),
            'linf_std':    float(np.std(linf_runs)),
            'rel_l2_mean': float(np.mean(rel_l2_runs)),
            'rel_l2_std':  float(np.std(rel_l2_runs)),
            'repeats':     args.repeats,
        }
        results.append(row)
        print(f"    L2={row['l2_mean']:.4f}  Linf={row['linf_mean']:.4f}  "
              f"relL2={row['rel_l2_mean']:.4f}  (std L2={row['l2_std']:.4f})")

    out_json = os.path.join(args.output_dir, "n_refinement_results.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({
            "parameters": {
                "nu": nu, "T": T, "dt": dt, "L": L, "amplitude": A,
                "repeats": args.repeats,
                "base_seed": args.seed,
                "seed_policy": "base_seed + repeat_index (paired across N)",
            },
            "results": results,
        }, fh, indent=2)
    print(f"\n  Saved results -> {out_json}")

    # Fit empirical convergence slope on L2 vs N (log-log)
    N_arr  = np.array([r['N']       for r in results], dtype=float)
    l2_arr = np.array([r['l2_mean'] for r in results], dtype=float)

    valid = np.isfinite(l2_arr) & (l2_arr > 0)
    slope_str = "N/A"
    if valid.sum() >= 2:
        coeffs = np.polyfit(np.log10(N_arr[valid]), np.log10(l2_arr[valid]), 1)
        slope = float(coeffs[0])
        slope_str = f"{slope:.3f}"
        print(f"\n  Empirical L2 slope (log-log): {slope:.4f}  "
              f"(MC theory: -0.50)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Relaxation GBMC: N-refinement  (nu={nu}, T={T}, A={A})\n"
        f"Empirical L2 slope = {slope_str}  (MC theory: -0.50)",
        fontsize=10,
    )

    ax = axes[0]
    ax.loglog(N_arr, l2_arr, 'bo-', lw=1.5, ms=6, label='L2 error (mean)')
    if args.repeats > 1:
        l2_std = np.array([r['l2_std'] for r in results])
        ax.fill_between(N_arr, l2_arr - l2_std, l2_arr + l2_std,
                        alpha=0.25, color='blue', label='±1 std')
    N_ref = np.array([N_arr[0], N_arr[-1]])
    for exp, ls, label in [(-0.5, '--', 'O(N^{-1/2})'), (-1.0, ':', 'O(N^{-1})')]:
        c = l2_arr[0] * (N_arr[0] ** (-exp))
        ax.loglog(N_ref, c * N_ref ** exp, ls, color='gray', lw=1.0, label=label)
    ax.set_xlabel('N  (number of particles)')
    ax.set_ylabel('L2 error')
    ax.set_title('L2 error vs N')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    linf_arr = np.array([r['linf_mean'] for r in results])
    rel_arr  = np.array([r['rel_l2_mean'] for r in results])
    ax.loglog(N_arr, linf_arr, 'rs-', lw=1.4, ms=6, label='Linf error')
    ax.loglog(N_arr, rel_arr,  'g^-', lw=1.4, ms=6, label='rel-L2 error')
    ax.set_xlabel('N  (number of particles)')
    ax.set_ylabel('error')
    ax.set_title('Linf and relative L2 vs N')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(args.output_dir, "n_refinement_plot.png")
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot   -> {out_png}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
