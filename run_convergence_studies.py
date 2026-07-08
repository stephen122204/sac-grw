#!/usr/bin/env python3
"""Systematic N-refinement and dt-refinement convergence studies for all four
GRW/GBMC methods:

  1. Heat GRW             -- N-refinement vs exact error-function solution
  2. Cole-Hopf Burgers    -- N-refinement vs exact stationary-shock solution
  3. Relaxation GBMC      -- N-refinement + dt-refinement vs exact stationary-shock
  4. FitzHugh-Nagumo GRW  -- N-refinement of front-location error vs exact traveling-wave

Results go to --output-dir (default: output/convergence_study/); see --help.
Method flags: heat, cole_hopf, gbmc, fhn, gbmc_dt, all (default: all).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SimulationConfig
from verify_solver import (
    exact_heat_step, exact_burgers_stationary_shock,
    compute_metrics,
)


def _fit_loglog(x_arr, y_arr):
    """Fit y = C * x^alpha in log-log space. Return (alpha, C, r2)."""
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0) & (y_arr > 0)
    if valid.sum() < 2:
        return float('nan'), float('nan'), float('nan')
    lx = np.log10(x_arr[valid])
    ly = np.log10(y_arr[valid])
    coeffs = np.polyfit(lx, ly, 1)
    alpha = float(coeffs[0])
    C = float(10.0 ** coeffs[1])
    ly_fit = np.polyval(coeffs, lx)
    ss_res = float(np.sum((ly - ly_fit) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return alpha, C, r2


def _bootstrap_slope_ci(x_arr, y_arr, n_boot=2000, ci=0.95, rng=None):
    """Bootstrap 95% CI on the log-log slope."""
    if rng is None:
        rng = np.random.default_rng(0)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0) & (y_arr > 0)
    n = int(valid.sum())
    if n < 3:
        return float('nan'), float('nan')
    lx = np.log10(x_arr[valid])
    ly = np.log10(y_arr[valid])
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            c = np.polyfit(lx[idx], ly[idx], 1)
            slopes.append(c[0])
        except Exception:
            pass
    if not slopes:
        return float('nan'), float('nan')
    lo = float(np.percentile(slopes, 100 * (1 - ci) / 2))
    hi = float(np.percentile(slopes, 100 * (1 + ci) / 2))
    return lo, hi


def _save_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda v: None)
    print(f"  [save] {path}")


def _convergence_plot(
    n_arr, y_mean, y_std, y_label, title, output_path,
    slope, slope_lo, slope_hi, ref_slopes=None,
    x_label="N  (particles/globs)",
):
    """Log-log convergence plot with ±1 std band, fitted slope, reference slopes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(n_arr, y_mean, 'bo-', lw=1.8, ms=6, label=f'{y_label} (mean over repeats)')
    if y_std is not None and np.any(np.array(y_std) > 0):
        lo = np.maximum(np.array(y_mean) - np.array(y_std), 1e-10)
        hi = np.array(y_mean) + np.array(y_std)
        ax.fill_between(n_arr, lo, hi, alpha=0.2, color='blue', label='±1 std')

    # Reference slope lines anchored at first point
    n_ref = np.array([n_arr[0], n_arr[-1]], dtype=float)
    if ref_slopes is None:
        ref_slopes = [(-0.5, '--', r'$O(N^{-1/2})$'), (-1.0, ':', r'$O(N^{-1})$')]
    for (exp, ls, lbl) in ref_slopes:
        c = y_mean[0] * (n_arr[0] ** (-exp))
        ax.loglog(n_ref, c * n_ref ** exp, ls, color='gray', lw=1.2, label=lbl)

    ci_str = ""
    if not np.isnan(slope_lo) and not np.isnan(slope_hi):
        ci_str = f"  95% CI [{slope_lo:.3f}, {slope_hi:.3f}]"
    ax.set_title(f"{title}\nFitted slope = {slope:.3f}{ci_str}", fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [plot] {output_path}")


# 1. Heat GRW N-refinement

def _run_heat_one(N, alpha, T, dt, L, x0, uL, uR, seed):
    """Run one Heat GRW instance; return error metrics dict."""
    np.random.seed(seed)
    from simulation import simulate_heat_equation

    # Step IC: all N globs at x0 with weight (uR-uL)/N
    jump = float(uR - uL)
    ic = [(float(x0), float(jump / N))] * N
    cfg = SimulationConfig(
        equation_type='heat',
        domain_type='Finite',
        domain_size=L,
        boundary_conditions={
            'LEFT':  {'type': 'Dirichlet', 'value': uL},
            'RIGHT': {'type': 'Dirichlet', 'value': uR},
        },
        diff_constant=alpha,
        time_step=dt,
        total_time=T,
        num_points=N,
        initial_conditions=ic,
        reaction_term=False,
    )
    globs = [{'position': float(p), 'value': float(v)} for p, v in ic]
    result = simulate_heat_equation(globs, cfg)

    positions = np.array([g['position'] for g in result])
    values    = np.array([g['value']    for g in result])

    # Reconstruct u on a uniform grid
    nbins = max(200, min(600, N // 10))
    edges = np.linspace(0.0, L, nbins + 1)
    bin_w, _ = np.histogram(positions, bins=edges, weights=values)
    u_num = uL + np.cumsum(bin_w)
    x_grid = 0.5 * (edges[:-1] + edges[1:])
    dx = float(edges[1] - edges[0])

    u_exact = exact_heat_step(x_grid, T, x0, uL, uR, alpha)
    m = compute_metrics(u_num, u_exact, dx)
    m['N'] = N
    m['seed'] = seed
    return m


def run_heat_n_refinement(output_dir, n_seq, repeats, base_seed, alpha, T, dt, L, x0, uL, uR):
    print(f"\n{'='*62}")
    print(f"  Heat GRW  N-refinement")
    print(f"  alpha={alpha}, T={T}, dt={dt}, L={L}, x0={x0}")
    print(f"  N={n_seq}, repeats={repeats}, base_seed={base_seed}")
    print(f"{'='*62}")

    results = []
    for N in n_seq:
        l2_runs, linf_runs, rel_runs = [], [], []
        for rep in range(repeats):
            seed = base_seed + rep if base_seed is not None else None
            t0 = time.perf_counter()
            m = _run_heat_one(N, alpha, T, dt, L, x0, uL, uR, seed)
            elapsed = time.perf_counter() - t0
            l2_runs.append(m['l2'])
            linf_runs.append(m['linf'])
            rel_runs.append(m['rel_l2'] or float('nan'))
            print(f"    N={N:6d}  rep={rep+1}/{repeats}  L2={m['l2']:.5f}  t={elapsed:.2f}s")

        row = {
            'N': N,
            'l2_mean':     float(np.nanmean(l2_runs)),
            'l2_std':      float(np.nanstd(l2_runs)),
            'linf_mean':   float(np.nanmean(linf_runs)),
            'linf_std':    float(np.nanstd(linf_runs)),
            'rel_l2_mean': float(np.nanmean(rel_runs)),
            'rel_l2_std':  float(np.nanstd(rel_runs)),
            'repeats': repeats,
            'run_seeds': ([base_seed + r for r in range(repeats)]
                          if base_seed is not None else None),
        }
        results.append(row)
        print(f"  -> L2={row['l2_mean']:.5f} ± {row['l2_std']:.5f}")

    n_arr   = np.array([r['N'] for r in results], dtype=float)
    l2_arr  = np.array([r['l2_mean'] for r in results])
    l2_std  = np.array([r['l2_std'] for r in results])
    slope, C, r2 = _fit_loglog(n_arr, l2_arr)
    rng_ci = np.random.default_rng(1)
    slo, shi = _bootstrap_slope_ci(n_arr, l2_arr, rng=rng_ci)
    print(f"\n  Fitted L2 slope: {slope:.4f}  95% CI [{slo:.4f}, {shi:.4f}]  R²={r2:.4f}")

    out = {
        'method': 'heat_grw',
        'parameters': {'alpha': alpha, 'T': T, 'dt': dt, 'L': L, 'x0': x0,
                       'uL': uL, 'uR': uR, 'repeats': repeats, 'base_seed': base_seed},
        'results': results,
        'fit': {'l2_slope': slope, 'l2_slope_ci_lo': slo, 'l2_slope_ci_hi': shi,
                'C': C, 'r2': r2},
    }
    _save_json(out, os.path.join(output_dir, 'heat', 'n_refinement.json'))

    _convergence_plot(
        n_arr, l2_arr, l2_std,
        y_label='L2 error',
        title=f'Heat GRW: N-refinement  (alpha={alpha}, T={T})',
        output_path=os.path.join(output_dir, 'heat', 'n_refinement_plot.png'),
        slope=slope, slope_lo=slo, slope_hi=shi,
        ref_slopes=[(-0.5, '--', r'$O(N^{-1/2})$'), (-1.0, ':', r'$O(N^{-1})$')],
    )
    return out


# 2. Cole-Hopf Burgers N-refinement

def _run_cole_hopf_one(N, nu, T, dt, L, amplitude, seed):
    """Run one Cole-Hopf GRW instance on the stationary shock."""
    np.random.seed(seed)
    from simulation import simulate_burgers_cole_hopf_grw
    from config import generate_burgers_stationary_shock_ic

    xc = L / 2.0
    ic = generate_burgers_stationary_shock_ic(L, N, nu, x_center=xc, amplitude=amplitude)
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
        burgers_mode='cole_hopf_grw',
        burgers_ic_type='stationary_shock',
        burgers_ic_amplitude=amplitude,
    )
    globs = [{'position': float(p), 'value': [float(v)]} for p, v in ic]
    result = simulate_burgers_cole_hopf_grw(globs, cfg)

    x_out = np.array([g['position'] for g in result])
    u_out = np.array([g['value'][0] for g in result])
    order = np.argsort(x_out)
    x_out, u_out = x_out[order], u_out[order]

    x_grid = np.linspace(0.0, L, N)
    u_exact = exact_burgers_stationary_shock(x_grid, nu, x_center=xc, amplitude=amplitude)
    u_num = np.interp(x_grid, x_out, u_out)
    dx = float(x_grid[1] - x_grid[0])
    m = compute_metrics(u_num, u_exact, dx)
    m['N'] = N
    m['seed'] = seed
    return m


def run_cole_hopf_n_refinement(output_dir, n_seq, repeats, base_seed, nu, T, dt, L, amplitude):
    print(f"\n{'='*62}")
    print(f"  Cole-Hopf Burgers GRW  N-refinement")
    print(f"  nu={nu}, T={T}, dt={dt}, L={L}, A={amplitude}")
    print(f"  N={n_seq}, repeats={repeats}, base_seed={base_seed}")
    print(f"{'='*62}")

    results = []
    for N in n_seq:
        l2_runs, linf_runs, rel_runs = [], [], []
        for rep in range(repeats):
            seed = base_seed + rep if base_seed is not None else None
            t0 = time.perf_counter()
            m = _run_cole_hopf_one(N, nu, T, dt, L, amplitude, seed)
            elapsed = time.perf_counter() - t0
            l2_runs.append(m['l2'])
            linf_runs.append(m['linf'])
            rel_runs.append(m['rel_l2'] or float('nan'))
            print(f"    N={N:5d}  rep={rep+1}/{repeats}  L2={m['l2']:.5f}  t={elapsed:.2f}s")

        row = {
            'N': N,
            'l2_mean':     float(np.nanmean(l2_runs)),
            'l2_std':      float(np.nanstd(l2_runs)),
            'linf_mean':   float(np.nanmean(linf_runs)),
            'linf_std':    float(np.nanstd(linf_runs)),
            'rel_l2_mean': float(np.nanmean(rel_runs)),
            'rel_l2_std':  float(np.nanstd(rel_runs)),
            'repeats': repeats,
            'run_seeds': ([base_seed + r for r in range(repeats)]
                          if base_seed is not None else None),
        }
        results.append(row)
        print(f"  -> L2={row['l2_mean']:.5f} ± {row['l2_std']:.5f}")

    n_arr  = np.array([r['N'] for r in results], dtype=float)
    l2_arr = np.array([r['l2_mean'] for r in results])
    l2_std = np.array([r['l2_std'] for r in results])
    slope, C, r2 = _fit_loglog(n_arr, l2_arr)
    rng_ci = np.random.default_rng(2)
    slo, shi = _bootstrap_slope_ci(n_arr, l2_arr, rng=rng_ci)
    print(f"\n  Fitted L2 slope: {slope:.4f}  95% CI [{slo:.4f}, {shi:.4f}]  R²={r2:.4f}")

    out = {
        'method': 'cole_hopf_grw',
        'parameters': {'nu': nu, 'T': T, 'dt': dt, 'L': L, 'amplitude': amplitude,
                       'repeats': repeats, 'base_seed': base_seed},
        'results': results,
        'fit': {'l2_slope': slope, 'l2_slope_ci_lo': slo, 'l2_slope_ci_hi': shi,
                'C': C, 'r2': r2},
    }
    _save_json(out, os.path.join(output_dir, 'cole_hopf', 'n_refinement.json'))

    _convergence_plot(
        n_arr, l2_arr, l2_std,
        y_label='L2 error',
        title=f'Cole-Hopf Burgers GRW: N-refinement  (nu={nu}, T={T})',
        output_path=os.path.join(output_dir, 'cole_hopf', 'n_refinement_plot.png'),
        slope=slope, slope_lo=slo, slope_hi=shi,
    )
    return out


# 3a. Relaxation GBMC N-refinement

def _run_gbmc_one(N, nu, T, dt, L, amplitude, seed, relaxation_a=2.0):
    """Run one Relaxation GBMC instance on the stationary shock."""
    from relaxation_gbmc import simulate_burgers_relaxation_gbmc
    from config import generate_burgers_stationary_shock_ic

    xc = L / 2.0
    ic = generate_burgers_stationary_shock_ic(L, N, nu, x_center=xc, amplitude=amplitude)
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
        relaxation_speed_a=relaxation_a,
        relaxation_domain_mode='whole_line',
        seed=seed,
        burgers_ic_center=xc,
    )
    globs = [{'position': float(p), 'value': [float(v)]} for p, v in ic]
    result = simulate_burgers_relaxation_gbmc(globs, cfg)

    x_out = np.array([g['position'] for g in result])
    u_out = np.array([g['value'][0] for g in result])
    order = np.argsort(x_out)
    x_out, u_out = x_out[order], u_out[order]

    x_grid = np.linspace(0.0, L, N)
    u_exact = exact_burgers_stationary_shock(x_grid, nu, x_center=xc, amplitude=amplitude)
    u_num = np.interp(x_grid, x_out, u_out)
    dx = float(x_grid[1] - x_grid[0])
    m = compute_metrics(u_num, u_exact, dx)
    m['N'] = N
    m['seed'] = seed
    return m


def run_gbmc_n_refinement(output_dir, n_seq, repeats, base_seed, nu, T, dt, L, amplitude, a=2.0):
    print(f"\n{'='*62}")
    print(f"  Relaxation GBMC  N-refinement")
    print(f"  nu={nu}, T={T}, dt={dt}, L={L}, A={amplitude}, a={a}")
    print(f"  N={n_seq}, repeats={repeats}, base_seed={base_seed}")
    print(f"{'='*62}")

    results = []
    for N in n_seq:
        l2_runs, linf_runs, rel_runs = [], [], []
        for rep in range(repeats):
            seed = base_seed + rep if base_seed is not None else None
            t0 = time.perf_counter()
            m = _run_gbmc_one(N, nu, T, dt, L, amplitude, seed, a)
            elapsed = time.perf_counter() - t0
            l2_runs.append(m['l2'])
            linf_runs.append(m['linf'])
            rel_runs.append(m['rel_l2'] or float('nan'))
            print(f"    N={N:5d}  rep={rep+1}/{repeats}  L2={m['l2']:.5f}  t={elapsed:.2f}s")

        row = {
            'N': N,
            'l2_mean':     float(np.nanmean(l2_runs)),
            'l2_std':      float(np.nanstd(l2_runs)),
            'linf_mean':   float(np.nanmean(linf_runs)),
            'linf_std':    float(np.nanstd(linf_runs)),
            'rel_l2_mean': float(np.nanmean(rel_runs)),
            'rel_l2_std':  float(np.nanstd(rel_runs)),
            'repeats': repeats,
            'run_seeds': ([base_seed + r for r in range(repeats)]
                          if base_seed is not None else None),
        }
        results.append(row)
        print(f"  -> L2={row['l2_mean']:.5f} ± {row['l2_std']:.5f}")

    n_arr  = np.array([r['N'] for r in results], dtype=float)
    l2_arr = np.array([r['l2_mean'] for r in results])
    l2_std = np.array([r['l2_std'] for r in results])
    slope, C, r2 = _fit_loglog(n_arr, l2_arr)
    rng_ci = np.random.default_rng(3)
    slo, shi = _bootstrap_slope_ci(n_arr, l2_arr, rng=rng_ci)
    print(f"\n  Fitted L2 slope: {slope:.4f}  95% CI [{slo:.4f}, {shi:.4f}]  R²={r2:.4f}")

    out = {
        'method': 'relaxation_gbmc',
        'parameters': {'nu': nu, 'T': T, 'dt': dt, 'L': L, 'amplitude': amplitude,
                       'a': a, 'repeats': repeats, 'base_seed': base_seed},
        'results': results,
        'fit': {'l2_slope': slope, 'l2_slope_ci_lo': slo, 'l2_slope_ci_hi': shi,
                'C': C, 'r2': r2},
    }
    _save_json(out, os.path.join(output_dir, 'gbmc', 'n_refinement.json'))

    _convergence_plot(
        n_arr, l2_arr, l2_std,
        y_label='L2 error',
        title=f'Relaxation GBMC: N-refinement  (nu={nu}, T={T}, a={a})',
        output_path=os.path.join(output_dir, 'gbmc', 'n_refinement_plot.png'),
        slope=slope, slope_lo=slo, slope_hi=shi,
    )
    return out


# 3b. Relaxation GBMC dt-refinement

def run_gbmc_dt_refinement(output_dir, dt_seq, repeats, base_seed,
                            nu, T, L, N_fixed, amplitude, a=2.0):
    """
    Vary dt at fixed N; estimate bias contribution from Lie splitting.
    The stationary-shock IC is time-independent, so bias = mean(error) - noise_floor.
    """
    print(f"\n{'='*62}")
    print(f"  Relaxation GBMC  dt-refinement")
    print(f"  nu={nu}, T={T}, L={L}, N={N_fixed}, A={amplitude}, a={a}")
    print(f"  dt={dt_seq}, repeats={repeats}, base_seed={base_seed}")
    print(f"{'='*62}")

    # Validate T/dt is integer for all dt values
    valid_dt = []
    for dt in dt_seq:
        n_steps = T / dt
        if abs(n_steps - round(n_steps)) > 1e-9:
            print(f"  WARNING: T/dt = {n_steps:.4f} is not integer for dt={dt}; skipping.")
        else:
            valid_dt.append(dt)
    dt_seq = valid_dt

    results = []
    for dt in dt_seq:
        l2_runs, linf_runs, rel_runs = [], [], []
        n_steps = int(round(T / dt))
        for rep in range(repeats):
            seed = base_seed + rep if base_seed is not None else None
            t0 = time.perf_counter()
            m = _run_gbmc_one(N_fixed, nu, T, dt, L, amplitude, seed, a)
            elapsed = time.perf_counter() - t0
            l2_runs.append(m['l2'])
            linf_runs.append(m['linf'])
            rel_runs.append(m['rel_l2'] or float('nan'))
            print(f"    dt={dt:.5f} ({n_steps} steps)  rep={rep+1}/{repeats}"
                  f"  L2={m['l2']:.5f}  t={elapsed:.2f}s")

        row = {
            'dt': dt,
            'n_steps': n_steps,
            'l2_mean':     float(np.nanmean(l2_runs)),
            'l2_std':      float(np.nanstd(l2_runs)),
            'linf_mean':   float(np.nanmean(linf_runs)),
            'linf_std':    float(np.nanstd(linf_runs)),
            'rel_l2_mean': float(np.nanmean(rel_runs)),
            'rel_l2_std':  float(np.nanstd(rel_runs)),
            'repeats': repeats,
        }
        results.append(row)
        print(f"  -> L2={row['l2_mean']:.5f} ± {row['l2_std']:.5f}")

    dt_arr = np.array([r['dt'] for r in results], dtype=float)
    l2_arr = np.array([r['l2_mean'] for r in results])
    l2_std = np.array([r['l2_std'] for r in results])
    slope, C, r2 = _fit_loglog(dt_arr, l2_arr)
    rng_ci = np.random.default_rng(4)
    slo, shi = _bootstrap_slope_ci(dt_arr, l2_arr, rng=rng_ci)
    print(f"\n  Fitted L2 slope vs dt: {slope:.4f}  95% CI [{slo:.4f}, {shi:.4f}]  R²={r2:.4f}")

    out = {
        'method': 'relaxation_gbmc',
        'study': 'dt_refinement',
        'parameters': {'nu': nu, 'T': T, 'L': L, 'N_fixed': N_fixed,
                       'amplitude': amplitude, 'a': a,
                       'repeats': repeats, 'base_seed': base_seed},
        'results': results,
        'fit': {'l2_slope_vs_dt': slope, 'l2_slope_ci_lo': slo, 'l2_slope_ci_hi': shi,
                'C': C, 'r2': r2,
                'interpretation': (
                    'Near-zero slope means dominant error is particle noise (dt-insensitive). '
                    'Positive slope (error increases with dt) would indicate Lie-splitting bias.'
                )},
    }
    _save_json(out, os.path.join(output_dir, 'gbmc', 'dt_refinement.json'))

    # dt-refinement plot (x-axis = dt)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(dt_arr, l2_arr, 'rs-', lw=1.8, ms=6, label='L2 error (mean)')
    lo = np.maximum(l2_arr - l2_std, 1e-10)
    ax.fill_between(dt_arr, lo, l2_arr + l2_std, alpha=0.2, color='red', label='±1 std')
    dt_ref = np.array([dt_arr[0], dt_arr[-1]])
    for (exp, ls, lbl) in [(1.0, '--', r'$O(\Delta t)$'), (0.5, ':', r'$O(\Delta t^{1/2})$')]:
        c = l2_arr[0] * (dt_arr[0] ** (-exp))
        ax.loglog(dt_ref, c * dt_ref ** exp, ls, color='gray', lw=1.2, label=lbl)
    ci_str = f"  95% CI [{slo:.3f}, {shi:.3f}]" if not np.isnan(slo) else ""
    ax.set_title(f'Relaxation GBMC: dt-refinement  (N={N_fixed}, nu={nu}, T={T})\n'
                 f'Fitted slope vs dt = {slope:.3f}{ci_str}', fontsize=10)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel('L2 error')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    p = os.path.join(output_dir, 'gbmc', 'dt_refinement_plot.png')
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [plot] {p}")
    return out


# 4. FHN GRW N-refinement

def _run_fhn_one(N, D, T, dt, L, a_param, x_center, seed):
    """Run one FHN scalar GRW instance; return front-location error at t=T."""
    np.random.seed(seed)
    from simulation import simulate_fitzhugh_nagumo_grw
    from config import generate_fhn_steady_ic

    ic = generate_fhn_steady_ic(N, a_param, x_center)
    cfg = SimulationConfig(
        equation_type='fitzhugh-nagumo',
        domain_type='Finite',
        domain_size=L,
        boundary_conditions={
            'LEFT':  {'type': 'Neumann', 'value': 0.0},
            'RIGHT': {'type': 'Neumann', 'value': 0.0},
        },
        diff_constant=D,
        time_step=dt,
        total_time=T,
        num_points=N,
        initial_conditions=ic,
        reaction_term=True,
        a=a_param,
    )
    globs = [{'position': float(p), 'value': float(v)} for p, v in ic]
    result = simulate_fitzhugh_nagumo_grw(globs, cfg)

    x = np.array([g['position'] for g in result])
    w = np.array([
        (float(g['value'][0]) if isinstance(g['value'], (list, tuple)) else float(g['value']))
        for g in result
    ])
    order = np.argsort(x)
    x, w = x[order], w[order]
    u_cum = np.cumsum(w)

    # Front: u = 0.5 crossing
    idx = int(np.clip(np.searchsorted(u_cum, 0.5), 0, len(x) - 1))
    front_num = float(x[idx])

    # Exact front at T
    theta = float(np.sqrt(2.0) * (0.5 - a_param))
    front_exact = float(x_center - theta * T)
    front_error = abs(front_num - front_exact)

    return {
        'N': N, 'seed': seed,
        'front_num': front_num, 'front_exact': front_exact,
        'front_error': front_error,
        'theta': theta,
    }


def run_fhn_n_refinement(output_dir, n_seq, repeats, base_seed, D, T, dt, L, a_param, x_center):
    print(f"\n{'='*62}")
    print(f"  FHN scalar GRW  N-refinement")
    print(f"  D={D}, T={T}, dt={dt}, L={L}, a={a_param}, xc={x_center}")
    print(f"  N={n_seq}, repeats={repeats}, base_seed={base_seed}")
    print(f"{'='*62}")

    results = []
    for N in n_seq:
        fe_runs = []
        for rep in range(repeats):
            seed = base_seed + rep if base_seed is not None else None
            t0 = time.perf_counter()
            m = _run_fhn_one(N, D, T, dt, L, a_param, x_center, seed)
            elapsed = time.perf_counter() - t0
            fe_runs.append(m['front_error'])
            print(f"    N={N:5d}  rep={rep+1}/{repeats}  "
                  f"front_err={m['front_error']:.5f}  t={elapsed:.2f}s")

        row = {
            'N': N,
            'front_error_mean': float(np.nanmean(fe_runs)),
            'front_error_std':  float(np.nanstd(fe_runs)),
            'repeats': repeats,
            'run_seeds': ([base_seed + r for r in range(repeats)]
                          if base_seed is not None else None),
        }
        results.append(row)
        print(f"  -> front_err={row['front_error_mean']:.5f} ± {row['front_error_std']:.5f}")

    n_arr  = np.array([r['N'] for r in results], dtype=float)
    fe_arr = np.array([r['front_error_mean'] for r in results])
    fe_std = np.array([r['front_error_std'] for r in results])
    slope, C, r2 = _fit_loglog(n_arr, fe_arr)
    rng_ci = np.random.default_rng(5)
    slo, shi = _bootstrap_slope_ci(n_arr, fe_arr, rng=rng_ci)
    theta = float(np.sqrt(2.0) * (0.5 - a_param))
    print(f"\n  theta = {theta:.4f}")
    print(f"  Fitted front-error slope: {slope:.4f}  95% CI [{slo:.4f}, {shi:.4f}]  R²={r2:.4f}")

    out = {
        'method': 'fhn_grw',
        'parameters': {'D': D, 'T': T, 'dt': dt, 'L': L, 'a': a_param,
                       'x_center': x_center, 'theta': theta,
                       'repeats': repeats, 'base_seed': base_seed},
        'results': results,
        'fit': {'front_error_slope': slope, 'slope_ci_lo': slo, 'slope_ci_hi': shi,
                'C': C, 'r2': r2},
    }
    _save_json(out, os.path.join(output_dir, 'fhn', 'n_refinement.json'))

    _convergence_plot(
        n_arr, fe_arr, fe_std,
        y_label='|front error|  (m)',
        title=f'FHN scalar GRW: N-refinement  (D={D}, T={T}, a={a_param})',
        output_path=os.path.join(output_dir, 'fhn', 'n_refinement_plot.png'),
        slope=slope, slope_lo=slo, slope_hi=shi,
        ref_slopes=[(-0.5, '--', r'$O(N^{-1/2})$'), (-1.0, ':', r'$O(N^{-1})$')],
    )
    return out


# 5. Cole-Hopf traveling-wave verification (not a convergence study)

def run_cole_hopf_traveling_wave(output_dir, nu, T, dt, L, N):
    """Single run of Cole-Hopf GRW on the traveling-wave IC; save comparison plot."""
    print(f"\n{'='*62}")
    print(f"  Cole-Hopf Burgers GRW  Traveling-wave verification")
    print(f"  nu={nu}, T={T}, dt={dt}, L={L}, N={N}")
    print(f"{'='*62}")
    np.random.seed(42)
    from simulation import simulate_burgers_cole_hopf_grw
    from verify_solver import exact_burgers_traveling_wave
    from config import generate_burgers_traveling_wave_ic

    xc = L / 2.0
    ic = generate_burgers_traveling_wave_ic(L, N, nu, x_center=xc)
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
        burgers_mode='cole_hopf_grw',
        burgers_ic_type='traveling_wave',
    )
    globs = [{'position': float(p), 'value': [float(v)]} for p, v in ic]
    result = simulate_burgers_cole_hopf_grw(globs, cfg)

    x_out = np.array([g['position'] for g in result])
    u_out = np.array([g['value'][0] for g in result])
    order = np.argsort(x_out)
    x_out, u_out = x_out[order], u_out[order]

    x_grid = np.linspace(0.0, L, N)
    u_exact = exact_burgers_traveling_wave(x_grid, T, nu, x_center=xc)
    u_num = np.interp(x_grid, x_out, u_out)
    dx = float(x_grid[1] - x_grid[0])
    m = compute_metrics(u_num, u_exact, dx)

    print(f"  L2={m['l2']:.5f}  relL2={m['rel_l2']:.5f}  Linf={m['linf']:.5f}")

    # Wave-centre tracking
    u_mid = float(1.0 - 2.0 * np.sqrt(nu))
    wave_loc_num   = float(x_grid[np.argmin(np.abs(u_num   - u_mid))])
    wave_loc_exact = float(x_grid[np.argmin(np.abs(u_exact - u_mid))])
    wave_speed_err = abs(wave_loc_num - wave_loc_exact)
    print(f"  Wave center: GRW={wave_loc_num:.4f}  exact={wave_loc_exact:.4f}  "
          f"diff={wave_speed_err:.5f}")
    m.update({'wave_loc_num': wave_loc_num, 'wave_loc_exact': wave_loc_exact,
              'wave_speed_error': wave_speed_err,
              'N': N, 'nu': nu, 'T': T, 'dt': dt})

    _save_json(m, os.path.join(output_dir, 'cole_hopf', 'traveling_wave_verification.json'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(x_grid, u_exact, 'k-', lw=2, label='Exact traveling wave')
    ax1.plot(x_grid, u_num, 'r--', lw=1.5, label=f'Cole-Hopf GRW (N={N})')
    ax1.set_title(f'Cole-Hopf GRW vs exact traveling wave  (nu={nu}, T={T})\n'
                  f'L2={m["l2"]:.4f}  relL2={m["rel_l2"]:.4f}  wave_err={wave_speed_err:.4f}')
    ax1.set_xlabel('x'); ax1.set_ylabel('u'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(x_grid, u_num - u_exact, color='steelblue', lw=1.5)
    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.set_title('Pointwise error'); ax2.set_xlabel('x'); ax2.set_ylabel('error')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(output_dir, 'cole_hopf', 'traveling_wave_plot.png')
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [plot] {p}")
    return m


# 6. Cole-Hopf vs GBMC comparison at matched N

def run_method_comparison(output_dir, n_seq, nu, T, dt, L, amplitude, repeats, base_seed, a=2.0):
    """Compare Cole-Hopf GRW vs Relaxation GBMC on the stationary shock at matched N."""
    print(f"\n{'='*62}")
    print(f"  Cole-Hopf vs GBMC comparison  (stationary shock)")
    print(f"  nu={nu}, T={T}, dt={dt}, L={L}, A={amplitude}, a={a}")
    print(f"{'='*62}")

    rows = []
    for N in n_seq:
        # Cole-Hopf
        ch_l2, ch_linf = [], []
        for rep in range(repeats):
            seed = base_seed + rep if base_seed is not None else None
            m = _run_cole_hopf_one(N, nu, T, dt, L, amplitude, seed)
            ch_l2.append(m['l2']); ch_linf.append(m['linf'])

        # GBMC
        gb_l2, gb_linf = [], []
        for rep in range(repeats):
            seed = base_seed + rep if base_seed is not None else None
            m = _run_gbmc_one(N, nu, T, dt, L, amplitude, seed, a)
            gb_l2.append(m['l2']); gb_linf.append(m['linf'])

        rows.append({
            'N': N,
            'cole_hopf_l2_mean':   float(np.mean(ch_l2)),
            'cole_hopf_l2_std':    float(np.std(ch_l2)),
            'cole_hopf_linf_mean': float(np.mean(ch_linf)),
            'gbmc_l2_mean':        float(np.mean(gb_l2)),
            'gbmc_l2_std':         float(np.std(gb_l2)),
            'gbmc_linf_mean':      float(np.mean(gb_linf)),
            'repeats': repeats,
        })
        print(f"  N={N:5d}  CH L2={rows[-1]['cole_hopf_l2_mean']:.5f}"
              f"  GBMC L2={rows[-1]['gbmc_l2_mean']:.5f}")

    out = {
        'study': 'cole_hopf_vs_gbmc',
        'parameters': {'nu': nu, 'T': T, 'dt': dt, 'L': L, 'amplitude': amplitude, 'a': a,
                       'repeats': repeats, 'base_seed': base_seed},
        'results': rows,
    }
    _save_json(out, os.path.join(output_dir, 'comparison', 'cole_hopf_vs_gbmc.json'))

    # Comparison plot
    fig, ax = plt.subplots(figsize=(7, 5))
    n_arr = np.array([r['N'] for r in rows], dtype=float)
    ch_arr = np.array([r['cole_hopf_l2_mean'] for r in rows])
    gb_arr = np.array([r['gbmc_l2_mean'] for r in rows])
    ax.loglog(n_arr, ch_arr, 'bs-', lw=1.8, ms=6, label='Cole-Hopf GRW')
    ax.loglog(n_arr, gb_arr, 'ro-', lw=1.8, ms=6, label='Relaxation GBMC')
    n_ref = np.array([n_arr[0], n_arr[-1]])
    c0 = ch_arr[0] * n_arr[0] ** 0.5
    ax.loglog(n_ref, c0 * n_ref ** (-0.5), 'k--', lw=1.2, label=r'$O(N^{-1/2})$')
    ax.set_title(f'Cole-Hopf GRW vs Relaxation GBMC: L2 error\n'
                 f'(nu={nu}, T={T}, stationary shock)')
    ax.set_xlabel('N'); ax.set_ylabel('L2 error')
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    p = os.path.join(output_dir, 'comparison', 'cole_hopf_vs_gbmc_plot.png')
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [plot] {p}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Systematic convergence studies for all GRW/GBMC methods",
    )
    parser.add_argument('--method', default='all',
                        choices=['heat', 'cole_hopf', 'gbmc', 'gbmc_dt', 'fhn', 'compare', 'all'],
                        help='Which study to run (default: all)')
    parser.add_argument('--output-dir', default='output/convergence_study')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    # Heat params
    parser.add_argument('--heat-n-seq', nargs='+', type=int,
                        default=[500, 1000, 2000, 5000, 10000, 20000])
    # Burgers/GBMC params
    parser.add_argument('--burgers-n-seq', nargs='+', type=int,
                        default=[50, 100, 200, 400, 800, 1600])
    parser.add_argument('--nu', type=float, default=0.5)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--L', type=float, default=4.0)
    parser.add_argument('--amplitude', type=float, default=1.0)
    parser.add_argument('--a', type=float, default=2.0, help='Relaxation speed a for GBMC')
    # dt-refinement params
    parser.add_argument('--gbmc-dt-seq', nargs='+', type=float,
                        default=[0.05, 0.025, 0.01, 0.005, 0.0025])
    parser.add_argument('--gbmc-dt-N', type=int, default=800,
                        help='Fixed N for dt-refinement (default: 800)')
    # FHN params
    parser.add_argument('--fhn-n-seq', nargs='+', type=int,
                        default=[100, 200, 500, 1000, 2000])
    parser.add_argument('--fhn-T', type=float, default=9.0)
    parser.add_argument('--fhn-dt', type=float, default=0.01)
    parser.add_argument('--fhn-L', type=float, default=30.0)
    parser.add_argument('--fhn-a', type=float, default=0.25)
    # Comparison params
    parser.add_argument('--compare-n-seq', nargs='+', type=int,
                        default=[100, 200, 400, 800])
    args = parser.parse_args()

    out = args.output_dir
    run_all = args.method == 'all'

    t_total = time.perf_counter()

    results_manifest = {}

    # Heat
    if run_all or args.method == 'heat':
        r = run_heat_n_refinement(
            out, args.heat_n_seq, args.repeats, args.seed,
            alpha=0.1, T=0.5, dt=0.001, L=10.0, x0=5.0, uL=0.0, uR=1.0,
        )
        results_manifest['heat_n_refinement'] = {
            'json': os.path.join(out, 'heat', 'n_refinement.json'),
            'plot': os.path.join(out, 'heat', 'n_refinement_plot.png'),
            'l2_slope': r['fit']['l2_slope'],
            'l2_slope_ci': [r['fit']['l2_slope_ci_lo'], r['fit']['l2_slope_ci_hi']],
        }

    # Cole-Hopf Burgers
    if run_all or args.method == 'cole_hopf':
        r = run_cole_hopf_n_refinement(
            out, args.burgers_n_seq, args.repeats, args.seed,
            nu=args.nu, T=args.T, dt=args.dt, L=args.L, amplitude=args.amplitude,
        )
        results_manifest['cole_hopf_n_refinement'] = {
            'json': os.path.join(out, 'cole_hopf', 'n_refinement.json'),
            'plot': os.path.join(out, 'cole_hopf', 'n_refinement_plot.png'),
            'l2_slope': r['fit']['l2_slope'],
            'l2_slope_ci': [r['fit']['l2_slope_ci_lo'], r['fit']['l2_slope_ci_hi']],
        }
        # Also run traveling wave verification
        r_tw = run_cole_hopf_traveling_wave(out, nu=args.nu, T=0.3, dt=args.dt, L=8.0, N=400)
        results_manifest['cole_hopf_traveling_wave'] = {
            'json': os.path.join(out, 'cole_hopf', 'traveling_wave_verification.json'),
            'plot': os.path.join(out, 'cole_hopf', 'traveling_wave_plot.png'),
            'l2': r_tw['l2'], 'wave_speed_error': r_tw['wave_speed_error'],
        }

    # Relaxation GBMC N-refinement
    if run_all or args.method == 'gbmc':
        r = run_gbmc_n_refinement(
            out, args.burgers_n_seq, args.repeats, args.seed,
            nu=args.nu, T=args.T, dt=args.dt, L=args.L,
            amplitude=args.amplitude, a=args.a,
        )
        results_manifest['gbmc_n_refinement'] = {
            'json': os.path.join(out, 'gbmc', 'n_refinement.json'),
            'plot': os.path.join(out, 'gbmc', 'n_refinement_plot.png'),
            'l2_slope': r['fit']['l2_slope'],
            'l2_slope_ci': [r['fit']['l2_slope_ci_lo'], r['fit']['l2_slope_ci_hi']],
        }

    # Relaxation GBMC dt-refinement
    if run_all or args.method == 'gbmc_dt':
        r = run_gbmc_dt_refinement(
            out, args.gbmc_dt_seq, args.repeats, args.seed,
            nu=args.nu, T=args.T, L=args.L,
            N_fixed=args.gbmc_dt_N, amplitude=args.amplitude, a=args.a,
        )
        results_manifest['gbmc_dt_refinement'] = {
            'json': os.path.join(out, 'gbmc', 'dt_refinement.json'),
            'plot': os.path.join(out, 'gbmc', 'dt_refinement_plot.png'),
            'l2_slope_vs_dt': r['fit']['l2_slope_vs_dt'],
            'l2_slope_ci': [r['fit']['l2_slope_ci_lo'], r['fit']['l2_slope_ci_hi']],
            'interpretation': r['fit']['interpretation'],
        }

    # FHN
    if run_all or args.method == 'fhn':
        r = run_fhn_n_refinement(
            out, args.fhn_n_seq, args.repeats, args.seed,
            D=0.5, T=args.fhn_T, dt=args.fhn_dt, L=args.fhn_L,
            a_param=args.fhn_a, x_center=15.0,
        )
        results_manifest['fhn_n_refinement'] = {
            'json': os.path.join(out, 'fhn', 'n_refinement.json'),
            'plot': os.path.join(out, 'fhn', 'n_refinement_plot.png'),
            'front_error_slope': r['fit']['front_error_slope'],
            'slope_ci': [r['fit']['slope_ci_lo'], r['fit']['slope_ci_hi']],
        }

    # Method comparison
    if run_all or args.method == 'compare':
        r = run_method_comparison(
            out, args.compare_n_seq,
            nu=args.nu, T=args.T, dt=args.dt, L=args.L,
            amplitude=args.amplitude, repeats=args.repeats,
            base_seed=args.seed, a=args.a,
        )
        results_manifest['cole_hopf_vs_gbmc'] = {
            'json': os.path.join(out, 'comparison', 'cole_hopf_vs_gbmc.json'),
            'plot': os.path.join(out, 'comparison', 'cole_hopf_vs_gbmc_plot.png'),
        }

    elapsed_total = time.perf_counter() - t_total
    results_manifest['_meta'] = {
        'total_runtime_s': round(elapsed_total, 1),
        'base_seed': args.seed,
        'repeats': args.repeats,
    }

    manifest_path = os.path.join(out, 'results_manifest.json')
    _save_json(results_manifest, manifest_path)
    print(f"\n{'='*62}")
    print(f"  All studies complete in {elapsed_total:.1f}s")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*62}")


if __name__ == '__main__':
    main()
