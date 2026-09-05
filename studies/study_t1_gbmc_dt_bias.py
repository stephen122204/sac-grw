"""Task 1: High-N GBMC time-step bias study.

Manuscript map: fixed-particle time-step study (`sec:gbmc-dt-bias`).
Production study; outputs are archived under
output/final_prepublication_tests/gbmc_dt_bias/. Unlike the stationary
sweeps, the field here is reconstructed on N points (equal to the particle
count), which the manuscript discloses when comparing error magnitudes.

Decomposes total error per dt into
  E_bias(dt)   = ||mean_u_dt - u_exact||_2
  E_spread(dt) = [mean_s ||u_s - mean_u_dt||_2^2]^(1/2)
  E_total(dt)  = [mean_s ||u_s - u_exact||_2^2]^(1/2)
with the identity E_total^2 = E_bias^2 + E_spread^2 verified, and fits
nu_fit(dt), x_center_fit(dt) from the ensemble-mean profile (the manuscript's
second fit, with the amplitude fixed at one). Reusing seed identifiers across
step sizes is reproducible but not a pairing: the draw count changes with dt.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SimulationConfig
from relaxation_gbmc import exact_stationary_shock

OUT_BASE = 'output/final_prepublication_tests/gbmc_dt_bias'


def _mk(*parts):
    p = os.path.join(*parts)
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    return p


def _savefig(fig, path_noext):
    for ext in ('png', 'pdf'):
        p = path_noext + '.' + ext
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _run_one(N, nu, T, dt, L, A, xc, a, seed):
    """Single relaxation GBMC run; return u_out on [0,L] grid of size N."""
    from relaxation_gbmc import simulate_burgers_relaxation_gbmc
    from config import generate_burgers_stationary_shock_ic

    ic = generate_burgers_stationary_shock_ic(L, N, nu, x_center=xc, amplitude=A)
    cfg = SimulationConfig(
        equation_type='burgers',
        domain_type='Finite',
        domain_size=L,
        boundary_conditions={'LEFT': {'type': 'Dirichlet', 'value': 0.0},
                             'RIGHT': {'type': 'Dirichlet', 'value': 0.0}},
        diff_constant=nu,
        time_step=dt,
        total_time=T,
        num_points=N,
        initial_conditions=ic,
        reaction_term=False,
        burgers_mode='relaxation_gbmc',
        burgers_ic_type='stationary_shock',
        burgers_ic_amplitude=A,
        relaxation_speed_a=a,
        relaxation_domain_mode='whole_line',
        seed=seed,
        burgers_ic_center=xc,
    )
    globs = [{'position': float(p), 'value': [float(v)]} for p, v in ic]
    t0 = time.perf_counter()
    result = simulate_burgers_relaxation_gbmc(globs, cfg)
    elapsed = time.perf_counter() - t0

    x_out = np.array([g['position'] for g in result])
    u_out = np.array([g['value'][0] for g in result])
    return x_out, u_out, elapsed


def _fit_tanh(x, u, A, xc_init, nu_init):
    """Fit u = -A*tanh(A*(x-xc_fit)/(2*nu_fit)) to profile; return xc_fit, nu_fit."""
    try:
        from scipy.optimize import curve_fit

        def model(x_, xc_, nu_):
            return -A * np.tanh(A * (x_ - xc_) / (2.0 * nu_))
        p0 = [xc_init, nu_init]
        bounds = ([0.0, 1e-4], [float(x[-1]), 10.0])
        popt, _ = curve_fit(model, x, u, p0=p0, bounds=bounds, maxfev=4000)
        return float(popt[0]), float(popt[1])
    except Exception as exc:
        raise RuntimeError(
            f"tanh curve_fit failed in the time-step study: {exc}"
        ) from exc


def _bootstrap_slope_ci(x_arr, y_arr, n_boot=2000, ci=0.95, rng=None):
    if rng is None:
        rng = np.random.default_rng(99)
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


def run_task1(N_vals=None, dt_seq=None, S=40, base_seed=42,
              nu=0.5, T=0.5, L=4.0, A=1.0, a=2.0):
    if N_vals is None:
        N_vals = [6400]
    if dt_seq is None:
        dt_seq = [0.01, 0.005, 0.0025, 0.00125, 0.000625]

    os.makedirs(OUT_BASE, exist_ok=True)
    xc = L / 2.0

    # Validate dt values: T/dt must be integer
    valid_dt = []
    for dt in dt_seq:
        n_steps = T / dt
        if abs(n_steps - round(n_steps)) > 1e-9:
            print(f"  WARNING: T/dt={n_steps:.4f} not integer for dt={dt}; skipping.")
        else:
            valid_dt.append(dt)
    dt_seq = valid_dt

    all_per_run = []
    summaries = []

    for N in N_vals:
        x_grid = np.linspace(0.0, L, N)
        u_exact = exact_stationary_shock(x_grid, nu, amplitude=A, center=xc)
        dx = float(x_grid[1] - x_grid[0])

        print(f"\n{'='*60}")
        print(f"  GBMC dt-bias study  N={N}")
        print(f"{'='*60}")

        dt_results = []
        for dt in dt_seq:
            n_steps = int(round(T / dt))
            print(f"\n  dt={dt}  ({n_steps} steps)")
            seeds = list(range(base_seed, base_seed + S))
            u_runs = []
            runtimes = []

            for s_idx, seed in enumerate(seeds):
                try:
                    x_out, u_run, elapsed = _run_one(N, nu, T, dt, L, A, xc, a, seed)
                    if len(x_out) != N or not np.allclose(x_out, x_grid, atol=1e-10):
                        u_run = np.interp(x_grid, x_out, u_run)
                    u_runs.append(u_run)
                    runtimes.append(elapsed)
                    if (s_idx + 1) % 10 == 0:
                        print(f"    {s_idx+1}/{S} seeds done  t={elapsed:.2f}s each")
                except Exception as e:
                    print(f"    WARNING: seed={seed} failed: {e}")

            if not u_runs:
                print(f"  ERROR: all runs failed for dt={dt}, N={N}")
                continue

            u_arr = np.array(u_runs)   # shape (S_actual, N)
            S_actual = len(u_arr)
            u_mean = u_arr.mean(axis=0)

            # Error decomposition (definitions in module docstring)
            E_bias = float(np.sqrt(np.sum((u_mean - u_exact)**2) * dx))
            E_spread = float(np.sqrt(np.mean(
                np.sum((u_arr - u_mean[None, :])**2 * dx, axis=1)
            )))
            E_total = float(np.sqrt(np.mean(
                np.sum((u_arr - u_exact[None, :])**2 * dx, axis=1)
            )))
            E_total_check = float(np.sqrt(E_bias**2 + E_spread**2))
            identity_err = abs(E_total - E_total_check) / max(E_total, 1e-15)

            l2_runs = [float(np.sqrt(np.sum((u_arr[i] - u_exact)**2) * dx))
                       for i in range(S_actual)]

            # Fit tanh to ensemble mean
            xc_fit, nu_fit = _fit_tanh(x_grid, u_mean, A, xc, nu)
            center_err = abs(xc_fit - xc)
            nu_err = abs(nu_fit - nu)

            mean_rt = float(np.mean(runtimes))

            print(f"    E_bias={E_bias:.5f}  E_spread={E_spread:.5f}  "
                  f"E_total={E_total:.5f}  identity_err={identity_err:.2e}")
            print(f"    xc_fit={xc_fit:.4f}  nu_fit={nu_fit:.4f}  "
                  f"center_err={center_err:.5f}  nu_err={nu_err:.5f}")
            print(f"    S_actual={S_actual}  mean_rt={mean_rt:.3f}s")

            csv_path = _mk(OUT_BASE, f'per_run_N{N}_dt{dt:.8f}.csv')
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['seed', 'l2', 'runtime_s'])
                for i, (seed, l2, rt) in enumerate(zip(seeds[:S_actual], l2_runs, runtimes)):
                    w.writerow([seed, f'{l2:.8f}', f'{rt:.6f}'])

            np.save(_mk(OUT_BASE, f'profile_N{N}_dt{dt:.8f}.npy'),
                    np.stack([x_grid, u_mean, u_exact]))

            for i, (seed, l2, rt) in enumerate(zip(seeds[:S_actual], l2_runs, runtimes)):
                all_per_run.append({
                    'N': N, 'dt': dt, 'seed': seed,
                    'l2': l2, 'runtime_s': rt,
                })

            dt_results.append({
                'N': N, 'dt': dt, 'n_steps': n_steps,
                'S': S_actual,
                'E_bias': E_bias, 'E_spread': E_spread, 'E_total': E_total,
                'E_total_from_identity': E_total_check,
                'identity_residual': identity_err,
                'l2_mean': float(np.mean(l2_runs)),
                'l2_std': float(np.std(l2_runs)),
                'xc_fit': xc_fit, 'nu_fit': nu_fit,
                'center_err': center_err, 'nu_err': nu_err,
                'mean_runtime_s': mean_rt,
            })
            summaries.append(dt_results[-1])

        if not dt_results:
            continue

        # ---- Fit E_bias vs dt ----
        dt_arr   = np.array([r['dt']     for r in dt_results])
        bias_arr = np.array([r['E_bias'] for r in dt_results])
        spr_arr  = np.array([r['E_spread'] for r in dt_results])
        tot_arr  = np.array([r['E_total']  for r in dt_results])
        nu_arr   = np.array([r['nu_fit']   for r in dt_results])
        xce_arr  = np.array([r['center_err'] for r in dt_results])
        rt_arr   = np.array([r['mean_runtime_s'] for r in dt_results])

        valid_b = bias_arr > 1e-8
        if valid_b.sum() >= 2:
            lx = np.log10(dt_arr[valid_b])
            ly = np.log10(bias_arr[valid_b])
            c = np.polyfit(lx, ly, 1)
            bias_slope = float(c[0])
            rng_ci = np.random.default_rng(101)
            b_lo, b_hi = _bootstrap_slope_ci(dt_arr[valid_b], bias_arr[valid_b], rng=rng_ci)
        else:
            bias_slope = float('nan')
            b_lo, b_hi = float('nan'), float('nan')

        print(f"\n  Fitted E_bias slope vs dt: {bias_slope:.4f}  CI=[{b_lo:.4f},{b_hi:.4f}]")

        if np.isnan(bias_slope):
            conclusion = "bias_unresolved: insufficient data points with E_bias > estimation noise"
        elif bias_arr.max() < 2 * spr_arr.min():
            conclusion = ("bias_below_spread: E_bias < 2*E_spread at all dt; "
                          "bias remains below ensemble-estimation noise")
        elif abs(bias_slope) < 0.1 and b_hi < 0.3:
            conclusion = "bias_not_decreasing: slope consistent with zero; requires investigation"
        elif b_lo > 0.5:
            conclusion = f"bias_decreases_rate_{bias_slope:.2f}: slope CI=[{b_lo:.3f},{b_hi:.3f}]"
        elif b_lo > 0.0:
            conclusion = f"bias_decreases_unresolved_rate: slope={bias_slope:.3f} CI=[{b_lo:.3f},{b_hi:.3f}]"
        else:
            conclusion = f"bias_decreases_rate_inconclusive: slope={bias_slope:.3f} CI=[{b_lo:.3f},{b_hi:.3f}]"

        print(f"  Conclusion: {conclusion}")

        fit_result = {
            'N': N, 'S': S,
            'bias_slope': bias_slope,
            'bias_slope_ci_lo': b_lo, 'bias_slope_ci_hi': b_hi,
            'conclusion': conclusion,
            'nu_converges': bool(float(nu_arr[-1]) > 0 and
                                 abs(nu_arr[-1] - nu) < abs(nu_arr[0] - nu)),
            'center_converges': bool(xce_arr[-1] < xce_arr[0]),
        }

        json_path = _mk(OUT_BASE, f'dt_bias_summary_N{N}.json')
        with open(json_path, 'w') as f:
            json.dump({'fit': fit_result, 'per_dt': dt_results,
                       'params': {'N': N, 'nu': nu, 'T': T, 'L': L,
                                  'A': A, 'a': a, 'S': S}}, f, indent=2)

        # ---- Per-dt summary CSV ----
        csv2_path = _mk(OUT_BASE, f'dt_bias_perdt_N{N}.csv')
        with open(csv2_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dt', 'n_steps', 'S', 'E_bias', 'E_spread', 'E_total',
                        'identity_residual', 'l2_mean', 'l2_std',
                        'xc_fit', 'nu_fit', 'center_err', 'nu_err', 'mean_rt_s'])
            for r in dt_results:
                w.writerow([r['dt'], r['n_steps'], r['S'],
                            f"{r['E_bias']:.8f}", f"{r['E_spread']:.8f}",
                            f"{r['E_total']:.8f}", f"{r['identity_residual']:.2e}",
                            f"{r['l2_mean']:.8f}", f"{r['l2_std']:.8f}",
                            f"{r['xc_fit']:.6f}", f"{r['nu_fit']:.6f}",
                            f"{r['center_err']:.6f}", f"{r['nu_err']:.6f}",
                            f"{r['mean_runtime_s']:.4f}"])

        # ---- Figure 1: bias/spread/total vs dt ----
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(dt_arr, bias_arr, 'bs-', lw=1.8, ms=6, label=r'$E_\mathrm{bias}$')
        ax.loglog(dt_arr, spr_arr,  'ro-', lw=1.8, ms=6, label=r'$E_\mathrm{spread}$')
        ax.loglog(dt_arr, tot_arr,  'g^-', lw=1.8, ms=6, label=r'$E_\mathrm{total}$')
        dt_ref = np.array([dt_arr.min(), dt_arr.max()])
        for exp, ls, lbl in [(1.0, '--', r'$O(\Delta t)$'), (0.5, ':', r'$O(\Delta t^{1/2})$')]:
            c0 = max(bias_arr) * (dt_arr.min() ** (-exp))
            ax.loglog(dt_ref, c0 * dt_ref**exp, ls, color='gray', lw=1.0, label=lbl)
        ci_str = f'[{b_lo:.3f},{b_hi:.3f}]' if not np.isnan(b_lo) else 'n/a'
        ax.set_title(f'GBMC dt-bias  N={N}  nu={nu}\n'
                     f'bias slope={bias_slope:.3f}  95%CI={ci_str}', fontsize=10)
        ax.set_xlabel(r'$\Delta t$')
        ax.set_ylabel('Error (L2)')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        _savefig(fig, _mk(OUT_BASE, 'gbmc_bias_vs_dt'))

        # ---- Figure 2: spread and total vs dt ----
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.loglog(dt_arr, spr_arr, 'ro-', lw=1.8, ms=6, label=r'$E_\mathrm{spread}$')
        ax.loglog(dt_arr, tot_arr, 'g^-', lw=1.8, ms=6, label=r'$E_\mathrm{total}$')
        ax.set_title(f'GBMC spread/total vs dt  N={N}')
        ax.set_xlabel(r'$\Delta t$')
        ax.set_ylabel('Error (L2)')
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        _savefig(fig, _mk(OUT_BASE, 'gbmc_spread_total_vs_dt'))

        # ---- Figure 3: fitted viscosity vs dt ----
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].semilogx(dt_arr, nu_arr, 'bs-', lw=1.5, ms=6)
        axes[0].axhline(nu, color='r', ls='--', lw=1.2, label=f'nu_exact={nu}')
        axes[0].set_title(f'Fitted nu_fit vs dt  (N={N})')
        axes[0].set_xlabel(r'$\Delta t$')
        axes[0].set_ylabel(r'$\nu_\mathrm{fit}$')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].semilogx(dt_arr, xce_arr, 'ro-', lw=1.5, ms=6)
        axes[1].set_title(f'Center error |xc_fit - xc| vs dt  (N={N})')
        axes[1].set_xlabel(r'$\Delta t$')
        axes[1].set_ylabel('|center error|')
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, _mk(OUT_BASE, 'gbmc_fitted_viscosity_vs_dt'))

        # ---- Figure 4: center error ----  (already in Fig 3 right panel)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogx(dt_arr, xce_arr, 'ro-', lw=1.8, ms=6)
        ax.set_title(f'GBMC shock-center error vs dt  N={N}')
        ax.set_xlabel(r'$\Delta t$')
        ax.set_ylabel(r'$|x_{c,\mathrm{fit}} - x_c|$')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, _mk(OUT_BASE, 'gbmc_center_error_vs_dt'))

        # ---- Figure 5: ensemble mean profiles for selected dt ----
        selected_dt = [dt_seq[0], dt_seq[len(dt_seq)//2], dt_seq[-1]]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_grid, u_exact, 'k-', lw=2.0, label='Exact', zorder=10)
        colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(selected_dt)))
        for col, dt_sel in zip(colors, selected_dt):
            p = os.path.join(OUT_BASE, f'profile_N{N}_dt{dt_sel:.8f}.npy')
            if os.path.exists(p):
                arr = np.load(p)
                ax.plot(arr[0], arr[1], color=col, lw=1.5, ls='--',
                        label=f'GBMC mean (dt={dt_sel})')
        ax.set_title(f'GBMC ensemble-mean profiles vs exact  N={N}')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, T)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _savefig(fig, _mk(OUT_BASE, 'gbmc_profiles_selected_dt'))

        # ---- Figure 6: runtime vs dt ----
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogx(dt_arr, rt_arr * S, 'bs-', lw=1.5, ms=6)
        ax.set_title(f'GBMC total runtime ({S} seeds) vs dt  N={N}')
        ax.set_xlabel(r'$\Delta t$')
        ax.set_ylabel('Wall time (s)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, _mk(OUT_BASE, 'gbmc_dt_runtime'))

    # Save per-run CSV (all N)
    csv_all = _mk(OUT_BASE, 'per_run_all.csv')
    with open(csv_all, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N', 'dt', 'seed', 'l2', 'runtime_s'])
        for r in all_per_run:
            w.writerow([r['N'], r['dt'], r['seed'], f"{r['l2']:.8f}", f"{r['runtime_s']:.4f}"])

    print(f"\n  [Task 1] Done. Outputs in {OUT_BASE}/")
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', nargs='+', type=int, default=[6400])
    parser.add_argument('--dt', nargs='+', type=float,
                        default=[0.01, 0.005, 0.0025, 0.00125, 0.000625])
    parser.add_argument('--seeds', type=int, default=40)
    parser.add_argument('--base-seed', type=int, default=42)
    args = parser.parse_args()
    run_task1(N_vals=args.N, dt_seq=args.dt, S=args.seeds, base_seed=args.base_seed)
