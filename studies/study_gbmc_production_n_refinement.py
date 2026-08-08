"""Production N-refinement study for the relaxation–Brownian GBMC solver.

Routes exclusively through the production path:
  simulate_burgers() [simulation.py dispatcher]
    -> simulate_burgers_relaxation_gbmc() [relaxation_gbmc.py]

Benchmark: stationary viscous Burgers shock
  u_exact(x) = -A * tanh(A*(x - xc) / (2*nu)),
  A=1, nu=0.5, a=2, T=0.5, dt=0.0025, L=4, xc=2.
N in {100, ..., 6400}, S=50 paired seeds (42 + s), fixed N_out=400 output grid.
"""
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SimulationConfig
from simulation import simulate_burgers

OUT_BASE = 'output/final_prepublication_tests/gbmc_production_n_refinement'


def _savez_deterministic(path, **arrays):
    """Write an uncompressed .npz whose bytes depend only on the array contents.
    numpy's np.savez stamps each zip member with the wall-clock time, so identical
    arrays would otherwise produce differing bytes on each run. Here every member
    uses a fixed timestamp and stored (uncompressed) mode, so a rerun that produces
    identical arrays produces a byte-identical file."""
    import io
    import zipfile
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.save(buf, np.asarray(arr))
            zi = zipfile.ZipInfo(f'{name}.npy', date_time=(1980, 1, 1, 0, 0, 0))
            zi.create_system = 3  # Unix, for platform-stable bytes
            zf.writestr(zi, buf.getvalue())

# Fixed study parameters (paper configuration)
A      = 1.0
NU     = 0.5
A_REL  = 2.0        # relaxation speed  (a > A always)
T      = 0.5
DT     = 0.0025
L      = 4.0
XC     = 2.0
N_SEQ  = [100, 200, 400, 800, 1600, 3200, 6400]
S      = 50
BASE_SEED = 42
N_OUT  = 400        # fixed output grid size for all runs


def _savefig(fig, path_noext):
    os.makedirs(os.path.dirname(os.path.abspath(path_noext + '.png')),
                exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(path_noext + '.' + ext, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _bootstrap_slope_ci(N_arr, E_arr, n_boot=4000, ci=0.95,
                         rng=None, per_run_profiles=None, S_=None):
    """
    Bootstrap CI for log-log slope.

    If per_run_profiles is given (shape S x N_out), resamples within each N
    to recompute E_spread before refitting.  Otherwise resamples (log N, log E)
    pairs directly.
    """
    rng = rng or np.random.default_rng(99)
    valid = np.isfinite(N_arr) & np.isfinite(E_arr) & (N_arr > 0) & (E_arr > 0)
    lx = np.log10(N_arr[valid])
    ly = np.log10(E_arr[valid])
    n  = int(valid.sum())
    if n < 3:
        return float('nan'), float('nan')

    slopes = []
    for _ in range(n_boot):
        if per_run_profiles is not None and S_ is not None:
            # Resample within each N -> recompute E_spread per N
            ly_boot = []
            for profiles in per_run_profiles:  # list of (S x N_out) arrays
                idx = rng.integers(0, S_, size=S_)
                samp = profiles[idx]
                mean_ = samp.mean(axis=0)
                spr = float(np.sqrt(np.mean(np.sum((samp - mean_[None, :])**2
                                                    * (L / N_OUT), axis=1))))
                ly_boot.append(np.log10(max(spr, 1e-15)))
            ly_b = np.array(ly_boot)[valid]
        else:
            ly_b = ly[rng.integers(0, n, size=n)]

        try:
            c = np.polyfit(lx, ly_b, 1)
            slopes.append(float(c[0]))
        except Exception:
            pass

    if not slopes:
        return float('nan'), float('nan')
    lo = float(np.percentile(slopes, 100 * (1 - ci) / 2))
    hi = float(np.percentile(slopes, 100 * (1 + ci) / 2))
    return lo, hi


def _fit_tanh(x, u, A_hint, xc_hint, nu_hint):
    """
    Fit u = -A_f * tanh(A_f*(x-xc_f)/(2*nu_f)) using zero-crossing and
    width estimate.  Falls back gracefully.  Returns (xc_fit, nu_fit, A_fit).
    """
    try:
        from scipy.optimize import curve_fit
        def model(x_, A_, xc_, nu_):
            return -A_ * np.tanh(A_ * (x_ - xc_) / (2.0 * nu_))
        p0 = [A_hint, xc_hint, nu_hint]
        bounds = ([0.5, 0.0, 0.05], [2.0, float(L), 2.0])
        popt, _ = curve_fit(model, x, u, p0=p0, bounds=bounds, maxfev=5000)
        return float(popt[1]), float(popt[2]), float(popt[0])
    except Exception:
        pass
    # Fallback: zero-crossing for xc
    idx_cross = int(np.argmin(np.abs(u)))
    xc_f = float(x[idx_cross])
    # Width: use slope at zero-crossing: du/dx|_{xc} = -A^2/(2*nu) => nu_f = -A^2/(2*slope)
    if 1 <= idx_cross < len(x) - 1:
        slope = float((u[idx_cross + 1] - u[idx_cross - 1]) /
                      (x[idx_cross + 1] - x[idx_cross - 1]))
        nu_f = float(-A_hint**2 / (2.0 * slope)) if slope < 0 else nu_hint
    else:
        nu_f = nu_hint
    return xc_f, max(nu_f, 0.01), A_hint


def u_exact_fn(x):
    return -A * np.tanh(A * (x - XC) / (2.0 * NU))


def _run_one(N, seed):
    """Run one GBMC production simulation through the simulate_burgers()
    dispatcher (burgers_mode='relaxation_gbmc').

    Returns dict of per-run quantities, or None on unrecoverable failure.
    """
    cfg = SimulationConfig(
        equation_type='burgers',
        domain_type='Finite',
        domain_size=L,
        boundary_conditions={
            'LEFT':  {'type': 'Dirichlet', 'value': 0.0},
            'RIGHT': {'type': 'Dirichlet', 'value': 0.0},
        },
        diff_constant=NU,
        time_step=DT,
        total_time=T,
        num_points=N,
        initial_conditions=[],
        reaction_term=False,
        burgers_mode='relaxation_gbmc',
        burgers_ic_type='stationary_shock',
        burgers_ic_amplitude=A,
        relaxation_speed_a=A_REL,
        relaxation_domain_mode='whole_line',
        burgers_ic_center=XC,
    )
    cfg.seed = seed   # private RNG inside relaxation_gbmc

    # N_OUT dummy globs: the solver overwrites their positions/values entirely
    globs = [{'position': float(i) * L / (N_OUT - 1), 'value': [0.0]}
             for i in range(N_OUT)]

    t0 = time.perf_counter()
    try:
        result = simulate_burgers(globs, cfg)
    except Exception as e:
        return {'failed': True, 'reason': str(e), 'N': N, 'seed': seed}
    elapsed = time.perf_counter() - t0

    x_out = np.array([g['position'] for g in result])
    u_out = np.array([g['value'][0] if isinstance(g['value'], list)
                      else float(g['value']) for g in result])

    if not np.all(np.isfinite(u_out)):
        return {'failed': True, 'reason': 'NaN/inf in u_out', 'N': N, 'seed': seed}

    dx = float(x_out[1] - x_out[0])
    u_ref = u_exact_fn(x_out)

    err = u_out - u_ref
    l1   = float(np.sum(np.abs(err)) * dx)
    l2   = float(np.sqrt(np.sum(err**2) * dx))
    linf = float(np.max(np.abs(err)))
    u_ref_l2 = float(np.sqrt(np.sum(u_ref**2) * dx))
    rel_l2 = l2 / max(u_ref_l2, 1e-15)

    # Secondary: fit tanh for center and width
    xc_fit, nu_fit, A_fit = _fit_tanh(x_out, u_out, A, XC, NU)

    # Mass diagnostics (from config: expected total mass = -2A)
    # These are read from relaxation_gbmc stdout diagnostics — here we
    # proxy them from the expected exact values + output.
    total_mass_expected = -2.0 * A
    max_abs_u = float(np.max(np.abs(u_out)))

    return {
        'failed':     False,
        'N':          N,
        'seed':       seed,
        'l1':         l1,
        'l2':         l2,
        'linf':       linf,
        'rel_l2':     rel_l2,
        'xc_fit':     xc_fit,
        'nu_fit':     nu_fit,
        'A_fit':      A_fit,
        'max_abs_u':  max_abs_u,
        'total_mass_expected': total_mass_expected,
        'runtime_s':  elapsed,
        'u_out':      u_out,   # kept in memory for decomposition
        'x_out':      x_out,
    }


def run_study():
    os.makedirs(OUT_BASE, exist_ok=True)

    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    seeds = [BASE_SEED + s for s in range(S)]
    x_ref = np.linspace(0.0, L, N_OUT)
    u_ref = u_exact_fn(x_ref)
    dx    = float(x_ref[1] - x_ref[0])

    print(f"\n{'#'*70}")
    print(f"#  GBMC Production N-Refinement Study")
    print(f"#  N={N_SEQ}  S={S}  dt={DT}  T={T}  A={A}  nu={NU}  a={A_REL}")
    print(f"#  dispatcher: simulate_burgers() -> simulate_burgers_relaxation_gbmc()")
    print(f"#  git: {git_hash}")
    print(f"{'#'*70}")

    all_per_run = []   # flat list of dicts (no u_out stored)
    N_results   = []   # one dict per N
    all_profiles_by_N = {}   # N -> (S x N_OUT) array

    for N in N_SEQ:
        print(f"\n{'='*60}")
        print(f"  N={N}  (S={S} seeds)")
        print(f"{'='*60}")

        u_runs   = []
        rec_rows = []
        n_fail   = 0

        for s_idx, seed in enumerate(seeds):
            r = _run_one(N, seed)
            if r is None or r.get('failed'):
                reason = (r or {}).get('reason', 'unknown')
                print(f"    FAIL seed={seed}: {reason}")
                n_fail += 1
                rec_rows.append({
                    'N': N, 'seed': seed, 'failed': True, 'reason': reason,
                    'l1': '', 'l2': '', 'linf': '', 'rel_l2': '',
                    'xc_fit': '', 'nu_fit': '', 'A_fit': '',
                    'max_abs_u': '', 'total_mass_expected': '',
                    'outside_N': '', 'outside_signed': '', 'outside_abs': '',
                    'runtime_s': '',
                })
                continue

            u_runs.append(r['u_out'])
            rec_rows.append({
                'N':          r['N'],
                'seed':       r['seed'],
                'failed':     False,
                'reason':     '',
                'l1':         r['l1'],
                'l2':         r['l2'],
                'linf':       r['linf'],
                'rel_l2':     r['rel_l2'],
                'xc_fit':     r['xc_fit'],
                'nu_fit':     r['nu_fit'],
                'A_fit':      r['A_fit'],
                'max_abs_u':  r['max_abs_u'],
                'total_mass_expected': r['total_mass_expected'],
                'outside_N':       '',  # extracted from RBMC stdout; logged above
                'outside_signed':  '',
                'outside_abs':     '',
                'runtime_s':  r['runtime_s'],
            })

            if (s_idx + 1) % 10 == 0:
                print(f"    {s_idx+1}/{S} seeds done")

        all_per_run.extend(rec_rows)
        S_actual = len(u_runs)
        print(f"    {S_actual}/{S} runs succeeded, {n_fail} failed")

        if S_actual < 3:
            print(f"  ERROR: too few runs for N={N}, skipping decomposition")
            continue

        u_arr  = np.array(u_runs)        # (S_actual, N_OUT)
        u_mean = u_arr.mean(axis=0)

        E_bias   = float(np.sqrt(np.sum((u_mean - u_ref)**2 * dx)))
        E_spread = float(np.sqrt(np.mean(
            np.sum((u_arr - u_mean[None, :])**2 * dx, axis=1)
        )))
        E_total  = float(np.sqrt(np.mean(
            np.sum((u_arr - u_ref[None, :])**2 * dx, axis=1)
        )))
        E_total_check = float(np.sqrt(E_bias**2 + E_spread**2))
        identity_err  = abs(E_total - E_total_check) / max(E_total, 1e-15)

        # Secondary center/width stats
        xc_fits  = [rec_rows[i]['xc_fit']  for i in range(len(rec_rows))
                    if not rec_rows[i]['failed'] and rec_rows[i]['xc_fit'] != '']
        nu_fits  = [rec_rows[i]['nu_fit']  for i in range(len(rec_rows))
                    if not rec_rows[i]['failed'] and rec_rows[i]['nu_fit'] != '']
        xc_arr   = np.array(xc_fits, dtype=float)
        nu_arr   = np.array(nu_fits, dtype=float)
        xc_std   = float(np.std(xc_arr)) if len(xc_arr) > 1 else float('nan')
        nu_std   = float(np.std(nu_arr)) if len(nu_arr) > 1 else float('nan')
        xc_mean  = float(np.mean(xc_arr)) if len(xc_arr) > 0 else float('nan')
        nu_mean  = float(np.mean(nu_arr)) if len(nu_arr) > 0 else float('nan')

        mean_rt = float(np.mean([r['runtime_s'] for r in rec_rows
                                  if not r['failed'] and r['runtime_s'] != '']))

        print(f"    E_bias={E_bias:.5f}  E_spread={E_spread:.5f}  "
              f"E_total={E_total:.5f}  identity_err={identity_err:.2e}")
        print(f"    xc_mean={xc_mean:.4f}±{xc_std:.4f}  "
              f"nu_mean={nu_mean:.4f}±{nu_std:.4f}  mean_rt={mean_rt:.3f}s")

        all_profiles_by_N[N] = u_arr
        N_results.append({
            'N':            N,
            'S_actual':     S_actual,
            'n_failed':     n_fail,
            'E_bias':       E_bias,
            'E_spread':     E_spread,
            'E_total':      E_total,
            'identity_err': identity_err,
            'xc_mean':      xc_mean,
            'xc_std':       xc_std,
            'nu_mean':      nu_mean,
            'nu_std':       nu_std,
            'mean_rt':      mean_rt,
        })

    if not N_results:
        print("ERROR: no results collected.")
        return

    N_arr      = np.array([r['N']        for r in N_results], dtype=float)
    bias_arr   = np.array([r['E_bias']   for r in N_results])
    spr_arr    = np.array([r['E_spread'] for r in N_results])
    tot_arr    = np.array([r['E_total']  for r in N_results])
    xc_std_arr = np.array([r['xc_std']  for r in N_results])
    nu_std_arr = np.array([r['nu_std']  for r in N_results])

    rng_boot = np.random.default_rng(123)

    # Spread slope (primary)
    valid = (N_arr > 0) & (spr_arr > 0)
    lx = np.log10(N_arr[valid])
    ly = np.log10(spr_arr[valid])
    coeffs_spr = np.polyfit(lx, ly, 1)
    spr_slope  = float(coeffs_spr[0])

    ly_pred = np.polyval(coeffs_spr, lx)
    ss_res  = float(np.sum((ly - ly_pred)**2))
    ss_tot  = float(np.sum((ly - ly.mean())**2))
    spr_r2  = float(1.0 - ss_res / max(ss_tot, 1e-30))

    # Adjacent empirical rates
    adj_rates = []
    Nv = N_arr[valid]; Sv = spr_arr[valid]
    for i in range(len(Nv) - 1):
        r_adj = float((np.log10(Sv[i+1]) - np.log10(Sv[i])) /
                      (np.log10(Nv[i+1]) - np.log10(Nv[i])))
        adj_rates.append({'N_lo': int(Nv[i]), 'N_hi': int(Nv[i+1]), 'rate': r_adj})

    # Bootstrap CI (resample profiles within each N)
    profiles_list = [all_profiles_by_N[int(n)] for n in Nv]
    ci_lo, ci_hi  = _bootstrap_slope_ci(
        N_arr[valid], spr_arr[valid],
        rng=rng_boot,
        per_run_profiles=profiles_list,
        S_=S,
    )

    # Bias and total slopes
    def _slope_r2(N_a, E_a):
        v = (N_a > 0) & np.isfinite(E_a) & (E_a > 0)
        if v.sum() < 3:
            return float('nan'), float('nan')
        lx_ = np.log10(N_a[v]); ly_ = np.log10(E_a[v])
        c_ = np.polyfit(lx_, ly_, 1)
        lp = np.polyval(c_, lx_)
        r2_ = 1.0 - float(np.sum((ly_-lp)**2)) / max(float(np.sum((ly_-ly_.mean())**2)), 1e-30)
        return float(c_[0]), float(r2_)

    bias_slope, bias_r2   = _slope_r2(N_arr, bias_arr)
    tot_slope,  tot_r2    = _slope_r2(N_arr, tot_arr)
    xcs_slope,  _         = _slope_r2(N_arr, xc_std_arr)
    nus_slope,  _         = _slope_r2(N_arr, nu_std_arr)

    print(f"\n{'='*60}")
    print(f"  Convergence results (S={S} seeds)")
    print(f"{'='*60}")
    print(f"  Spread slope: {spr_slope:.4f}  R²={spr_r2:.3f}  "
          f"CI=[{ci_lo:.4f},{ci_hi:.4f}]")
    print(f"  Bias   slope: {bias_slope:.4f}  R²={bias_r2:.3f}")
    print(f"  Total  slope: {tot_slope:.4f}   R²={tot_r2:.3f}")
    print(f"  Center std slope: {xcs_slope:.4f}")
    print(f"  Width  std slope: {nus_slope:.4f}")
    print(f"  Adjacent rates:")
    for ar in adj_rates:
        print(f"    N={ar['N_lo']}->{ar['N_hi']}: rate={ar['rate']:.4f}")

    # Compatibility with -0.5
    expected = -0.5
    compat   = (ci_lo <= expected <= ci_hi)
    earlier  = -0.507
    ci_earlier = (-0.534, -0.479)
    overlap  = max(ci_lo, ci_earlier[0]) <= min(ci_hi, ci_earlier[1])
    print(f"\n  Compatible with -1/2 (within 95% CI): {compat}")
    print(f"  Overlap with earlier CI [-0.534,-0.479]: {overlap}")

    per_run_csv = os.path.join(OUT_BASE, 'per_run.csv')
    fieldnames  = ['N', 'seed', 'failed', 'reason', 'l1', 'l2', 'linf',
                   'rel_l2', 'xc_fit', 'nu_fit', 'A_fit', 'max_abs_u',
                   'total_mass_expected', 'outside_N', 'outside_signed',
                   'outside_abs', 'runtime_s']
    with open(per_run_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader(); w.writerows(all_per_run)

    per_N_csv = os.path.join(OUT_BASE, 'per_N_summary.csv')
    with open(per_N_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(N_results[0].keys()))
        w.writeheader(); w.writerows(N_results)

    rates = {
        'spread_slope':   spr_slope,
        'spread_r2':      spr_r2,
        'spread_ci_lo':   ci_lo,
        'spread_ci_hi':   ci_hi,
        'bias_slope':     bias_slope,
        'bias_r2':        bias_r2,
        'total_slope':    tot_slope,
        'total_r2':       tot_r2,
        'xc_std_slope':   xcs_slope,
        'nu_std_slope':   nus_slope,
        'adjacent_rates': adj_rates,
        'compatible_with_minus_half': compat,
        'overlap_with_earlier_ci':    overlap,
        'earlier_slope': earlier,
        'earlier_ci':    list(ci_earlier),
        'S': S, 'N_seq': N_SEQ, 'N_out': N_OUT,
        'A': A, 'nu': NU, 'a_rel': A_REL, 'T': T, 'dt': DT,
        'L': L, 'xc': XC, 'base_seed': BASE_SEED,
    }
    with open(os.path.join(OUT_BASE, 'rates.json'), 'w') as f:
        json.dump(rates, f, indent=2)

    meta = {
        'git_commit': git_hash,
        'dispatcher': 'simulate_burgers() in simulation.py',
        'routes_to':  'simulate_burgers_relaxation_gbmc() in relaxation_gbmc.py',
        'burgers_mode': 'relaxation_gbmc',
        'seeds': seeds,
        'N_seq': N_SEQ,
        'S': S,
        'N_out_fixed': N_OUT,
        'params': {'A': A, 'nu': NU, 'a': A_REL, 'T': T, 'dt': DT,
                   'L': L, 'xc': XC},
        'reconstruction': 'raw cumsum: u_inf + cumsum(m_sorted)[searchsorted(x_sorted, x_out, right)]',
        'no_smoothing': True,
        'command': f'python study_gbmc_production_n_refinement.py',
    }
    with open(os.path.join(OUT_BASE, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Figures
    # 1. E_spread vs N (primary)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N_arr, spr_arr, 'o-', color='steelblue', lw=2,
              ms=7, label=f'$E_{{spread}}$  slope={spr_slope:.3f}')
    xfit = np.array([N_arr.min(), N_arr.max()])
    yfit = 10**np.polyval(coeffs_spr, np.log10(xfit))
    ax.loglog(xfit, yfit, '--', color='steelblue', alpha=0.6,
              label=f'fit  CI=[{ci_lo:.3f},{ci_hi:.3f}]')
    # reference N^{-1/2} line through midpoint
    i_mid = len(N_arr) // 2
    ref_y = spr_arr[i_mid] * (xfit / N_arr[i_mid])**(-0.5)
    ax.loglog(xfit, ref_y, 'k:', lw=1.2, label=r'$N^{-1/2}$ reference')
    ax.set_xlabel('N (particles)')
    ax.set_ylabel(r'$E_{\mathrm{spread}}$')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    _savefig(fig, os.path.join(OUT_BASE, 'production_gbmc_spread_vs_N'))

    # 2. Bias / spread / total vs N
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N_arr, tot_arr,  's-', color='black',     lw=2, ms=7,
              label=f'$E_{{total}}$ slope={tot_slope:.3f}')
    ax.loglog(N_arr, spr_arr,  'o-', color='steelblue', lw=2, ms=7,
              label=f'$E_{{spread}}$ slope={spr_slope:.3f}')
    ax.loglog(N_arr, bias_arr, '^-', color='firebrick',  lw=2, ms=7,
              label=f'$E_{{bias}}$ slope={bias_slope:.3f}')
    ax.loglog(xfit, ref_y, 'k:', lw=1.2, label=r'$N^{-1/2}$ reference')
    ax.set_xlabel('N'); ax.set_ylabel('Error')
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
    _savefig(fig, os.path.join(OUT_BASE, 'production_gbmc_bias_spread_total_vs_N'))

    # 3. Center std and width std vs N
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog(N_arr, xc_std_arr, 'o-', color='darkorange', lw=2, ms=7)
    axes[0].loglog(xfit, xc_std_arr[i_mid] * (xfit / N_arr[i_mid])**(-0.5),
                   'k:', lw=1.2)
    axes[0].set_xlabel('N'); axes[0].set_ylabel(r'std($x_c$)')
    axes[0].set_title(f'Center variability  slope={xcs_slope:.3f}')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[1].loglog(N_arr, nu_std_arr, 'o-', color='purple', lw=2, ms=7)
    axes[1].loglog(xfit, nu_std_arr[i_mid] * (xfit / N_arr[i_mid])**(-0.5),
                   'k:', lw=1.2)
    axes[1].set_xlabel('N'); axes[1].set_ylabel(r'std($\nu_{fit}$)')
    axes[1].set_title(f'Width variability  slope={nus_slope:.3f}')
    axes[1].grid(True, which='both', alpha=0.3)
    fig.suptitle('GBMC: secondary center/width variability', fontsize=11)
    fig.tight_layout()
    _savefig(fig, os.path.join(OUT_BASE, 'production_gbmc_center_width_std_vs_N'))

    # 4. Profiles at selected N
    N_plot = [N for N in [100, 400, 1600, 6400] if N in all_profiles_by_N]
    # Save the mean +/- std profiles the paper figure uses, so the figure is
    # reproducible from a checked-in file. Diagnostic data only: this does not
    # affect any solver output, statistic, or reported value.
    _savez_deterministic(os.path.join(OUT_BASE, 'production_profiles.npz'),
                         x=x_ref, u_exact=u_ref,
                         **{f'N{N}_mean': all_profiles_by_N[N].mean(axis=0) for N in N_plot},
                         **{f'N{N}_std':  all_profiles_by_N[N].std(axis=0)  for N in N_plot})
    colors_ = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_plot)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_ref, u_ref, 'k-', lw=2, label='Exact', zorder=10)
    for col, N in zip(colors_, N_plot):
        u_arr_N = all_profiles_by_N[N]
        um = u_arr_N.mean(axis=0)
        us = u_arr_N.std(axis=0)
        ax.plot(x_ref, um, '-', color=col, lw=1.5, label=f'N={N}', zorder=5)
        ax.fill_between(x_ref, um - us, um + us, color=col, alpha=0.15)
    ax.set_xlabel('x'); ax.set_ylabel('u')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(OUT_BASE, 'production_gbmc_profiles_selected_N'))

    # 5. Fitted viscosity vs N
    nu_means = np.array([r['nu_mean'] for r in N_results])
    nu_stds  = np.array([r['nu_std']  for r in N_results])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(N_arr, nu_means, yerr=nu_stds, fmt='o-', color='purple',
                lw=2, ms=7, capsize=4, label=r'$\nu_{\mathrm{fit}}$ (mean ± std)')
    ax.axhline(NU, color='k', ls='--', lw=1.5, label=f'Exact ν={NU}')
    ax.set_xscale('log')
    ax.set_xlabel('N'); ax.set_ylabel(r'$\nu_{\mathrm{fit}}$')
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
    _savefig(fig, os.path.join(OUT_BASE, 'production_gbmc_fitted_viscosity_vs_N'))

    # 6. Runtime vs N
    rt_means = np.array([r['mean_rt'] for r in N_results])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(N_arr, rt_means, 'o-', color='teal', lw=2, ms=7)
    ax.set_xlabel('N'); ax.set_ylabel('Mean runtime (s)')
    ax.set_title('GBMC production: runtime vs N')
    ax.grid(True, which='both', alpha=0.3)
    _savefig(fig, os.path.join(OUT_BASE, 'production_gbmc_runtime_vs_N'))

    print(f"\n  Outputs in {OUT_BASE}/")
    return rates


if __name__ == '__main__':
    t0_wall = time.perf_counter()
    rates = run_study()
    elapsed_total = time.perf_counter() - t0_wall
    print(f"\n  Total wall time: {elapsed_total:.1f}s")
    if rates:
        compat = rates.get('compatible_with_minus_half', False)
        slope  = rates.get('spread_slope', float('nan'))
        ci_lo  = rates.get('spread_ci_lo', float('nan'))
        ci_hi  = rates.get('spread_ci_hi', float('nan'))
        print(f"\n  ANSWER: spread slope={slope:.4f}  "
              f"CI=[{ci_lo:.4f},{ci_hi:.4f}]")
        print(f"  Compatible with N^(-1/2): {compat}")
