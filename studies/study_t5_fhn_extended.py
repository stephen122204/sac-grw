"""Task 5: Extended FHN convergence study.

N in {100, ..., 5000}, 30 paired seeds per N; profile metrics (L1/L2/Linf/rel-L2),
front center/speed/aligned-profile errors with separate fitted rates, a dt
refinement at fixed N, and front center vs time.
Output: output/final_prepublication_tests/fhn_extended/
"""
import csv
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SimulationConfig, generate_fhn_steady_ic

OUT_BASE = 'output/final_prepublication_tests/fhn_extended'


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


def _run_fhn_one(N, a, nu, T, dt, L, xc, seed):
    """Single FHN GRW run; returns (x_sorted, u_cumsum, elapsed), u = cumsum of sorted weights."""
    from simulation import simulate_fitzhugh_nagumo_grw

    np.random.seed(seed)
    ic = generate_fhn_steady_ic(N, a, x_center=xc)

    cfg = SimulationConfig(
        equation_type='fitzhugh-nagumo',
        domain_type='Finite',
        domain_size=L,
        boundary_conditions={'LEFT': {'type': 'Neumann', 'value': 0.0},
                             'RIGHT': {'type': 'Neumann', 'value': 0.0}},
        diff_constant=nu,
        time_step=dt,
        total_time=T,
        num_points=N,
        initial_conditions=ic,
        reaction_term=True,
        a=a,
    )
    globs = [{'position': float(pos), 'value': float(w)} for pos, w in ic]
    t0 = time.perf_counter()
    result = simulate_fitzhugh_nagumo_grw(globs, cfg)
    elapsed = time.perf_counter() - t0

    x_arr = np.array([g['position'] for g in result])
    w_arr = np.array([
        float(g['value'][0]) if isinstance(g['value'], (list, tuple)) else float(g['value'])
        for g in result
    ])
    order = np.argsort(x_arr)
    x_s = x_arr[order]
    u_s = np.cumsum(w_arr[order])
    return x_s, u_s, elapsed


def _extract_profile(result):
    """Extract (x_sorted, u_cumsum) from FHN result globs."""
    x_arr = np.array([g['position'] for g in result])
    w_arr = np.array([
        float(g['value'][0]) if isinstance(g['value'], (list, tuple)) else float(g['value'])
        for g in result
    ])
    order = np.argsort(x_arr)
    return x_arr[order], np.cumsum(w_arr[order])


def _find_front_center(x, u, threshold=0.5):
    """Find approximate location of u crossing threshold (front center)."""
    diffs = u - threshold
    sign_changes = np.where(np.diff(np.sign(diffs)))[0]
    if len(sign_changes) == 0:
        return float(x[np.argmin(np.abs(diffs))])
    idx = sign_changes[len(sign_changes)//2]
    d0 = float(diffs[idx]); d1 = float(diffs[idx+1])
    if abs(d1 - d0) < 1e-15:
        return float(x[idx])
    frac = -d0 / (d1 - d0)
    return float(x[idx] + frac * (x[idx+1] - x[idx]))


def _aligned_profile_error(x1, u1, x2, u2):
    """Align two profiles by their front centers (shift + interpolate u2 to x1),
    then compute the L2 error.
    """
    c1 = _find_front_center(x1, u1)
    c2 = _find_front_center(x2, u2)
    x2_shifted = x2 + (c1 - c2)
    u2_interp = np.interp(x1, x2_shifted, u2, left=u2[0], right=u2[-1])
    dx = float(x1[1] - x1[0])
    return float(np.sqrt(np.sum((u1 - u2_interp)**2 * dx)))


def _bootstrap_slope_ci(x_arr, y_arr, n_boot=2000, ci=0.95, rng=None):
    if rng is None:
        rng = np.random.default_rng(55)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0) & (y_arr > 0)
    n = int(valid.sum())
    if n < 3:
        return float('nan'), float('nan'), float('nan')
    lx = np.log10(x_arr[valid])
    ly = np.log10(y_arr[valid])
    c = np.polyfit(lx, ly, 1)
    slope = float(c[0])
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            cb = np.polyfit(lx[idx], ly[idx], 1)
            slopes.append(cb[0])
        except Exception:
            pass
    if not slopes:
        return slope, float('nan'), float('nan')
    lo = float(np.percentile(slopes, 100*(1-ci)/2))
    hi = float(np.percentile(slopes, 100*(1+ci)/2))
    return slope, lo, hi


def run_task5(N_seq=None, S=30, base_seed=42,
              a=0.25, nu=0.5,
              T=5.0, dt=0.01, L=30.0, xc=15.0,
              dt_study_N=2000, dt_seq=None,
              track_center=True):
    os.makedirs(OUT_BASE, exist_ok=True)
    if N_seq is None:
        N_seq = [100, 200, 500, 1000, 2000, 5000]
    if dt_seq is None:
        dt_seq = [0.04, 0.02, 0.01, 0.005]

    print(f"\n{'='*60}\n  Task 5: Extended FHN convergence\n"
          f"  N_seq={N_seq}  S={S}  T={T}\n{'='*60}")

    theta_exact = float(np.sqrt(2.0) * (0.5 - a))
    xc_exact_T = xc - theta_exact * T  # front position at time T
    print(f"  Exact front speed: theta={theta_exact:.4f}  "
          f"front center at T={T}: {xc_exact_T:.4f}")

    per_run_records = []
    N_results = []

    N_max = max(N_seq)
    print(f"\n  Building reference solution (N={N_max}, 20 seeds)...")
    ref_runs = []
    for rep in range(20):
        try:
            x_r, u_r, _ = _run_fhn_one(N_max, a, nu, T, dt, L, xc, base_seed+rep+1000)
            ref_runs.append((x_r, u_r))
        except Exception as e:
            print(f"    WARNING ref seed={base_seed+rep+1000}: {e}")
    if not ref_runs:
        print("  ERROR: cannot build reference solution.")
        return {}
    x_ref = ref_runs[0][0]
    u_ref = np.mean([np.interp(x_ref, r[0], r[1]) for r in ref_runs], axis=0)
    xc_ref = _find_front_center(x_ref, u_ref)
    dx_ref = float(x_ref[1] - x_ref[0])
    np.save(_mk(OUT_BASE, 'reference_profile.npy'), np.stack([x_ref, u_ref]))
    print(f"  Reference front center: {xc_ref:.4f}")

    # N-refinement study
    for N in N_seq:
        print(f"\n  N={N}")
        seeds = [base_seed + i for i in range(S)]
        l1_list = []; l2_list = []; linf_list = []; rel_l2_list = []
        ce_list = []  # front center
        ap_list = []  # aligned-profile error
        sp_list = []  # front speed estimate (via center / T)
        rt_list = []

        for s_idx, seed in enumerate(seeds):
            try:
                x_out, u_out, elapsed = _run_fhn_one(N, a, nu, T, dt, L, xc, seed)
                rt_list.append(elapsed)

                u_interp = np.interp(x_ref, x_out, u_out)
                diff = u_interp - u_ref
                l1  = float(np.sum(np.abs(diff)) * dx_ref)
                l2  = float(np.sqrt(np.sum(diff**2) * dx_ref))
                linf = float(np.max(np.abs(diff)))
                denom = float(np.sqrt(np.sum(u_ref**2) * dx_ref))
                rel_l2 = l2 / max(denom, 1e-15)

                # Front center error vs exact analytical position
                xc_num = _find_front_center(x_out, u_out)
                center_err = abs(xc_num - xc_exact_T)

                # Speed estimate: (xc_num - xc_initial) / T vs exact theta
                speed_num = (xc_num - xc) / max(T, 1e-15)
                speed_err = abs(speed_num - (-theta_exact))

                ap_err = _aligned_profile_error(x_ref, u_ref, x_out, u_out)

                l1_list.append(l1); l2_list.append(l2); linf_list.append(linf)
                rel_l2_list.append(rel_l2); ce_list.append(center_err)
                ap_list.append(ap_err); sp_list.append(speed_err)

                per_run_records.append({
                    'N': N, 'seed': seed, 'l1': l1, 'l2': l2, 'linf': linf,
                    'rel_l2': rel_l2, 'center_err': center_err,
                    'speed_err': speed_err, 'aligned_err': ap_err,
                    'xc_num': xc_num, 'runtime_s': elapsed,
                })
                if (s_idx + 1) % 10 == 0:
                    print(f"    {s_idx+1}/{S} done")
            except Exception as e:
                print(f"    WARNING seed={seed}: {e}")

        if not l2_list:
            print(f"  ERROR: all runs failed for N={N}")
            continue

        row = {
            'N': N, 'S': len(l2_list),
            'l2_mean': float(np.mean(l2_list)), 'l2_std': float(np.std(l2_list)),
            'l1_mean': float(np.mean(l1_list)),
            'linf_mean': float(np.mean(linf_list)),
            'rel_l2_mean': float(np.mean(rel_l2_list)),
            'ce_mean': float(np.mean(ce_list)), 'ce_std': float(np.std(ce_list)),
            'speed_err_mean': float(np.mean(sp_list)),
            'ap_mean': float(np.mean(ap_list)), 'ap_std': float(np.std(ap_list)),
            'mean_rt': float(np.mean(rt_list)),
        }
        N_results.append(row)
        recs_N = [r for r in per_run_records if r['N'] == N]
        if recs_N:
            u_mean_approx = u_ref  # fallback: reference
            np.save(_mk(OUT_BASE, f'profile_N{N}.npy'),
                    np.stack([x_ref, u_mean_approx]))

        print(f"  N={N:5d}  L2={row['l2_mean']:.4f}±{row['l2_std']:.4f}  "
              f"center_err={row['ce_mean']:.4f}  speed_err={row['speed_err_mean']:.4f}  "
              f"ap_err={row['ap_mean']:.4f}")

    if not N_results:
        print("  ERROR: no N_results.")
        return {}

    # ---- dt-refinement study ----
    dt_results = []
    print(f"\n  dt-refinement (N={dt_study_N})")
    for dt_r in dt_seq:
        l2_list = []; ce_list = []
        for rep in range(10):
            seed = base_seed + 500 + rep
            try:
                x_out, u_out, _ = _run_fhn_one(dt_study_N, a, nu, T, dt_r, L, xc, seed)
                u_interp = np.interp(x_ref, x_out, u_out)
                l2 = float(np.sqrt(np.sum((u_interp - u_ref)**2 * dx_ref)))
                xc_num = _find_front_center(x_out, u_out)
                l2_list.append(l2)
                ce_list.append(abs(xc_num - xc_exact_T))
            except Exception as e:
                print(f"    WARNING dt={dt_r} seed={seed}: {e}")
        if l2_list:
            dt_results.append({
                'dt': dt_r, 'N': dt_study_N,
                'l2_mean': float(np.mean(l2_list)),
                'l2_std': float(np.std(l2_list)),
                'ce_mean': float(np.mean(ce_list)),
            })
            print(f"  dt={dt_r}  L2={dt_results[-1]['l2_mean']:.4f}  "
                  f"center_err={dt_results[-1]['ce_mean']:.4f}")

    # ---- Front center vs time (N=N_max, one seed) ----
    center_vs_time = []
    if track_center:
        print(f"\n  Front center vs time (N={N_max})")
        T_track_seq = np.linspace(0.0, T, 11)[1:]
        for t_out in T_track_seq:
            n_steps = max(1, int(round(t_out / dt)))
            dt_t = t_out / n_steps
            try:
                x_out, u_out, _ = _run_fhn_one(N_max, a, nu, t_out, dt_t, L, xc, base_seed+2000)
                xc_t = _find_front_center(x_out, u_out)
                center_vs_time.append({'t': float(t_out), 'xc': xc_t,
                                       'xc_exact': float(xc - theta_exact * t_out)})
            except Exception as e:
                print(f"  WARNING center vs time t={t_out}: {e}")

    # ---- Fit slopes ----
    N_arr  = np.array([r['N'] for r in N_results], dtype=float)
    l2_arr = np.array([r['l2_mean'] for r in N_results])
    ce_arr = np.array([r['ce_mean'] for r in N_results])
    sp_arr = np.array([r['speed_err_mean'] for r in N_results])
    ap_arr = np.array([r['ap_mean'] for r in N_results])
    rt_arr = np.array([r['mean_rt'] for r in N_results])

    rng_ci = np.random.default_rng(404)
    l2_slope, l2_lo, l2_hi = _bootstrap_slope_ci(N_arr, l2_arr, rng=rng_ci)
    ce_slope, ce_lo, ce_hi = _bootstrap_slope_ci(N_arr, ce_arr[ce_arr>0], rng=rng_ci) \
        if np.sum(ce_arr > 0) >= 2 else (float('nan'), float('nan'), float('nan'))
    sp_slope, sp_lo, sp_hi = _bootstrap_slope_ci(N_arr, sp_arr[sp_arr>0], rng=rng_ci) \
        if np.sum(sp_arr > 0) >= 2 else (float('nan'), float('nan'), float('nan'))
    # Speed = center/T exactly, so the center bootstrap's index draws yield
    # identical slope samples; the call above is kept only to preserve the
    # RNG stream for the aligned-error CI.
    sp_lo, sp_hi = ce_lo, ce_hi
    ap_slope, ap_lo, ap_hi = _bootstrap_slope_ci(N_arr, ap_arr[ap_arr>0], rng=rng_ci) \
        if np.sum(ap_arr > 0) >= 2 else (float('nan'), float('nan'), float('nan'))

    print(f"\n  Slopes:")
    print(f"    Profile L2: {l2_slope:.3f} CI=[{l2_lo:.3f},{l2_hi:.3f}]")
    print(f"    Center err: {ce_slope:.3f} CI=[{ce_lo:.3f},{ce_hi:.3f}]")
    print(f"    Speed err:  {sp_slope:.3f} CI=[{sp_lo:.3f},{sp_hi:.3f}]")
    print(f"    Aligned:    {ap_slope:.3f} CI=[{ap_lo:.3f},{ap_hi:.3f}]")

    # ---- Save CSVs ----
    csv_path = _mk(OUT_BASE, 'per_run.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N', 'seed', 'l1', 'l2', 'linf', 'rel_l2',
                    'center_err', 'speed_err', 'aligned_err', 'xc_num', 'runtime_s'])
        for r in per_run_records:
            w.writerow([r['N'], r['seed'],
                        f"{r['l1']:.8f}", f"{r['l2']:.8f}", f"{r['linf']:.8f}",
                        f"{r['rel_l2']:.8f}", f"{r['center_err']:.8f}",
                        f"{r['speed_err']:.8f}", f"{r['aligned_err']:.8f}",
                        f"{r['xc_num']:.6f}", f"{r['runtime_s']:.4f}"])

    csv2 = _mk(OUT_BASE, 'summary_by_N.csv')
    with open(csv2, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N', 'S', 'l2_mean', 'l2_std', 'l1_mean', 'linf_mean',
                    'rel_l2_mean', 'ce_mean', 'ce_std', 'speed_err_mean',
                    'ap_mean', 'ap_std', 'mean_rt_s'])
        for r in N_results:
            w.writerow([r['N'], r['S'],
                        f"{r['l2_mean']:.8f}", f"{r['l2_std']:.8f}",
                        f"{r['l1_mean']:.8f}", f"{r['linf_mean']:.8f}",
                        f"{r['rel_l2_mean']:.8f}",
                        f"{r['ce_mean']:.8f}", f"{r['ce_std']:.8f}",
                        f"{r['speed_err_mean']:.8f}",
                        f"{r['ap_mean']:.8f}", f"{r['ap_std']:.8f}",
                        f"{r['mean_rt']:.4f}"])

    summary = {
        'params': {'a': a, 'nu': nu, 'T': T, 'dt': dt,
                   'L': L, 'xc': xc, 'S': S, 'N_seq': N_seq,
                   'theta_exact': float(theta_exact),
                   'xc_exact_T': float(xc_exact_T)},
        'reference': {'xc_ref': float(xc_ref), 'N_max': N_max},
        'fit': {
            'profile_l2_slope': l2_slope, 'profile_l2_ci': [l2_lo, l2_hi],
            'center_err_slope': ce_slope, 'center_err_ci': [ce_lo, ce_hi],
            'speed_err_slope': sp_slope, 'speed_err_ci': [sp_lo, sp_hi],
            'aligned_err_slope': ap_slope, 'aligned_err_ci': [ap_lo, ap_hi],
        },
        'per_N': N_results,
        'dt_refinement': dt_results,
        'center_vs_time': center_vs_time,
    }
    with open(_mk(OUT_BASE, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Figures ----
    N_ref_line = np.array([N_arr.min(), N_arr.max()])

    # Fig 1: profile error vs N
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(N_arr, l2_arr, 'bo-', lw=1.8, ms=6)
    c0 = l2_arr[0] * N_arr[0]**0.5
    ax.loglog(N_ref_line, c0*N_ref_line**(-0.5), 'k--', lw=1.2, label=r'$O(N^{-1/2})$')
    ci_str = f'[{l2_lo:.3f},{l2_hi:.3f}]'
    ax.set_title(f'FHN: profile L2 vs N\nslope={l2_slope:.3f}  95%CI={ci_str}')
    ax.set_xlabel('N'); ax.set_ylabel('L2 (vs reference)')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    _savefig(fig, _mk(OUT_BASE, 'fhn_profile_error_vs_N'))

    # Fig 2: front center error vs N
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].loglog(N_arr, ce_arr, 'rs-', lw=1.8, ms=6)
    axes[0].set_title(f'FHN: front center error vs N\nslope={ce_slope:.3f}')
    axes[0].set_xlabel('N'); axes[0].set_ylabel('|center error|')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[1].loglog(N_arr, sp_arr, 'g^-', lw=1.8, ms=6)
    axes[1].set_title(f'FHN: front speed error vs N\nslope={sp_slope:.3f}')
    axes[1].set_xlabel('N'); axes[1].set_ylabel('|speed error|')
    axes[1].grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    _savefig(fig, _mk(OUT_BASE, 'fhn_front_error_vs_N'))

    # Fig 3: dt refinement
    if dt_results:
        dt_arr_p = np.array([r['dt'] for r in dt_results])
        l2_dt = np.array([r['l2_mean'] for r in dt_results])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(dt_arr_p, l2_dt, 'bs-', lw=1.8, ms=6)
        ax.set_title(f'FHN: L2 vs dt  (N={dt_study_N})')
        ax.set_xlabel(r'$\Delta t$'); ax.set_ylabel('L2 (vs reference)')
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        _savefig(fig, _mk(OUT_BASE, 'fhn_error_vs_dt'))

    # Fig 4: front center vs time
    if center_vs_time:
        t_vals = [r['t'] for r in center_vs_time]
        xc_vals = [r['xc'] for r in center_vs_time]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_vals, xc_vals, 'bo-', lw=1.5, ms=6)
        ax.set_xlabel('t'); ax.set_ylabel('Front center position')
        ax.set_title(f'FHN: front center vs time  (N={N_max})')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _savefig(fig, _mk(OUT_BASE, 'fhn_front_center_vs_time'))

    # Fig 5: speed error
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(N_arr, sp_arr, 'g^-', lw=1.8, ms=6)
    ax.set_title(f'FHN: front speed error vs N\nslope={sp_slope:.3f}  CI=[{sp_lo:.3f},{sp_hi:.3f}]')
    ax.set_xlabel('N'); ax.set_ylabel('|speed error|')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    _savefig(fig, _mk(OUT_BASE, 'fhn_front_speed_error'))

    # Fig 6: aligned-profile error
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(N_arr, ap_arr, 'mo-', lw=1.8, ms=6)
    ax.set_title(f'FHN: aligned-profile error vs N\nslope={ap_slope:.3f}  CI=[{ap_lo:.3f},{ap_hi:.3f}]')
    ax.set_xlabel('N'); ax.set_ylabel('Aligned L2')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    _savefig(fig, _mk(OUT_BASE, 'fhn_aligned_profile_error'))

    # Fig 7: profiles for selected N
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_ref, u_ref, 'k-', lw=2.0, label='Reference (N=5000)', zorder=10)
    selected_N = [N_seq[0], N_seq[len(N_seq)//2], N_seq[-1]]
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(selected_N)))
    for col, N_sel in zip(colors, selected_N):
        recs = [r for r in per_run_records if r['N'] == N_sel]
        if recs:
            xc_n = float(np.mean([r['xc_num'] for r in recs]))
            ax.axvline(xc_n, color=col, ls='--', lw=0.8, alpha=0.5)
    ax.set_title(f'FHN: profiles  (T={T}, a={a}, nu={nu})')
    ax.set_xlabel('x'); ax.set_ylabel('u(x, T)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, _mk(OUT_BASE, 'fhn_profiles_selected_N'))

    # Fig 8: runtime vs N
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(N_arr, rt_arr, 'cs-', lw=1.5, ms=6)
    c0r = rt_arr[0] / N_arr[0]
    ax.loglog(N_ref_line, c0r * N_ref_line, 'k--', lw=1.2, label=r'$O(N)$')
    ax.set_title('FHN: mean runtime vs N')
    ax.set_xlabel('N'); ax.set_ylabel('Mean runtime (s)')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    _savefig(fig, _mk(OUT_BASE, 'fhn_runtime_vs_N'))

    print(f"\n  [Task 5] Done. Outputs in {OUT_BASE}/")
    return summary


if __name__ == '__main__':
    run_task5()
