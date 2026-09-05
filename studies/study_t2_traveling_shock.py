"""Task 2: Traveling-shock validation for relaxation GBMC.

Manuscript map: `sec:gbmc-traveling`. Production study; outputs are archived
under output/final_prepublication_tests/gbmc_traveling_shock/. The driver
changes only the initial distribution, left state, and diagnostics; the time
stepping is the same shared routine as the stationary studies.

Exact solution: u(x, t) = c - A * tanh(A * (x - x0 - c*t) / (2*nu)),
asymptotic states u_L = c + A, u_R = c - A.

The validation driver changes only the shock offset and diagnostics. It calls
the same shared Lie-splitting particle stepper as the stationary study.
Stops (reports and aborts) on wrong mean wave speed, mass failure, or
subcharacteristic violation.
Output: output/final_prepublication_tests/gbmc_traveling_shock/
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
from relaxation_gbmc import (
    advance_rbgbmc_particles,
    initialize_tanh_shock_particles,
    reconstruct_cumulative_field,
)

OUT_BASE = 'output/final_prepublication_tests/gbmc_traveling_shock'


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


def exact_traveling_shock(x, t, nu, A, c, x0):
    """u(x,t) = c - A*tanh(A*(x - x0 - c*t)/(2*nu))"""
    return c - A * np.tanh(A * (x - x0 - c * t) / (2.0 * nu))


def _run_traveling(N, nu, T, dt, L, A, c, x0, a, seed, output_times=None):
    """
    Run the traveling-shock validation driver.

    Uses ``advance_rbgbmc_particles``, the shared production stepper, with the
    traveling-shock initial state and traveling-wave diagnostics.

    Returns dict with results at each requested output time.
    """
    rng = np.random.default_rng(int(seed))

    x_p, m_p, u_inf = initialize_tanh_shock_particles(
        N, nu, A, x0, mean_level=c
    )
    if not np.all(np.isfinite(x_p)):
        raise RuntimeError("Non-finite particle positions at t=0.")

    n_steps_total = int(round(T / dt))
    if abs(T / dt - n_steps_total) > 1e-8:
        raise ValueError(f"T/dt not integer: T={T}, dt={dt}")

    if output_times is None:
        output_times = [T]
    output_steps = {}
    for t_out in output_times:
        step_idx = int(round(t_out / dt))
        if abs(t_out / dt - step_idx) > 1e-6:
            print(f"  WARNING: output_time {t_out} is not an integer multiple of dt={dt}; rounding.")
        if step_idx <= n_steps_total:
            output_steps[step_idx] = t_out

    results = {}
    total_mass_init = float(m_p.sum())
    wall_start = time.perf_counter()

    run = advance_rbgbmc_particles(
        x_p, m_p, u_inf, nu, a, dt, n_steps_total, rng,
        snapshot_steps=set(output_steps),
    )
    elapsed = time.perf_counter() - wall_start

    for step, t_now in sorted(output_steps.items()):
        x_now, m_now = run['snapshots'][step]
        x_grid = np.linspace(0.0, L, N)
        u_out = reconstruct_cumulative_field(x_now, m_now, u_inf, x_grid)
        u_exact = exact_traveling_shock(x_grid, t_now, nu, A, c, x0)
        dx = float(x_grid[1] - x_grid[0])
        xc_exact = x0 + c * t_now

        diffs = u_out - float(c)
        sign_changes = np.where(np.diff(np.sign(diffs)))[0]
        if len(sign_changes) > 0:
            idx_c = sign_changes[len(sign_changes) // 2]
            xc_num = float(
                x_grid[idx_c]
                + (x_grid[idx_c + 1] - x_grid[idx_c])
                * (-diffs[idx_c] / (diffs[idx_c + 1] - diffs[idx_c] + 1e-15))
            )
        else:
            xc_num = float(x_grid[np.argmin(np.abs(u_out - float(c)))])

        total_mass_now = float(m_now.sum())
        outside = (x_now < 0.0) | (x_now > L)
        mass_outside = float(np.abs(m_now[outside]).sum())
        results[t_now] = {
            'x_grid': x_grid,
            'u_out': u_out,
            'u_exact': u_exact,
            'l1': float(np.sum(np.abs(u_out - u_exact)) * dx),
            'l2': float(np.sqrt(np.sum((u_out - u_exact) ** 2) * dx)),
            'linf': float(np.max(np.abs(u_out - u_exact))),
            'rel_l2': float(
                np.sqrt(np.sum((u_out - u_exact) ** 2) * dx)
                / max(float(np.sqrt(np.sum(u_exact ** 2) * dx)), 1e-15)
            ),
            'xc_exact': xc_exact,
            'xc_num': xc_num,
            'center_err': abs(xc_num - xc_exact),
            'total_mass': total_mass_now,
            'mass_outside': mass_outside,
            'mass_change': abs(total_mass_now - total_mass_init),
            'runtime_s': elapsed,
        }

    return results


def run_task2(nu=0.5, A=1.0, c=0.5, a=2.0, L=8.0, x0=3.0,
              output_times=None, N_seq=None, dt_seq=None,
              S=10, base_seed=42,
              nu_sharp=0.2, run_sharp=True):
    """
    Run Task 2: traveling-shock validation for relaxation GBMC.

    nu=0.5, A=1, c=0.5: u_L=1.5, u_R=-0.5.
    Subcharacteristic: a=2 > max(|u_L|,|u_R|) = 1.5. OK.
    Domain [0,8] with x0=3 so the shock stays inside through T=1 (center moves to 3.5).
    """
    os.makedirs(OUT_BASE, exist_ok=True)
    if output_times is None:
        output_times = [0.25, 0.5, 1.0]
    if N_seq is None:
        N_seq = [100, 200, 400, 800, 1600, 3200]
    if dt_seq is None:
        dt_seq = [0.02, 0.01, 0.005, 0.0025]

    T_max = max(output_times)

    # Validate subcharacteristic
    u_L = c + A; u_R = c - A
    print(f"  Traveling shock: u_L={u_L}, u_R={u_R}, a={a}")
    if a <= max(abs(u_L), abs(u_R)):
        raise ValueError(
            f"Subcharacteristic condition violated: "
            f"a={a} <= max(|u_L|={abs(u_L)},|u_R|={abs(u_R)}) = {max(abs(u_L),abs(u_R))}"
        )
    print(f"  Subcharacteristic OK: a={a} > {max(abs(u_L),abs(u_R))}")

    # Check dt divides all output times and T_max
    valid_dt = []
    for dt in dt_seq:
        ok = all(abs(t_out/dt - round(t_out/dt)) < 1e-9 for t_out in output_times)
        if not ok:
            print(f"  WARNING: dt={dt} does not evenly divide all output times; skipping.")
        else:
            valid_dt.append(dt)
    dt_seq = valid_dt

    stop_triggered = False
    stop_reasons = []

    # ---- N-refinement study (multiple seeds, dt=0.005, T_max) ----
    dt_nr = 0.005
    if dt_nr not in dt_seq:
        n_steps_check = T_max / dt_nr
        if abs(n_steps_check - round(n_steps_check)) < 1e-9:
            dt_seq_nr = [dt_nr]
        else:
            dt_seq_nr = dt_seq[:1]
    else:
        dt_seq_nr = [dt_nr]
    dt_nr_use = dt_seq_nr[0]

    N_results = []
    per_run_records = []

    print(f"\n{'='*60}\n  Task 2: Traveling-shock N-refinement  (dt={dt_nr_use})\n{'='*60}")
    for N in N_seq:
        l2_by_T = {t: [] for t in output_times}
        l1_by_T = {t: [] for t in output_times}
        linf_by_T = {t: [] for t in output_times}
        ce_by_T = {t: [] for t in output_times}  # center error
        rt_list = []
        mass_ok = True

        for rep in range(S):
            seed = base_seed + rep
            try:
                t0 = time.perf_counter()
                results = _run_traveling(N, nu, T_max, dt_nr_use, L, A, c, x0, a, seed,
                                         output_times=output_times)
                rt_list.append(time.perf_counter() - t0)
            except RuntimeError as e:
                stop_reasons.append(f"N={N} seed={seed}: {e}")
                print(f"  STOP: {e}")
                stop_triggered = True
                break

            for t_out, res in results.items():
                if abs(res['mass_change']) > 0.1 * abs(-2.0 * A):
                    mass_ok = False
                    stop_reasons.append(f"Mass conservation failed at N={N}, t={t_out}: "
                                        f"change={res['mass_change']:.4e}")
                l2_by_T[t_out].append(res['l2'])
                l1_by_T[t_out].append(res['l1'])
                linf_by_T[t_out].append(res['linf'])
                ce_by_T[t_out].append(res['center_err'])
                per_run_records.append({
                    'N': N, 'dt': dt_nr_use, 'seed': seed,
                    't_out': t_out,
                    'l2': res['l2'], 'l1': res['l1'], 'linf': res['linf'],
                    'rel_l2': res['rel_l2'],
                    'center_err': res['center_err'],
                    'xc_exact': res['xc_exact'], 'xc_num': res['xc_num'],
                    'mass_change': res['mass_change'],
                    'mass_outside': res['mass_outside'],
                })

            if (rep + 1) % 5 == 0:
                t_key = output_times[-1]
                if l2_by_T[t_key]:
                    print(f"    N={N:5d}  rep={rep+1}/{S}  "
                          f"L2(T={t_key})={np.mean(l2_by_T[t_key]):.4f}  "
                          f"center_err={np.mean(ce_by_T[t_key]):.4f}")

        if not mass_ok:
            stop_triggered = True
            print(f"  STOP: mass conservation failed at N={N}")
            break

        if stop_triggered:
            break

        for t_out in output_times:
            if not l2_by_T[t_out]:
                continue
            N_results.append({
                'N': N, 'dt': dt_nr_use, 't_out': t_out,
                'l2_mean': float(np.mean(l2_by_T[t_out])),
                'l2_std': float(np.std(l2_by_T[t_out])),
                'l1_mean': float(np.mean(l1_by_T[t_out])),
                'linf_mean': float(np.mean(linf_by_T[t_out])),
                'ce_mean': float(np.mean(ce_by_T[t_out])),
                'ce_std': float(np.std(ce_by_T[t_out])),
                'S': len(l2_by_T[t_out]),
                'mean_rt': float(np.mean(rt_list)) if rt_list else float('nan'),
            })
            print(f"  N={N:5d}  T={t_out}  "
                  f"L2={N_results[-1]['l2_mean']:.4f}±{N_results[-1]['l2_std']:.4f}  "
                  f"center_err={N_results[-1]['ce_mean']:.4f}")

    # ---- dt-refinement study (N=1600 fixed) ----
    N_dt = 1600
    dt_results = []
    if not stop_triggered:
        print(f"\n{'='*60}\n  Task 2: Traveling-shock dt-refinement  (N={N_dt})\n{'='*60}")
        for dt in dt_seq:
            l2_list = []; ce_list = []
            for rep in range(S):
                seed = base_seed + 200 + rep
                try:
                    res_all = _run_traveling(N_dt, nu, T_max, dt, L, A, c, x0, a, seed,
                                             output_times=[T_max])
                    res = res_all[T_max]
                    l2_list.append(res['l2']); ce_list.append(res['center_err'])
                except Exception as e:
                    print(f"  WARNING dt={dt} seed={seed}: {e}")

            if l2_list:
                dt_results.append({
                    'dt': dt, 'N': N_dt,
                    'l2_mean': float(np.mean(l2_list)),
                    'l2_std': float(np.std(l2_list)),
                    'ce_mean': float(np.mean(ce_list)),
                })
                print(f"  dt={dt:.5f}  L2={dt_results[-1]['l2_mean']:.4f}  "
                      f"center_err={dt_results[-1]['ce_mean']:.4f}")

    # ---- Sharp layer: nu=0.2 ----
    sharp_results = []
    if run_sharp and not stop_triggered:
        nu2 = nu_sharp; A2 = A; c2 = c; x02 = x0
        u_L2 = c2 + A2; u_R2 = c2 - A2
        a2 = a  # 2 > max(|1.5|, |0.5|) still
        print(f"\n{'='*60}\n  Task 2: Sharp layer nu={nu2}  N-refinement\n{'='*60}")
        N_sharp = [200, 400, 800, 1600]
        for N in N_sharp:
            l2_list = []
            for rep in range(S):
                seed = base_seed + 400 + rep
                try:
                    res_all = _run_traveling(N, nu2, T_max, 0.005, L, A2, c2, x02, a2, seed,
                                             output_times=[T_max])
                    l2_list.append(res_all[T_max]['l2'])
                except Exception as e:
                    print(f"  WARNING sharp N={N} seed={seed}: {e}")
            if l2_list:
                sharp_results.append({
                    'N': N, 'nu': nu2,
                    'l2_mean': float(np.mean(l2_list)),
                    'l2_std': float(np.std(l2_list)),
                })
                print(f"  N={N:5d}  L2={sharp_results[-1]['l2_mean']:.4f}")

    # ---- Save CSVs ----
    csv_path = _mk(OUT_BASE, 'per_run.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N','dt','seed','t_out','l2','l1','linf','rel_l2','center_err',
                    'xc_exact','xc_num','mass_change','mass_outside'])
        for r in per_run_records:
            w.writerow([r['N'],r['dt'],r['seed'],r['t_out'],
                        f"{r['l2']:.8f}",f"{r['l1']:.8f}",f"{r['linf']:.8f}",
                        f"{r['rel_l2']:.8f}",f"{r['center_err']:.8f}",
                        f"{r['xc_exact']:.6f}",f"{r['xc_num']:.6f}",
                        f"{r['mass_change']:.4e}",f"{r['mass_outside']:.4e}"])

    # ---- Save JSON ----
    # Fit N slope at T_max
    T_key = output_times[-1]
    nr_Tmax = [r for r in N_results if abs(r['t_out'] - T_key) < 1e-9]
    n_arr = np.array([r['N'] for r in nr_Tmax], dtype=float)
    l2_arr = np.array([r['l2_mean'] for r in nr_Tmax])
    # Per-seed L2 at T_max, grouped by N, for the realization-level bootstrap.
    per_N_l2 = {}
    for r in per_run_records:
        if abs(r['t_out'] - T_key) < 1e-9:
            per_N_l2.setdefault(int(r['N']), []).append(float(r['l2']))
    if len(n_arr) >= 2:
        valid = l2_arr > 0
        if valid.sum() >= 2:
            N_fit = [int(n) for n, v in zip(n_arr, valid) if v]
            # Realization-level bootstrap: resample the per-seed errors within
            # each N and refit, the same procedure as the FHN and production
            # studies (Section on the ensemble bootstrap in the paper).
            N_slope, N_lo, N_hi = _realization_boot(per_N_l2, N_fit)
        else:
            N_slope = float('nan'); N_lo = float('nan'); N_hi = float('nan')
    else:
        N_slope = float('nan'); N_lo = float('nan'); N_hi = float('nan')

    summary = {
        'stop_triggered': stop_triggered,
        'stop_reasons': stop_reasons,
        'params': {'nu': nu, 'A': A, 'c': c, 'a': a, 'L': L, 'x0': x0,
                   'u_L': float(u_L), 'u_R': float(u_R),
                   'output_times': output_times, 'S': S},
        'N_refinement': N_results,
        'N_slope_T_max': N_slope,
        'N_slope_ci': [N_lo, N_hi],
        'dt_refinement': dt_results,
        'sharp_layer': sharp_results,
    }
    with open(_mk(OUT_BASE, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Figures ----
    if N_results:
        # Fig 1: profiles by time (largest N)
        N_max = max(r['N'] for r in N_results)
        fig, axes = plt.subplots(1, len(output_times), figsize=(5*len(output_times), 4))
        if len(output_times) == 1:
            axes = [axes]
        for ax, t_out in zip(axes, output_times):
            recs = [r for r in per_run_records if r['N'] == N_max and abs(r['t_out']-t_out)<1e-9]
            if recs:
                xg = np.linspace(0.0, L, N_max)
                u_ex = exact_traveling_shock(xg, t_out, nu, A, c, x0)
                ax.plot(xg, u_ex, 'k-', lw=2, label='Exact')
                ax.set_title(f'T={t_out}  N={N_max}')
                ax.set_xlabel('x'); ax.set_ylabel('u')
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
        fig.suptitle(f'Traveling shock profiles (c={c}, nu={nu}, A={A})', fontsize=10)
        fig.tight_layout()
        _savefig(fig, _mk(OUT_BASE, 'gbmc_traveling_profiles_by_time'))

        # Fig 2: L2 error vs N (T=T_max)
        if nr_Tmax:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.loglog(n_arr, l2_arr, 'bo-', lw=1.8, ms=6, label=f'L2 (T={T_key})')
            n_ref = np.array([n_arr.min(), n_arr.max()])
            c0 = l2_arr[0] * n_arr[0]**0.5
            ax.loglog(n_ref, c0*n_ref**(-0.5), 'k--', lw=1, label=r'$O(N^{-1/2})$')
            ax.set_title(f'Traveling shock: L2 vs N  (nu={nu}, c={c}, T={T_key})\n'
                         f'Fitted slope={N_slope:.3f}  CI=[{N_lo:.3f},{N_hi:.3f}]')
            ax.set_xlabel('N'); ax.set_ylabel('L2 error')
            ax.legend(); ax.grid(True, which='both', alpha=0.3)
            fig.tight_layout()
            _savefig(fig, _mk(OUT_BASE, 'gbmc_traveling_error_vs_N'))

        # Fig 3: center error vs N
        ce_arr = np.array([r['ce_mean'] for r in nr_Tmax])
        if len(ce_arr) > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.loglog(n_arr, ce_arr, 'rs-', lw=1.8, ms=6)
            ax.set_title(f'Traveling shock: center error vs N  (T={T_key})')
            ax.set_xlabel('N'); ax.set_ylabel('|center error| (m)')
            ax.grid(True, which='both', alpha=0.3)
            fig.tight_layout()
            _savefig(fig, _mk(OUT_BASE, 'gbmc_traveling_center_error'))

        # Fig 4: center vs time (N=N_max, one seed)
        if not stop_triggered:
            T_span = max(output_times)
            dt_plot = 0.005
            if all(abs(t/dt_plot - round(t/dt_plot)) < 1e-9 for t in output_times):
                try:
                    res_track = _run_traveling(N_max, nu, T_span, dt_plot, L, A, c, x0, a,
                                               base_seed, output_times=output_times)
                    t_track = sorted(res_track.keys())
                    xc_num_list = [res_track[t]['xc_num'] for t in t_track]
                    xc_ex_list  = [x0 + c*t for t in t_track]
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                    ax1.plot(t_track, xc_ex_list, 'k-', lw=2, label='Exact center')
                    ax1.plot(t_track, xc_num_list, 'bo-', lw=1.5, ms=6, label='GBMC center')
                    ax1.set_xlabel('t'); ax1.set_ylabel('Shock center')
                    ax1.set_title(f'Center vs time  N={N_max}')
                    ax1.legend(); ax1.grid(alpha=0.3)
                    ax2.plot(t_track, [abs(n-e) for n,e in zip(xc_num_list, xc_ex_list)],
                             'ro-', lw=1.5, ms=6)
                    ax2.set_xlabel('t'); ax2.set_ylabel('|center error|')
                    ax2.set_title('Center error vs time')
                    ax2.grid(alpha=0.3)
                    fig.suptitle(f'Traveling shock center tracking  (c={c}, nu={nu}, N={N_max})')
                    fig.tight_layout()
                    _savefig(fig, _mk(OUT_BASE, 'gbmc_traveling_center_vs_time'))
                    # Speed error figure
                    if len(t_track) >= 2:
                        xc_num_arr = np.array(xc_num_list)
                        t_arr_tr = np.array(t_track)
                        if len(t_arr_tr) >= 2:
                            speed_num = float(np.polyfit(t_arr_tr, xc_num_arr, 1)[0])
                            speed_err = abs(speed_num - c)
                            fig2, ax2 = plt.subplots(figsize=(5,3))
                            ax2.bar(['GBMC speed', 'Exact speed'], [speed_num, c],
                                    color=['steelblue','orange'])
                            ax2.set_title(f'Wave speed: GBMC={speed_num:.4f}  exact={c:.4f}  err={speed_err:.4f}')
                            ax2.set_ylabel('Wave speed')
                            fig2.tight_layout()
                            _savefig(fig2, _mk(OUT_BASE, 'gbmc_traveling_speed_error'))
                except Exception as e:
                    print(f"  WARNING: center tracking failed: {e}")

        # Fig 5: dt refinement
        if dt_results:
            dt_arr_p = np.array([r['dt'] for r in dt_results])
            l2_dt = np.array([r['l2_mean'] for r in dt_results])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.loglog(dt_arr_p, l2_dt, 'rs-', lw=1.8, ms=6)
            ax.set_title(f'Traveling shock: L2 vs dt  (N={N_dt}, T={T_max})')
            ax.set_xlabel(r'$\Delta t$'); ax.set_ylabel('L2 error')
            ax.grid(True, which='both', alpha=0.3)
            fig.tight_layout()
            _savefig(fig, _mk(OUT_BASE, 'gbmc_traveling_error_vs_dt'))

        # Fig 6: sharp layer
        if sharp_results:
            n_s = np.array([r['N'] for r in sharp_results], dtype=float)
            l2_s = np.array([r['l2_mean'] for r in sharp_results])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.loglog(n_s, l2_s, 'g^-', lw=1.8, ms=6, label=f'nu={nu_sharp}')
            ax.loglog(n_arr, l2_arr, 'bo-', lw=1.8, ms=6, label=f'nu={nu}')
            ax.set_title(f'Traveling shock: sharp layer comparison  (T={T_max})')
            ax.set_xlabel('N'); ax.set_ylabel('L2 error')
            ax.legend(); ax.grid(True, which='both', alpha=0.3)
            fig.tight_layout()
            _savefig(fig, _mk(OUT_BASE, 'gbmc_traveling_sharp_layer'))

    if stop_triggered:
        print(f"\n  [Task 2] STOPPED: {stop_reasons}")
    else:
        print(f"\n  [Task 2] Complete. Outputs in {OUT_BASE}/")

    return summary


def _realization_boot(per_N, N_seq, n_boot=5000, seed=12345):
    """Realization-level bootstrap of a log-log slope: resample the per-seed
    values within each N (with replacement), recompute the ensemble mean, refit.
    Matches the FHN and production studies so all realization-level intervals use
    the same procedure."""
    rng = np.random.default_rng(seed)
    logN = np.log(np.array(N_seq, dtype=float))
    arrs = {N: np.asarray(per_N[N], dtype=float) for N in N_seq}
    def _slope(means):
        return float(np.polyfit(logN, np.log(np.array(means)), 1)[0])
    base = _slope([arrs[N].mean() for N in N_seq])
    bs = []
    for _ in range(n_boot):
        means = [float(arrs[N][rng.integers(0, len(arrs[N]), size=len(arrs[N]))].mean())
                 for N in N_seq]
        bs.append(_slope(means))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return base, float(lo), float(hi)


if __name__ == '__main__':
    run_task2()
