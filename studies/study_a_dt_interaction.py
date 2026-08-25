"""Joint relaxation-speed and time-step interaction study for Paper 2.

Motivation.  Consider the displacement of a particle over one loop pass: a
transport step using its carried two-speed label, followed by one independent
Brownian increment.  Conditional on the reconstructed state ``u`` at which the
carried label was drawn,

    E[dX | u]   = u * dt,
    Var(dX | u) = 2*nu*dt + (a**2 - u**2) * dt**2.

The Brownian term is the prescribed physical variance; the second is an excess
variance from sampling the two-speed transport label.  Under a frozen-state
heuristic, accumulating this excess over T/dt steps suggests an O(dt)
contribution that grows with the relaxation speed ``a`` and decreases under
time-step refinement toward a residual finite-N and fit contribution.  This
study measures
the error decomposition and recovered shock width across the a-by-dt matrix to
test whether the observations are consistent with that scaling; it does not by
itself isolate label sampling as the sole cause of the width error.

For this stationary equal-mass shock the label-variance proxy
D_label = (dt/2) * <a**2 - u**2> is an exact analytic design scale, not a
trajectory statistic: every particle carries mass -2A/N, so the sorted
reconstruction is always u_i = A(1 - 2 i / N), giving
<u**2> = A**2 (1/3 + 2/(3 N**2)).  The runtime accumulator is retained because
it becomes genuinely trajectory-dependent for later signed-gradient (e.g.
Gaussian) initial conditions.

Design.  Stationary shock, N=6400 particles, M=400 fixed reconstruction points,
S=50 reused seeds, a in {1.5,2,3,4}, dt in {0.01,0.005,0.0025,0.00125,0.000625}.
The (a=2, dt=0.0025) cell reproduces the production N-refinement study at
N=6400 exactly; a per-seed cross-check asserts this at run time.  The existing
one-dimensional time-step sweep uses M=6400, S=40 and is therefore a
qualitative cross-check only, not interchangeable primary data.
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
    'output', 'final_prepublication_tests', 'gbmc_a_dt_interaction',
)
PRODUCTION_PER_RUN = os.path.join(
    'output', 'final_prepublication_tests',
    'gbmc_production_n_refinement', 'per_run.csv',
)

A_SEQ = [1.5, 2.0, 3.0, 4.0]
DT_SEQ = [0.01, 0.005, 0.0025, 0.00125, 0.000625]
N_FIXED = 6400
S = 50
REUSE_A = 2.0
REUSE_DT = 0.0025
CROSS_CHECK_TOL = 1e-12


def _package_versions():
    import platform
    import numpy as _np
    import scipy as _sp
    out = {'python': platform.python_version(), 'numpy': _np.__version__,
           'scipy': _sp.__version__}
    try:
        import matplotlib as _mpl
        out['matplotlib'] = _mpl.__version__
    except Exception:
        pass
    return out


def _analytic_label_scale(a_rel, dt, N):
    """Exact <a^2 - u^2> and D_label for the stationary equal-mass shock.

    Every particle carries mass -2A/N, so after sorting the reconstructed
    states are u_i = A(1 - 2 i / N) for i=1..N, independent of positions,
    time, seed, or Brownian history. Hence <u_i^2> = A^2 (1/3 + 2/(3 N^2))
    and D_label = (dt/2)[a^2 - <u_i^2>] is an exact design scale, not a
    quantity extracted from evolving trajectories.
    """
    u2 = A * A * (1.0 / 3.0 + 2.0 / (3.0 * N * N))
    return a_rel * a_rel - u2, 0.5 * dt * (a_rel * a_rel - u2)


def _paired_dnu_interval(per_run_rows, dt, a_lo, a_hi, rng, n_boot=5000):
    """Paired seed-level bootstrap CI for nu_fit(a_hi) - nu_fit(a_lo) at fixed dt.

    Valid because the RNG draws are aligned across a at a fixed step count,
    so the two runs share Brownian and label streams and the comparison is
    genuinely paired.  Comparisons across different dt are NOT paired.
    """
    by_seed = {}
    for r in per_run_rows:
        if abs(float(r['dt']) - dt) > 1e-15:
            continue
        by_seed.setdefault(int(float(r['seed'])), {})[float(r['a'])] = float(r['nu_fit'])
    diffs = np.array([d[a_hi] - d[a_lo] for d in by_seed.values()
                      if a_lo in d and a_hi in d])
    if len(diffs) < 3:
        return None
    boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Delta D_label between the two speeds is analytic and deterministic:
    # D_label(a) = (dt/2)[a^2 - <u^2>], so the <u^2> term cancels in the
    # difference, leaving (dt/2)(a_hi^2 - a_lo^2) > 0. Dividing the paired
    # Delta-nu interval by this positive constant preserves order.
    d_dlabel = 0.5 * dt * (a_hi * a_hi - a_lo * a_lo)
    return {'dt': dt, 'a_lo': a_lo, 'a_hi': a_hi, 'n_pairs': int(len(diffs)),
            'mean_dnu': float(diffs.mean()), 'ci_lo': float(lo),
            'ci_hi': float(hi), 'resolves_sign': bool(lo > 0 or hi < 0),
            'delta_D_label': float(d_dlabel),
            'ratio_mean': float(diffs.mean() / d_dlabel),
            'ratio_ci_lo': float(lo / d_dlabel),
            'ratio_ci_hi': float(hi / d_dlabel)}


def _build_metadata(S_use, cross_checked):
    """Study metadata, shared by run_study and the finalize-only refresh."""
    seeds = [BASE_SEED + s for s in range(S_use)] if S_use else None
    return {
        'purpose': ('Joint a-by-dt interaction test of the conditional '
                    'label-variance scale for the stationary shock.'),
        'a_seq': A_SEQ, 'dt_seq': DT_SEQ, 'N': N_FIXED, 'M': N_OUT,
        'S': S_use, 'seeds': seeds,
        'parameters': {'A': A, 'nu': NU, 'T': T, 'L': L, 'xc': XC},
        'package_versions': _package_versions(),
        'seed_design': (f'seeds = {BASE_SEED}..{BASE_SEED + (S_use or 1) - 1}, '
                        'reused at every (a, dt). At a fixed dt the draws are '
                        'aligned across a, so a-comparisons are paired; '
                        'comparisons across dt are not paired.'),
        'cross_check': ('The a=2, dt=0.0025 cell reproduces the production '
                        'N-refinement study at N=6400 per seed; the full '
                        'per-seed (l2, nu_fit, xc_fit) and aggregate '
                        '(E_bias, E_spread, E_total, nu_mean) check is in '
                        f'analysis.json. Exercised: {cross_checked}.'),
        'note_1d_sweep': ('The one-dimensional time-step sweep uses M=6400, '
                          'S=40 and is a qualitative cross-check only.'),
        'analytic_D_label': ('D_label is analytic for this equal-mass shock '
                             '(u_i = A(1 - 2 i / N)); see analysis.json. It is '
                             'not a trajectory statistic here.'),
        'shared_stepper': 'advance_rbgbmc_particles in relaxation_gbmc.py',
        'command': 'python reproduce.py adt',
    }


def finalize_analysis(out_dir, production_per_run=None):
    """Archived, reproducible post-analysis computed from stored per-run data.

    Writes analysis.json with (i) the exact analytic-D_label verification,
    (ii) the full per-seed and aggregate cross-check against the production
    N=6400 study, and (iii) paired seed-level intervals across a at each dt.
    No solver rerun: everything is derived from the stored CSVs.
    """
    prod_per_run = production_per_run or PRODUCTION_PER_RUN
    per_run_rows = list(csv.DictReader(open(os.path.join(out_dir, 'per_run.csv'))))
    summary_rows = list(csv.DictReader(open(os.path.join(out_dir, 'summary.csv'))))

    # (i) analytic D_label identity: stored vs closed form
    max_dev = 0.0
    for r in summary_rows:
        _, d_an = _analytic_label_scale(float(r['a']), float(r['dt']), N_FIXED)
        max_dev = max(max_dev, abs(d_an - float(r['D_label'])))

    # (ii) full cross-check vs production N=6400 (per-seed metrics + aggregates)
    cross = {'exercised': False}
    if os.path.exists(prod_per_run):
        def keyed(rows, keep):
            out = {}
            for r in rows:
                if r.get('failed', '').strip().lower() in ('true', '1'):
                    continue
                if keep(r):
                    out[int(float(r['seed']))] = r
            return out
        adt = keyed(per_run_rows, lambda r: abs(float(r['a']) - REUSE_A) < 1e-12
                    and abs(float(r['dt']) - REUSE_DT) < 1e-12)
        prod = keyed(list(csv.DictReader(open(prod_per_run))),
                     lambda r: int(float(r['N'])) == N_FIXED)
        shared = sorted(set(adt) & set(prod))
        metrics = ['l2', 'nu_fit', 'xc_fit']
        worst = {}
        for m in metrics:
            worst[m] = max((abs(float(adt[s][m]) - float(prod[s][m]))
                            for s in shared
                            if adt[s].get(m, '') != '' and prod[s].get(m, '') != ''),
                           default=None)
        cross = {'exercised': True, 'n_shared_seeds': len(shared),
                 'per_seed_max_abs_delta': worst}

        # aggregate cross-check: the shared cell's ensemble E_bias/E_spread/
        # E_total/nu_mean must match the production N=6400 row and the
        # relaxation-speed (a=2, N=6400) row, since they are the same runs.
        adt_cell = next((r for r in summary_rows
                         if abs(float(r['a']) - REUSE_A) < 1e-12
                         and abs(float(r['dt']) - REUSE_DT) < 1e-12), None)
        agg = {'exercised': False}
        if adt_cell is not None:
            refs = {}
            prod_pern = os.path.join(os.path.dirname(prod_per_run),
                                     'per_N_summary.csv')
            if os.path.exists(prod_pern):
                for r in csv.DictReader(open(prod_pern)):
                    if int(float(r['N'])) == N_FIXED:
                        refs['production_per_N'] = r
            ta_sum = os.path.join(
                os.path.dirname(os.path.dirname(prod_per_run)),
                'gbmc_relaxation_speed_sensitivity', 'summary.csv')
            if os.path.exists(ta_sum):
                for r in csv.DictReader(open(ta_sum)):
                    if (abs(float(r['a']) - REUSE_A) < 1e-12
                            and int(float(r['N'])) == N_FIXED):
                        refs['relaxation_speed'] = r
            deltas = {}
            for name, ref in refs.items():
                deltas[name] = {c: abs(float(adt_cell[c]) - float(ref[c]))
                                for c in ('E_bias', 'E_spread', 'E_total', 'nu_mean')
                                if c in ref and ref[c] not in ('',)}
            agg = {'exercised': bool(refs), 'refs': sorted(refs),
                   'max_abs_delta': deltas,
                   'overall_max': max((v for d in deltas.values()
                                       for v in d.values()), default=None)}
        cross['aggregate_vs_reference_summaries'] = agg

    # (iii) paired intervals across a at each dt (adjacent pairs)
    rng = np.random.default_rng(20260821)
    pairs = list(zip(A_SEQ[:-1], A_SEQ[1:]))
    intervals = [iv for dt in DT_SEQ for (lo, hi) in pairs
                 if (iv := _paired_dnu_interval(per_run_rows, dt, lo, hi, rng))]
    r15 = [iv['ratio_mean'] for iv in intervals
           if iv['a_lo'] == 1.5 and iv['a_hi'] == 2.0]

    agg_overall = cross.get('aggregate_vs_reference_summaries', {}).get('overall_max')
    gate_summary = {
        'analytic_max_abs_dev_below_1e10': bool(max_dev < 1e-10),
        'cross_check_exercised': bool(cross.get('exercised')),
        'cross_check_n_shared_seeds': cross.get('n_shared_seeds', 0),
        'cross_check_per_seed_max_l2':
            (cross.get('per_seed_max_abs_delta') or {}).get('l2'),
        'aggregate_cross_check_overall_max': agg_overall,
        'all_paired_a_intervals_resolve_sign':
            bool(intervals) and all(iv['resolves_sign'] for iv in intervals),
        'ratio_a1p5_2_min': (min(r15) if r15 else None),
        'ratio_a1p5_2_max': (max(r15) if r15 else None),
    }

    analysis = {
        'package_versions': _package_versions(),
        'analytic_D_label': {
            'formula': 'D_label = (dt/2)[a^2 - A^2(1/3 + 2/(3 N^2))]',
            'reconstructed_states': 'u_i = A(1 - 2 i / N), i=1..N (equal-mass)',
            'max_abs_dev_stored_vs_analytic': max_dev,
            'note': ('For the stationary equal-mass shock D_label is an exact '
                     'design scale, not a trajectory statistic.'),
        },
        'cross_check_vs_production_N6400': cross,
        'paired_intervals_across_a_at_fixed_dt': intervals,
        'observed_ratio_a1p5_to_2_across_dt': r15,
        'pairing_note': ('Runs across a at a fixed dt share aligned Brownian and '
                         'label streams and are genuinely paired; comparisons '
                         'across different dt are not paired. Paired differencing '
                         'reduces contributions common to both speeds or only '
                         'weakly dependent on a (reconstruction, finite-N, fit '
                         'bias), isolating the a-dependent broadening. The '
                         'a=1.5->2 ratio Delta-nu / Delta-D_label is observed in '
                         'the range 0.37-0.42 across dt, not an exact constant.'),
        'gate_summary': gate_summary,
    }
    with open(os.path.join(out_dir, 'analysis.json'), 'w') as stream:
        json.dump(analysis, stream, indent=2)

    # item 5: refresh summary.json metadata from stored outputs, no solver rerun
    sj_path = os.path.join(out_dir, 'summary.json')
    if os.path.exists(sj_path):
        sj = json.load(open(sj_path))
        S_stored = int(float(summary_rows[0]['S'])) if summary_rows else None
        sj['metadata'] = _build_metadata(S_stored, cross.get('exercised', False))
        json.dump(sj, open(sj_path, 'w'), indent=2)

    print(f"analytic-D_label max dev {max_dev:.2e}; cross-check shared seeds "
          f"{cross.get('n_shared_seeds')}; aggregate overall max {agg_overall}; "
          f"paired intervals {len(intervals)}; ratio a1.5->2 "
          f"[{gate_summary['ratio_a1p5_2_min']}, {gate_summary['ratio_a1p5_2_max']}]")
    return analysis


def _case(a_rel, dt):
    return f"a={a_rel:g},dt={dt:g}"


def _array_key(a_rel, dt):
    return (f"a{a_rel:g}".replace('.', 'p')
            + f"_dt{dt:g}".replace('.', 'p'))


def _load_production_seed_l2():
    """Return {seed: l2} for the production N=6400 rows, for the cross-check."""
    if not os.path.exists(PRODUCTION_PER_RUN):
        return None
    out = {}
    with open(PRODUCTION_PER_RUN, newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('failed', '').strip().lower() in ('true', '1'):
                continue
            try:
                if int(float(row['N'])) == N_FIXED:
                    out[int(float(row['seed']))] = float(row['l2'])
            except (KeyError, ValueError):
                continue
    return out or None


def run_study(S_override=None, out_base=None, quiet=True):
    S_use = int(S_override) if S_override else S
    out_dir = out_base or OUT_BASE
    os.makedirs(out_dir, exist_ok=True)
    seeds = [BASE_SEED + s for s in range(S_use)]
    x_ref = np.linspace(0.0, L, N_OUT)
    u_ref = u_exact_fn(x_ref)
    dx = float(x_ref[1] - x_ref[0])
    rng_boot = np.random.default_rng(20260821)

    production_l2 = _load_production_seed_l2()
    cross_checked = False

    summaries = []
    per_run = []
    stored_arrays = {'x': x_ref, 'u_exact': u_ref}

    print("Relaxation-speed x time-step interaction")
    print(f"a={A_SEQ}, dt={DT_SEQ}, N={N_FIXED}, M={N_OUT}, S={S_use}, nu={NU}")

    for dt in DT_SEQ:
        for a_rel in A_SEQ:
            profiles = []
            l2s = []
            nu_fits = []
            label_means = []
            for seed in seeds:
                with contextlib.redirect_stdout(io.StringIO()):
                    r = _run_one(N_FIXED, seed, a_rel=a_rel, dt=dt,
                                 collect_label=True)
                if r is None or r.get('failed'):
                    reason = (r or {}).get('reason', 'unknown failure')
                    raise RuntimeError(
                        f"Run failed for a={a_rel}, dt={dt}, seed={seed}: {reason}"
                    )
                profiles.append(r['u_out'])
                l2s.append(r['l2'])
                nu_fits.append(r['nu_fit'])
                label_means.append(r['label_excess_mean'])
                per_run.append({
                    'case': _case(a_rel, dt), 'a': a_rel, 'dt': dt,
                    'N': N_FIXED, 'seed': seed, 'l2': r['l2'],
                    'nu_fit': r['nu_fit'], 'xc_fit': r['xc_fit'],
                    'label_excess_mean': r['label_excess_mean'],
                    'D_label': r['D_label'], 'rho_label': r['rho_label'],
                    'runtime_s': r['runtime_s'],
                })

                # Exact cross-check against the production study at the shared cell.
                if (production_l2 is not None
                        and abs(a_rel - REUSE_A) < 1e-12
                        and abs(dt - REUSE_DT) < 1e-12
                        and seed in production_l2):
                    delta = abs(r['l2'] - production_l2[seed])
                    if delta > CROSS_CHECK_TOL:
                        raise RuntimeError(
                            f"Cross-check FAILED at a=2,dt=0.0025,seed={seed}: "
                            f"study l2={r['l2']:.12g} vs production "
                            f"{production_l2[seed]:.12g} (|delta|={delta:.3e})."
                        )
                    cross_checked = True

            u_arr = np.asarray(profiles)
            u_mean = u_arr.mean(axis=0)
            E_bias = float(np.sqrt(np.sum((u_mean - u_ref) ** 2 * dx)))
            E_spread = float(np.sqrt(np.mean(
                np.sum((u_arr - u_mean[None, :]) ** 2 * dx, axis=1))))
            E_total = float(np.sqrt(np.mean(
                np.sum((u_arr - u_ref[None, :]) ** 2 * dx, axis=1))))
            identity_error = abs(E_total ** 2 - E_bias ** 2 - E_spread ** 2)
            nu_arr = np.asarray(nu_fits)
            label_mean = float(np.mean(label_means))
            D_label = 0.5 * dt * label_mean
            summary = {
                'case': _case(a_rel, dt), 'a': a_rel, 'dt': dt,
                'N': N_FIXED, 'S': S_use,
                'E_bias': E_bias, 'E_spread': E_spread, 'E_total': E_total,
                'identity_error': identity_error,
                'finite_ensemble_scale': E_spread / np.sqrt(S_use - 1),
                'nu_mean': float(nu_arr.mean()),
                'nu_std': float(nu_arr.std()),
                'nu_offset': float(nu_arr.mean()) - NU,
                'label_excess_mean': label_mean,
                'D_label': D_label,
                'rho_label': D_label / NU,
                'n_failed': 0,
            }
            summaries.append(summary)
            stored_arrays[_array_key(a_rel, dt)] = u_arr
            print(f"{summary['case']:18s} bias={E_bias:.5f} spread={E_spread:.5f} "
                  f"total={E_total:.5f} nu={summary['nu_mean']:.5f} "
                  f"D_label={D_label:.3e} rho={summary['rho_label']:.3e}")

    if production_l2 is not None and not cross_checked:
        print("  WARNING: production per_run.csv present but no seed overlap "
              "at a=2,dt=0.0025; exact cross-check not exercised.")
    elif cross_checked:
        print("  Cross-check PASS: a=2,dt=0.0025 reproduces production N=6400 "
              f"per-seed within {CROSS_CHECK_TOL:g}.")

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with open(os.path.join(out_dir, 'per_run.csv'), 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)
    metadata = _build_metadata(S_use, cross_checked)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as stream:
        json.dump({'metadata': metadata, 'results': summaries}, stream, indent=2)
    _savez_deterministic(os.path.join(out_dir, 'profiles.npz'), **stored_arrays)
    print(f"Wrote summary.csv, per_run.csv, summary.json, profiles.npz to {out_dir}")
    finalize_analysis(out_dir)
    return summaries


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=None,
                    help='override number of seeds (pilot use)')
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--finalize-only', action='store_true',
                    help='recompute analysis.json from stored per_run.csv '
                         '(no solver rerun)')
    args = ap.parse_args()
    if args.finalize_only:
        finalize_analysis(args.out or OUT_BASE)
    else:
        run_study(S_override=args.seeds, out_base=args.out)
