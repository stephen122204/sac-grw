"""Ordering pilot: carried-label schedule versus redraw-after-diffusion.

Purpose.  The multi-viscosity sweep resolved a small positive fitted-viscosity
offset of the conditional-mean CONTROL at the three smallest viscosities
(about +1.2%, +3.0%, +5.8% of nu at nu = 0.1, 0.05, 0.025 with dt = 0.0025).
The control has no label sampling, so that residual must come from some other
mechanism: the carried-velocity timing (the transport velocity is set from the
post-transport reconstruction, then a Brownian displacement intervenes before
it is used), the time discretization at fixed dt/nu, the nu-dependent
evaluation-window/fit design, or reconstruction.  This pilot isolates the
timing candidate: an opt-in stepper schedule sets the transport velocity from
the post-Brownian reconstruction instead
(``redraw_after_diffusion=True``), changing nothing else.

Design.
  nu in {0.1, 0.05, 0.025}   (subset of the multi-viscosity canonical list)
  dt in {0.0025, 0.00125}
  schedules: 'carried' (production) and 'postdiff' (redraw after diffusion)
  arms: conditional-mean control first; two-speed a=2, a=4 appended only when
        the control comparison justifies extension (--arms).
  N = 6400, S = 50, same evaluation windows and strict fit bounds as the
  multi-viscosity sweep.

Seeds reuse the multi-viscosity value-keyed streams
(SeedSequence([base, round(nu*1e6), seed_idx])), so
  * at dt = 0.0025 the 'carried' cells REPRODUCE the archived multi-viscosity
    cells bit-for-bit (asserted at run time for the control), and
  * at fixed (nu, dt) the two schedules share Brownian streams and are paired
    through common random numbers (aligned by sorted rank; this is a CRN
    pairing, not an exact cancellation).
Comparisons across dt or nu are not paired.

The 'postdiff' schedule is an opt-in diagnostic; the production path is
bit-identical with the flag off (pinned by the test suite).
"""

import contextlib
import csv
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxation_gbmc import (
    advance_rbgbmc_particles,
    initialize_tanh_shock_particles,
    reconstruct_cumulative_field,
)
from studies.study_gbmc_production_n_refinement import (
    A, T, XC, N_OUT, _fit_tanh, _savez_deterministic,
)
from studies import study_multiviscosity_sweep as mv

OUT_BASE = os.path.join(
    'output', 'final_prepublication_tests', 'gbmc_ordering_pilot')
NU_SEQ = [0.1, 0.05, 0.025]
DT_SEQ = [0.0025, 0.00125]
SCHEDULES = ['carried', 'postdiff']
N_FIXED = 6400
S = 50
ARM_DEFS = {'cond_mean': ('conditional_mean', 2.0),
            'two_speed_a2': ('two_speed', 2.0),
            'two_speed_a4': ('two_speed', 4.0)}
# The staged decision rule ran the control first; the completed archive
# includes the two-speed extension, so the default now matches it.
DEFAULT_ARMS = ['cond_mean', 'two_speed_a2', 'two_speed_a4']
CROSS_CHECK_TOL = 0.0   # carried dt=0.0025 must equal the multinu archive


def _fingerprint(S_use, arms):
    return {'N': N_FIXED, 'T': T, 'N_OUT': N_OUT, 'A': A, 'XC': XC,
            'S': int(S_use), 'nu_seq': list(NU_SEQ), 'dt_seq': list(DT_SEQ),
            'schedules': list(SCHEDULES), 'arms': list(arms),
            'window_shock_widths': mv.WINDOW_SHOCK_WIDTHS,
            'fit_bounds': 'nu_lo=nu/50, nu_hi=max(2, 20*nu); at_bound within 2%',
            'seed_design': ('multi-viscosity streams: SeedSequence([base, '
                            'round(nu*1e6), seed_idx]), bases 910000/920000; '
                            'schedules share streams at fixed (nu, seed)')}


def _run_cell_member(nu, dt, schedule, arm, seed_idx, x_out, u_ref, dx):
    transport, a_rel = ARM_DEFS[arm]
    x0, m0, u_left = initialize_tanh_shock_particles(N_FIXED, nu, A, XC)
    n_steps = int(round(T / dt))
    rng_label, rng_brownian = mv._rngs(nu, seed_idx)
    t0 = time.perf_counter()
    run = advance_rbgbmc_particles(
        x0, m0, u_left, nu, a_rel, dt, n_steps, rng_label,
        rng_brownian=rng_brownian,
        conditional_mean_transport=(transport == 'conditional_mean'),
        redraw_after_diffusion=(schedule == 'postdiff'))
    u_out = reconstruct_cumulative_field(run['x'], run['m'], u_left, x_out)
    l2 = float(np.sqrt(np.sum((u_out - u_ref) ** 2) * dx))
    nu_lo, nu_hi = mv._fit_bounds(nu)
    xc_fit, nu_fit, A_fit = _fit_tanh(x_out, u_out, A, XC, nu,
                                      nu_lo=nu_lo, nu_hi=nu_hi, strict=True)
    at_bound = bool(nu_fit <= 1.02 * nu_lo or nu_fit >= 0.98 * nu_hi)
    return {'u_out': u_out, 'l2': l2, 'nu_fit': nu_fit, 'A_fit': A_fit,
            'at_bound': at_bound, 'runtime_s': time.perf_counter() - t0}


def _paired_ci(diffs, rng, n_boot=5000):
    diffs = np.asarray(diffs)
    boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'mean': float(diffs.mean()), 'ci_lo': float(lo), 'ci_hi': float(hi),
            'resolves_sign': bool(lo > 0 or hi < 0)}


def _multinu_reference():
    """Per-seed nu_fit of the archived multi-viscosity cells (for the exact
    carried/dt=0.0025 cross-check)."""
    path = os.path.join(mv.OUT_BASE, 'per_run.csv')
    if not os.path.exists(path):
        return None
    ref = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            ref[(float(row['nu']), row['arm'], int(row['seed_idx']))] = \
                float(row['nu_fit'])
    return ref or None


def run_study(S_override=None, out_base=None, arms=None):
    S_use = int(S_override) if S_override else S
    arms = list(arms) if arms else list(DEFAULT_ARMS)
    unknown = sorted(set(arms) - set(ARM_DEFS))
    if unknown:
        raise ValueError(f"Unknown arms {unknown}; choose from {sorted(ARM_DEFS)}")
    fp = _fingerprint(S_use, arms)
    out_dir = out_base or OUT_BASE
    os.makedirs(out_dir, exist_ok=True)

    windows = {nu: mv._window(nu) for nu in NU_SEQ}
    urefs = {nu: mv.u_exact_nu(windows[nu][0], nu) for nu in NU_SEQ}
    mn_ref = _multinu_reference()

    manifest_path = os.path.join(out_dir, 'manifest.json')
    done = set()
    per_run = []
    fit_by = {}   # (nu, dt, schedule, arm) -> {seed_idx: nu_fit}
    prof = {}

    def _cell_npz(nu_c, dt_c, sch_c, arm_c):
        stem = f'cell_{arm_c}_{sch_c}_nu{nu_c:g}_dt{dt_c:g}'.replace('.', 'p')
        return os.path.join(out_dir, stem + '.npz')

    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        stored = manifest.get('config')
        # arms may legitimately grow (control first, then two-speed):
        # everything except the arm list must match, and the stored arms must
        # be a subset of the requested ones.
        cmp_stored = {k: v for k, v in (stored or {}).items() if k != 'arms'}
        cmp_now = {k: v for k, v in fp.items() if k != 'arms'}
        if stored is None or cmp_stored != cmp_now or \
                not set(stored['arms']) <= set(arms):
            raise RuntimeError(
                f"Refusing to resume {out_dir}: stored configuration "
                f"fingerprint differs.\n  stored:  {stored}\n  current: {fp}")
        claimed = set(tuple(c) for c in manifest.get('done', []))
        if os.path.exists(os.path.join(out_dir, 'per_run.csv')):
            per_run = list(csv.DictReader(open(os.path.join(out_dir, 'per_run.csv'))))
        for row in per_run:
            key = (float(row['nu']), float(row['dt']), row['schedule'], row['arm'])
            fit_by.setdefault(key, {})[int(row['seed_idx'])] = float(row['nu_fit'])
        for cell in claimed:
            nu_c, dt_c, sch_c, arm_c = float(cell[0]), float(cell[1]), cell[2], cell[3]
            key = (nu_c, dt_c, sch_c, arm_c)
            path = _cell_npz(nu_c, dt_c, sch_c, arm_c)
            if not os.path.exists(path):
                continue
            arr = np.load(path)['profiles']
            if arr.shape != (S_use, N_OUT):
                continue
            if sorted(fit_by.get(key, {})) != list(range(S_use)):
                continue
            prof[key] = arr
            done.add(key)
        per_run = [r for r in per_run
                   if (float(r['nu']), float(r['dt']), r['schedule'], r['arm']) in done]
        fit_by = {k: v for k, v in fit_by.items() if k in done}
        print(f"resuming: {len(done)}/{len(claimed)} claimed cells are complete")

    print(f"Ordering pilot: nu={NU_SEQ}, dt={DT_SEQ}, schedules={SCHEDULES}, "
          f"arms={arms}, N={N_FIXED}, M={N_OUT}, S={S_use}")
    n_checked = 0
    for nu in NU_SEQ:
        x_out, dx = windows[nu]
        u_ref = urefs[nu]
        for dt in DT_SEQ:
            for schedule in SCHEDULES:
                for arm in arms:
                    key = (nu, dt, schedule, arm)
                    if key in done:
                        continue
                    profiles, fits, n_bound = [], {}, 0
                    for s in range(S_use):
                        r = _run_cell_member(nu, dt, schedule, arm, s,
                                             x_out, u_ref, dx)
                        profiles.append(r['u_out'])
                        fits[s] = r['nu_fit']
                        n_bound += int(r['at_bound'])
                        per_run.append({'nu': nu, 'dt': dt,
                                        'schedule': schedule, 'arm': arm,
                                        'seed_idx': s, 'l2': r['l2'],
                                        'nu_fit': r['nu_fit'],
                                        'A_fit': r['A_fit'],
                                        'at_bound': int(r['at_bound']),
                                        'runtime_s': r['runtime_s']})
                        # Exact cross-check: carried schedule at dt=0.0025
                        # shares streams and schedule with the archived
                        # multi-viscosity sweep, so it must reproduce it.
                        if (mn_ref is not None and schedule == 'carried'
                                and abs(dt - 0.0025) < 1e-15
                                and (nu, arm, s) in mn_ref):
                            delta = abs(r['nu_fit'] - mn_ref[(nu, arm, s)])
                            if delta > CROSS_CHECK_TOL:
                                raise RuntimeError(
                                    f"Cross-check FAILED at nu={nu}, arm={arm}, "
                                    f"seed={s}: pilot {r['nu_fit']!r} vs "
                                    f"multinu {mn_ref[(nu, arm, s)]!r}")
                            n_checked += 1
                    prof[key] = np.asarray(profiles)
                    fit_by[key] = fits
                    done.add(key)
                    with open(os.path.join(out_dir, 'per_run.csv'), 'w',
                              newline='') as fh:
                        w = csv.DictWriter(fh, fieldnames=list(per_run[0]))
                        w.writeheader(); w.writerows(per_run)
                    _savez_deterministic(_cell_npz(nu, dt, schedule, arm),
                                         profiles=prof[key])
                    stored_arms = sorted({k[3] for k in done} | set(arms))
                    json.dump({'config': _fingerprint(S_use, stored_arms),
                               'done': sorted([list(map(str, k)) for k in done])},
                              open(manifest_path, 'w'))
                    mean_fit = float(np.mean(list(fits.values())))
                    print(f"  nu={nu:<6g} dt={dt:<8g} {schedule:<8s} "
                          f"{arm:<13s} nu_hat={mean_fit:.5f} "
                          f"off={mean_fit - nu:+.5f} "
                          f"rel={(mean_fit - nu) / nu:+.3%} "
                          f"at_bound={n_bound}/{S_use}")
    if mn_ref is not None and n_checked:
        print(f"  Cross-check PASS: {n_checked} freshly run carried/dt=0.0025 "
              "members reproduce the archived multi-viscosity cells exactly.")

    # Definitive archive-vs-archive cross-check, re-verified on EVERY
    # invocation (including analysis-only resumes): every stored
    # carried/dt=0.0025 member must equal the archived multi-viscosity value.
    n_archive_checked = 0
    if mn_ref is not None:
        for (nu_c, dt_c, sch_c, arm_c), fits in fit_by.items():
            if sch_c != 'carried' or abs(dt_c - 0.0025) > 1e-15:
                continue
            for s, value in fits.items():
                if (nu_c, arm_c, s) in mn_ref:
                    if value != mn_ref[(nu_c, arm_c, s)]:
                        raise RuntimeError(
                            f"Archive cross-check FAILED at nu={nu_c}, "
                            f"arm={arm_c}, seed={s}: pilot {value!r} vs "
                            f"multinu {mn_ref[(nu_c, arm_c, s)]!r}")
                    n_archive_checked += 1
        print(f"  Archive cross-check PASS: {n_archive_checked} stored "
              "members match the multi-viscosity archive exactly.")

    # ------------------------------------------------------------- analysis
    rng = np.random.default_rng(20260824)
    completed_arms = sorted({k[3] for k in done},
                            key=list(ARM_DEFS).index)
    summary, schedule_contrasts = [], []
    for nu in NU_SEQ:
        for dt in DT_SEQ:
            for arm in completed_arms:
                for schedule in SCHEDULES:
                    key = (nu, dt, schedule, arm)
                    if key not in fit_by:
                        continue
                    fits = np.array([fit_by[key][s] for s in sorted(fit_by[key])])
                    off = fits - nu
                    boot = np.array([off[rng.integers(0, len(off), len(off))].mean()
                                     for _ in range(5000)])
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    summary.append({'nu': nu, 'dt': dt, 'schedule': schedule,
                                    'arm': arm, 'S': S_use,
                                    'nu_mean': float(fits.mean()),
                                    'nu_offset_abs': float(off.mean()),
                                    'nu_offset_abs_ci_lo': float(lo),
                                    'nu_offset_abs_ci_hi': float(hi),
                                    'nu_offset_rel': float(off.mean() / nu),
                                    'offset_ci_excludes_zero':
                                        bool(lo > 0 or hi < 0),
                                    'nu_std': float(fits.std())})
                ka = (nu, dt, 'postdiff', arm)
                kb = (nu, dt, 'carried', arm)
                if ka in fit_by and kb in fit_by:
                    seeds = sorted(set(fit_by[ka]) & set(fit_by[kb]))
                    d = np.array([fit_by[ka][s] - fit_by[kb][s] for s in seeds])
                    schedule_contrasts.append(
                        {'nu': nu, 'dt': dt, 'arm': arm,
                         'contrast': 'postdiff - carried',
                         'n_pairs': len(seeds), **_paired_ci(d, rng)})

    # within-schedule two-speed-minus-control label contrasts, with the
    # analytic D_label at that step size for reference
    label_contrasts = []
    for nu in NU_SEQ:
        for dt in DT_SEQ:
            for schedule in SCHEDULES:
                kc = (nu, dt, schedule, 'cond_mean')
                if kc not in fit_by:
                    continue
                for arm, a_rel in (('two_speed_a2', 2.0), ('two_speed_a4', 4.0)):
                    kt = (nu, dt, schedule, arm)
                    if kt not in fit_by:
                        continue
                    seeds = sorted(set(fit_by[kt]) & set(fit_by[kc]))
                    d = np.array([fit_by[kt][s] - fit_by[kc][s] for s in seeds])
                    d_lab = 0.5 * dt * (a_rel ** 2
                                        - A * A * (1.0 / 3.0
                                                   + 2.0 / (3.0 * N_FIXED ** 2)))
                    iv = _paired_ci(d, rng)
                    label_contrasts.append(
                        {'nu': nu, 'dt': dt, 'schedule': schedule, 'arm': arm,
                         'contrast': f'{arm} - cond_mean',
                         'n_pairs': len(seeds), **iv,
                         'D_label': float(d_lab),
                         'excess_over_D_label': float(iv['mean'] / d_lab)})

    out = {'metadata': {
                'purpose': ('carried-label vs redraw-after-diffusion timing '
                            'pilot for the low-viscosity control residual'),
                'nu_seq': NU_SEQ, 'dt_seq': DT_SEQ, 'schedules': SCHEDULES,
                'arms': completed_arms, 'N': N_FIXED, 'M': N_OUT, 'S': S_use,
                'T': T, 'A': A,
                'n_failed': sum(1 for r in per_run
                                if str(r.get('l2', '')).strip() in ('', 'nan')),
                'n_at_fit_bound': sum(int(r.get('at_bound', 0)) for r in per_run),
                'cross_check_members_vs_multinu': n_archive_checked,
                'pairing': ('schedules share label and Brownian streams at '
                            'fixed (nu, seed): paired through common random '
                            'numbers, aligned by sorted rank, not an exact '
                            'cancellation. Not paired across nu or dt.'),
                'command': 'python studies/study_ordering_pilot.py'},
           'summary': summary,
           'schedule_contrasts': schedule_contrasts,
           'label_contrasts': label_contrasts}
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    json.dump(out, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=2)
    print(f"Wrote summary.csv, summary.json, per_run.csv to {out_dir}")
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=None)
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--arms', type=str, nargs='*', default=None,
                    help="subset of {cond_mean, two_speed_a2, two_speed_a4}; "
                         "default cond_mean (extend only if the control "
                         "comparison justifies it)")
    args = ap.parse_args()
    run_study(S_override=args.seeds, out_base=args.out, arms=args.arms)
