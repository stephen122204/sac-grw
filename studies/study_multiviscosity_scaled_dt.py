"""Scaled-time-step companion to the multi-viscosity sweep: dt proportional
to nu.

Manuscript map: the companion sweep of `sec:gbmc-multinu` with dt/nu held
fixed. Production study; outputs are archived under
output/final_prepublication_tests/gbmc_multiviscosity_scaled_dt/. Same
value-keyed streams and per-cell resume rules as the fixed-step sweep; the
nu = 0.5 cells are shared with it and verified to match on every invocation.

The fixed-step sweep varies nu at dt = 0.0025, so the dimensionless ratio
dt/nu grows as the layer sharpens and viscosity dependence is confounded with
temporal resolution.  This companion holds dt/nu fixed instead:

    dt(nu) = 0.0025 * (nu / 0.5),   nu in {0.5, 0.25, 0.1, 0.05, 0.025},

so every row has dt/nu = 0.005 and the analytic label scale satisfies
D_label(nu)/nu = (dt/(2*nu)) * [a^2 - A^2(1/3 + 2/(3 N^2))] = constant.
Under the local label-variance mechanism the paired RELATIVE excess
(nu_hat_two-speed - nu_hat_control)/nu should therefore stay approximately
flat across nu, which is the parameter-guidance statement the fixed-step
sweep alone cannot isolate.

Design mirrors the fixed-step sweep exactly except for dt: N = 6400, S = 50,
arms {two-speed a=2, a=4, conditional-mean control}, the same nu-scaled
evaluation windows, strict nu-scaled fit bounds, and the same value-keyed
random streams (SeedSequence([base, round(nu*1e6), seed_idx])).  The nu = 0.5
row uses dt = 0.0025 and identical streams, so it must reproduce the archived
fixed-step nu = 0.5 cells bit-for-bit; that is asserted on every invocation.
Rows at different nu are NOT paired (different step counts and draw orders).
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
    'output', 'final_prepublication_tests', 'gbmc_multiviscosity_scaled_dt')
NU_SEQ = [0.5, 0.25, 0.1, 0.05, 0.025]
DT_OVER_NU = 0.005          # dt(nu) = DT_OVER_NU * nu
N_FIXED = 6400
S = 50
ARMS = [('two_speed_a2', 'two_speed', 2.0),
        ('two_speed_a4', 'two_speed', 4.0),
        ('cond_mean', 'conditional_mean', 2.0)]


def dt_for(nu):
    return DT_OVER_NU * nu


def _d_label(a, dt):
    return 0.5 * dt * (a ** 2 - A ** 2 * (1.0 / 3.0 + 2.0 / (3.0 * N_FIXED ** 2)))


def _fingerprint(S_use):
    return {'N': N_FIXED, 'dt_rule': f'dt = {DT_OVER_NU} * nu (dt/nu fixed)',
            'T': T, 'N_OUT': N_OUT, 'A': A, 'XC': XC,
            'S': int(S_use), 'arms': [a[0] for a in ARMS],
            'nu_seq': list(NU_SEQ),
            'window_shock_widths': mv.WINDOW_SHOCK_WIDTHS,
            'fit_bounds': 'nu_lo=nu/50, nu_hi=max(2, 20*nu); at_bound within 2%',
            'seed_label_base': mv.SEED_LABEL_BASE,
            'seed_brownian_base': mv.SEED_BROWNIAN_BASE,
            'seed_design': 'SeedSequence([base, round(nu*1e6), seed_idx])'}


def _run_arm(nu, seed_idx, transport, a_rel, x_out, u_ref, dx):
    x0, m0, u_left = initialize_tanh_shock_particles(N_FIXED, nu, A, XC)
    dt = dt_for(nu)
    n_steps = int(round(T / dt))
    rng_label, rng_brownian = mv._rngs(nu, seed_idx)
    t0 = time.perf_counter()
    run = advance_rbgbmc_particles(
        x0, m0, u_left, nu, a_rel, dt, n_steps, rng_label,
        rng_brownian=rng_brownian,
        conditional_mean_transport=(transport == 'conditional_mean'))
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


def run_study(S_override=None, out_base=None, nu_seq=None):
    S_use = int(S_override) if S_override else S
    fp = _fingerprint(S_use)
    nus = [float(v) for v in nu_seq] if nu_seq else list(NU_SEQ)
    unknown = [nu for nu in nus
               if mv._nu_key(nu) not in {mv._nu_key(v) for v in NU_SEQ}]
    if unknown:
        raise ValueError(
            f"Requested viscosities {unknown} are not in the canonical study "
            f"list {NU_SEQ}.")
    out_dir = out_base or OUT_BASE
    os.makedirs(out_dir, exist_ok=True)

    windows = {nu: mv._window(nu) for nu in NU_SEQ}
    urefs = {nu: mv.u_exact_nu(windows[nu][0], nu) for nu in NU_SEQ}

    manifest_path = os.path.join(out_dir, 'manifest.json')
    done = set()
    per_run = []
    fit_by = {}
    prof = {}

    def _cell_npz(nu_c, arm_c):
        return os.path.join(out_dir,
                            f'cell_{arm_c}_nu{nu_c:g}'.replace('.', 'p') + '.npz')

    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        if manifest.get('config') != fp:
            raise RuntimeError(
                f"Refusing to resume {out_dir}: stored configuration fingerprint "
                f"differs.\n  stored:  {manifest.get('config')}\n  current: {fp}")
        claimed = set(tuple(c) for c in manifest.get('done', []))
        if os.path.exists(os.path.join(out_dir, 'per_run.csv')):
            per_run = list(csv.DictReader(open(os.path.join(out_dir, 'per_run.csv'))))
        for row in per_run:
            fit_by.setdefault((float(row['nu']), row['arm']), {})[
                int(row['seed_idx'])] = float(row['nu_fit'])
        for (nu_c, arm_c) in claimed:
            if not os.path.exists(_cell_npz(nu_c, arm_c)):
                continue
            arr = np.load(_cell_npz(nu_c, arm_c))['profiles']
            if arr.shape != (S_use, N_OUT):
                continue
            if sorted(fit_by.get((nu_c, arm_c), {})) != list(range(S_use)):
                continue
            prof[(nu_c, arm_c)] = arr
            done.add((nu_c, arm_c))
        per_run = [r for r in per_run if (float(r['nu']), r['arm']) in done]
        fit_by = {k: v for k, v in fit_by.items() if k in done}
        print(f"resuming: {len(done)}/{len(claimed)} claimed cells are complete")

    print(f"Scaled-dt multi-viscosity sweep: nu={nus} (dt = {DT_OVER_NU}*nu), "
          f"arms={[a[0] for a in ARMS]}, N={N_FIXED}, M={N_OUT}, S={S_use}")
    for nu in nus:
        x_out, dx = windows[nu]
        u_ref = urefs[nu]
        for arm_id, transport, a_rel in ARMS:
            if (nu, arm_id) in done:
                continue
            profiles, fits, n_bound = [], {}, 0
            for s in range(S_use):
                r = _run_arm(nu, s, transport, a_rel, x_out, u_ref, dx)
                profiles.append(r['u_out'])
                fits[s] = r['nu_fit']
                n_bound += int(r['at_bound'])
                per_run.append({'nu': nu, 'dt': dt_for(nu), 'arm': arm_id,
                                'transport': transport, 'a': a_rel,
                                'seed_idx': s, 'l2': r['l2'],
                                'nu_fit': r['nu_fit'], 'A_fit': r['A_fit'],
                                'at_bound': int(r['at_bound']),
                                'runtime_s': r['runtime_s']})
            prof[(nu, arm_id)] = np.asarray(profiles)
            fit_by[(nu, arm_id)] = fits
            done.add((nu, arm_id))
            with open(os.path.join(out_dir, 'per_run.csv'), 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(per_run[0]))
                w.writeheader(); w.writerows(per_run)
            _savez_deterministic(_cell_npz(nu, arm_id), profiles=prof[(nu, arm_id)])
            json.dump({'config': fp, 'done': sorted(done)},
                      open(manifest_path, 'w'))
            mean_fit = float(np.mean(list(fits.values())))
            print(f"  nu={nu:<7g} dt={dt_for(nu):<9g} {arm_id:<13s} "
                  f"nu_hat={mean_fit:.5f} abs_off={mean_fit - nu:+.5f} "
                  f"rel_off={(mean_fit - nu) / nu:+.3%} at_bound={n_bound}/{S_use}")

    # nu=0.5 shares dt and streams with the fixed-step sweep: archive-vs-archive
    # equality is asserted on every invocation.
    n_checked = 0
    mn_path = os.path.join(mv.OUT_BASE, 'per_run.csv')
    if os.path.exists(mn_path):
        ref = {}
        with open(mn_path) as handle:
            for row in csv.DictReader(handle):
                ref[(float(row['nu']), row['arm'], int(row['seed_idx']))] = \
                    float(row['nu_fit'])
        for (nu_c, arm_c), fits in fit_by.items():
            if abs(nu_c - 0.5) > 1e-15:
                continue
            for s, value in fits.items():
                if (0.5, arm_c, s) in ref:
                    if value != ref[(0.5, arm_c, s)]:
                        raise RuntimeError(
                            f"nu=0.5 cross-check FAILED (arm={arm_c}, seed={s}): "
                            f"scaled-dt {value!r} vs fixed-step "
                            f"{ref[(0.5, arm_c, s)]!r}")
                    n_checked += 1
        if n_checked:
            print(f"  Archive cross-check PASS: {n_checked} nu=0.5 members "
                  "match the fixed-step sweep exactly.")

    # ---------------------------------------------------------------- analysis
    summary, paired = [], []
    completed_nus = [nu for nu in NU_SEQ
                     if any((nu, arm_id) in prof for arm_id, _, _ in ARMS)]
    rng = np.random.default_rng(20260825)
    for nu in completed_nus:
        dt = dt_for(nu)
        for arm_id, transport, a_rel in ARMS:
            if (nu, arm_id) not in prof:
                continue
            fits = np.array([fit_by[(nu, arm_id)][s]
                             for s in sorted(fit_by[(nu, arm_id)])])
            off = fits - nu
            off_boot = np.array([off[rng.integers(0, len(off), len(off))].mean()
                                 for _ in range(5000)])
            ci_lo, ci_hi = np.percentile(off_boot, [2.5, 97.5])
            d_lab = _d_label(a_rel, dt) if transport == 'two_speed' else 0.0
            summary.append({
                'nu': nu, 'dt': dt, 'arm': arm_id, 'a': a_rel, 'S': S_use,
                'nu_mean': float(fits.mean()),
                'nu_offset_abs': float(off.mean()),
                'nu_offset_abs_ci_lo': float(ci_lo),
                'nu_offset_abs_ci_hi': float(ci_hi),
                'nu_offset_rel': float(off.mean() / nu),
                'D_label': float(d_lab),
                'D_label_over_nu': float(d_lab / nu),
                'nu_std': float(fits.std())})
        def dfit(arm_a, arm_b):
            a_map, b_map = fit_by.get((nu, arm_a)), fit_by.get((nu, arm_b))
            if not a_map or not b_map:
                return None
            seeds = sorted(set(a_map) & set(b_map))
            d = np.array([a_map[s] - b_map[s] for s in seeds])
            iv = _paired_ci(d, rng)
            a_rel = {'two_speed_a2': 2.0, 'two_speed_a4': 4.0}.get(arm_a)
            extra = {}
            if a_rel is not None and arm_b == 'cond_mean':
                d_lab = _d_label(a_rel, dt)
                extra = {'D_label': float(d_lab),
                         'excess_over_D_label': float(iv['mean'] / d_lab),
                         'rel_excess': float(iv['mean'] / nu)}
            return {'nu': nu, 'dt': dt, 'contrast': f'{arm_a} - {arm_b}',
                    'n_pairs': len(seeds), **iv, **extra}
        for pair in [('two_speed_a2', 'cond_mean'),
                     ('two_speed_a4', 'cond_mean'),
                     ('two_speed_a4', 'two_speed_a2')]:
            iv = dfit(*pair)
            if iv:
                paired.append(iv)

    if summary:
        with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader()
            w.writerows(summary)
    out = {'metadata': {
                'purpose': ('scaled-time-step companion: dt/nu held fixed at '
                            f'{DT_OVER_NU} while nu varies'),
                'nu_seq': completed_nus, 'requested_nu_seq': nus,
                'dt_rule': f'dt = {DT_OVER_NU} * nu',
                'arms': [a[0] for a in ARMS],
                'N': N_FIXED, 'T': T, 'M': N_OUT, 'S': S_use, 'A': A,
                'n_failed': sum(1 for r in per_run
                                if str(r.get('l2', '')).strip() in ('', 'nan')),
                'n_at_fit_bound': sum(int(r.get('at_bound', 0)) for r in per_run),
                'cross_check_nu0p5_members_vs_fixed_step': n_checked,
                'pairing': ('paired only within a viscosity (shared value-keyed '
                            'streams); rows at different nu have different step '
                            'counts and are not paired'),
                'command': 'python studies/study_multiviscosity_scaled_dt.py'},
           'summary': summary, 'paired_nu_contrasts': paired}
    json.dump(out, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=2)
    print(f"Wrote summary.csv, summary.json, per_run.csv to {out_dir}")
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=None)
    ap.add_argument('--nu', type=float, nargs='*', default=None)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    run_study(S_override=args.seeds, out_base=args.out, nu_seq=args.nu)
