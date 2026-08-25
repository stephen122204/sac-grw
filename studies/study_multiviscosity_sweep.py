"""Multi-viscosity sweep for Paper 2 (RB-GBMC).

Manuscript map: fixed-time-step viscosity sweep of `sec:gbmc-multinu`.
Production study; outputs are archived under
output/final_prepublication_tests/gbmc_multiviscosity_sweep/. Resumable by
cell with a validated configuration fingerprint; seeds are keyed by the
viscosity value, so subsets and reorderings reproduce identical cells, and
arms share the Brownian stream only within a viscosity.

Purpose.  The stationary-shock studies fix nu = 0.5.  The analytic label scale
D_label = (dt/2)[a^2 - A^2(1/3 + 2/(3N^2))] is INDEPENDENT of nu, so the theory
predicts that the absolute recovered-viscosity offset (nu_hat - nu) stays
roughly flat across nu (tracking D_label), while the RELATIVE offset
(nu_hat - nu)/nu and the ratio D_label/nu grow as nu falls (sharper layer).
This sweep tests that prediction, and whether the conditional-mean control
removes the a-dependent excess at every viscosity, not only at nu = 0.5.

Archived outcome (see summary.json and the manuscript's multi-viscosity
section): the raw absolute offset is NOT flat; the paired
two-speed-minus-control excess rises toward the nu-independent D_label as nu
falls, and the relative quantities grow approximately like D_label/nu. The
'prediction' field stored in summary.json records the preregistered
expectation, not the outcome.

Preregistered safeguards (all implemented here):
  * resolve narrower shocks: the evaluation window is centered on the shock and
    scaled to the layer, so points across the transition do not collapse at low
    nu.  Particles still evolve on the whole line; only the reconstruction
    points narrow.
  * lower and TEST the nu_hat fit bound: the tanh fit uses nu_lo = nu/50 (far
    below the smallest prescribed nu), and every run flags whether nu_hat sits
    at a bound.
  * conditional-mean control at EVERY viscosity, paired within that viscosity.
  * report absolute (nu_hat - nu), relative (nu_hat - nu)/nu, and D_label/nu.
  * pairing only within a viscosity (shared Brownian stream per (nu, seed));
    never paired across nu.
  * resumable by cell, with archived per-run results.

This control is an internal diagnostic; it is NOT Roberts' method or a competing
solver.
"""

import csv
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
    A, T, L, XC, N_OUT, _fit_tanh, _savez_deterministic,
)

OUT_BASE = os.path.join(
    'output', 'final_prepublication_tests', 'gbmc_multiviscosity_sweep')
NU_SEQ = [0.5, 0.25, 0.1, 0.05, 0.025]
DT = 0.0025
N_FIXED = 6400
S = 50
WINDOW_SHOCK_WIDTHS = 12.0   # evaluation half-window, in shock-widths (2*nu)
SEED_LABEL_BASE = 910000
SEED_BROWNIAN_BASE = 920000
# (arm_id, transport, a): the control's a only sets the subcharacteristic
# margin; its transport velocity is u_i regardless.
ARMS = [('two_speed_a2', 'two_speed', 2.0),
        ('two_speed_a4', 'two_speed', 4.0),
        ('cond_mean', 'conditional_mean', 2.0)]


def _versions():
    import platform, scipy
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__}


def u_exact_nu(x, nu):
    return -A * np.tanh(A * (x - XC) / (2.0 * nu))


def _window(nu):
    """Evaluation window centered on the shock and scaled to the layer, clipped
    to the domain.  Particles evolve on the whole line; only the reconstruction
    points narrow, so points across the transition stay adequate at small nu."""
    half = min(L / 2.0, WINDOW_SHOCK_WIDTHS * 2.0 * nu)
    x_out = np.linspace(XC - half, XC + half, N_OUT)
    return x_out, float(x_out[1] - x_out[0])


def _d_label(a):
    """Analytic equal-mass stationary D_label (independent of nu)."""
    return 0.5 * DT * (a ** 2 - A ** 2 * (1.0 / 3.0 + 2.0 / (3.0 * N_FIXED ** 2)))


def _fit_bounds(nu):
    return nu / 50.0, max(2.0, 20.0 * nu)


def _nu_key(nu):
    """Integer key from the viscosity VALUE (not its position in the requested
    list), so a cell's realizations are invariant to nu-list order and subset
    selection."""
    return int(round(nu * 1_000_000))


def _rngs(nu, seed_idx):
    """Label and Brownian generators for one (nu, seed) cell, keyed by the
    viscosity value and seed index only (not the arm).  All arms within a
    viscosity therefore share the Brownian stream (paired within nu), and the
    streams are reproducible and independent across (nu, seed)."""
    nk = _nu_key(nu)
    rng_label = np.random.default_rng(
        np.random.SeedSequence([SEED_LABEL_BASE, nk, seed_idx]))
    rng_brownian = np.random.default_rng(
        np.random.SeedSequence([SEED_BROWNIAN_BASE, nk, seed_idx]))
    return rng_label, rng_brownian


def _fingerprint(S_use):
    """Configuration fingerprint stored in the manifest.  A resume is refused
    when it differs, so an incompatible S / N / dt / window / viscosity-list /
    fit-bound / seed design cannot silently reuse cells.  The fingerprint pins
    the CANONICAL viscosity list; a --nu invocation may select any subset of
    it (cells are keyed by value), but a viscosity outside the canonical list
    is a design change and is refused before any run."""
    return {'N': N_FIXED, 'dt': DT, 'T': T, 'N_OUT': N_OUT, 'A': A, 'XC': XC,
            'L': L, 'S': int(S_use), 'arms': [a[0] for a in ARMS],
            'nu_seq': list(NU_SEQ),
            'window_shock_widths': WINDOW_SHOCK_WIDTHS,
            'fit_bounds': 'nu_lo=nu/50, nu_hi=max(2, 20*nu); at_bound within 2%',
            'seed_label_base': SEED_LABEL_BASE,
            'seed_brownian_base': SEED_BROWNIAN_BASE,
            'seed_design': 'SeedSequence([base, round(nu*1e6), seed_idx])'}


def _run_arm(nu, seed_idx, transport, a_rel, x_out, u_ref, dx):
    x0, m0, u_left = initialize_tanh_shock_particles(N_FIXED, nu, A, XC)
    n_steps = int(round(T / DT))
    rng_label, rng_brownian = _rngs(nu, seed_idx)
    t0 = time.perf_counter()
    run = advance_rbgbmc_particles(
        x0, m0, u_left, nu, a_rel, DT, n_steps, rng_label,
        rng_brownian=rng_brownian,
        conditional_mean_transport=(transport == 'conditional_mean'))
    u_out = reconstruct_cumulative_field(run['x'], run['m'], u_left, x_out)
    l2 = float(np.sqrt(np.sum((u_out - u_ref) ** 2) * dx))
    nu_lo, nu_hi = _fit_bounds(nu)
    xc_fit, nu_fit, A_fit = _fit_tanh(x_out, u_out, A, XC, nu,
                                      nu_lo=nu_lo, nu_hi=nu_hi, strict=True)
    at_bound = bool(nu_fit <= 1.02 * nu_lo or nu_fit >= 0.98 * nu_hi)
    return {'u_out': u_out, 'l2': l2, 'nu_fit': nu_fit, 'A_fit': A_fit,
            'at_bound': at_bound, 'runtime_s': time.perf_counter() - t0}


def _ensemble(profiles, u_ref, dx):
    u = np.asarray(profiles)
    um = u.mean(axis=0)
    return {
        'E_bias': float(np.sqrt(np.sum((um - u_ref) ** 2 * dx))),
        'E_spread': float(np.sqrt(np.mean(np.sum((u - um[None, :]) ** 2 * dx, axis=1)))),
        'E_total': float(np.sqrt(np.mean(np.sum((u - u_ref[None, :]) ** 2 * dx, axis=1)))),
    }


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
    unknown = [nu for nu in nus if _nu_key(nu) not in {_nu_key(v) for v in NU_SEQ}]
    if unknown:
        raise ValueError(
            f"Requested viscosities {unknown} are not in the canonical study "
            f"list {NU_SEQ}. Changing the viscosity design is a new study: "
            "edit NU_SEQ (and rerun everything) rather than passing ad-hoc "
            "values against the pinned archive.")
    out_dir = out_base or OUT_BASE
    os.makedirs(out_dir, exist_ok=True)

    # Windows over the CANONICAL list (requested and completed cells are both
    # subsets of it), so the closing summary can always cover every completed
    # cell regardless of which subset this invocation ran.
    windows = {nu: _window(nu) for nu in NU_SEQ}
    urefs = {nu: u_exact_nu(windows[nu][0], nu) for nu in NU_SEQ}

    # resumable by cell: a cell is (nu, arm_id), done only when BOTH its scalar
    # rows and its profiles were checkpointed.
    manifest_path = os.path.join(out_dir, 'manifest.json')
    done = set()
    per_run = []
    fit_by = {}   # (nu, arm_id) -> {seed_idx: nu_fit}
    prof = {}     # (nu, arm_id) -> (S, N_OUT) profiles

    def _cell_npz(nu_c, arm_c):
        return os.path.join(out_dir,
                            f'cell_{arm_c}_nu{nu_c:g}'.replace('.', 'p') + '.npz')

    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        if manifest.get('config') != fp:
            raise RuntimeError(
                f"Refusing to resume {out_dir}: stored configuration fingerprint "
                f"differs from the current run. Use a fresh --out or remove the "
                f"directory.\n  stored:  {manifest.get('config')}\n  current: {fp}")
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
            if arr.shape != (S_use, N_OUT):   # reject stale/incompatible shapes
                continue
            seeds_seen = sorted(fit_by.get((nu_c, arm_c), {}))
            if seeds_seen != list(range(S_use)):   # reject incomplete/duplicated rows
                continue
            prof[(nu_c, arm_c)] = arr
            done.add((nu_c, arm_c))
        per_run = [r for r in per_run if (float(r['nu']), r['arm']) in done]
        fit_by = {k: v for k, v in fit_by.items() if k in done}
        print(f"resuming: {len(done)}/{len(claimed)} claimed cells are complete")

    print(f"Multi-viscosity sweep: nu={nus}, arms={[a[0] for a in ARMS]}, "
          f"N={N_FIXED}, dt={DT}, M={N_OUT}, S={S_use}")
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
                per_run.append({'nu': nu, 'arm': arm_id, 'transport': transport,
                                'a': a_rel, 'seed_idx': s, 'l2': r['l2'],
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
            json.dump({'config': fp, 'done': sorted(done)}, open(manifest_path, 'w'))
            mean_fit = float(np.mean(list(fits.values())))
            print(f"  nu={nu:<7g} {arm_id:<13s} nu_hat={mean_fit:.5f} "
                  f"abs_off={mean_fit - nu:+.5f} rel_off={(mean_fit - nu) / nu:+.3f} "
                  f"at_bound={n_bound}/{S_use}")

    # ---------------------------------------------------------------- analysis
    # The closing summary always covers EVERY completed cell in canonical
    # order, so a subset --nu invocation can never overwrite a complete
    # summary with only the requested rows, and the bootstrap random stream is
    # consumed in the same order for every invocation pattern.
    summary, paired = [], []
    completed_nus = [nu for nu in NU_SEQ
                     if any((nu, arm_id) in prof for arm_id, _, _ in ARMS)]
    rng = np.random.default_rng(20260821)
    for nu in completed_nus:
        x_out, dx = windows[nu]
        u_ref = urefs[nu]
        for arm_id, transport, a_rel in ARMS:
            if (nu, arm_id) not in prof:
                continue
            em = _ensemble(prof[(nu, arm_id)], u_ref, dx)
            fits = np.array(list(fit_by[(nu, arm_id)].values()))
            off = fits - nu
            off_boot = np.array([off[rng.integers(0, len(off), len(off))].mean()
                                 for _ in range(5000)])
            ci_lo, ci_hi = np.percentile(off_boot, [2.5, 97.5])
            d_lab = _d_label(a_rel) if transport == 'two_speed' else 0.0
            summary.append({
                'nu': nu, 'arm': arm_id, 'a': a_rel, 'S': S_use, **em,
                'nu_mean': float(fits.mean()),
                'nu_offset_abs': float(off.mean()),
                'nu_offset_abs_ci_lo': float(ci_lo),
                'nu_offset_abs_ci_hi': float(ci_hi),
                'nu_offset_rel': float(off.mean() / nu),
                'nu_offset_rel_ci_lo': float(ci_lo / nu),
                'nu_offset_rel_ci_hi': float(ci_hi / nu),
                'D_label': float(d_lab),
                'D_label_over_nu': float(d_lab / nu),
                'nu_std': float(fits.std())})
        # paired nu_fit differences WITHIN this viscosity only (shared Brownian)
        def dfit(arm_a, arm_b):
            a_map, b_map = fit_by.get((nu, arm_a)), fit_by.get((nu, arm_b))
            if not a_map or not b_map:
                return None
            seeds = sorted(set(a_map) & set(b_map))
            d = np.array([a_map[s] - b_map[s] for s in seeds])
            return {'nu': nu, 'contrast': f'{arm_a} - {arm_b}',
                    'n_pairs': len(seeds), **_paired_ci(d, rng)}
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
    n_failed = sum(1 for r in per_run
                   if str(r.get('l2', '')).strip() in ('', 'nan'))
    n_at_bound = sum(int(r.get('at_bound', 0)) for r in per_run)
    out = {'metadata': {
                'purpose': 'multi-viscosity sweep of the recovered-viscosity offset',
                'nu_seq': completed_nus, 'requested_nu_seq': nus,
                'arms': [a[0] for a in ARMS],
                'N': N_FIXED, 'dt': DT, 'T': T, 'M': N_OUT, 'S': S_use, 'A': A,
                'window_half_widths_in_shock_widths': WINDOW_SHOCK_WIDTHS,
                'n_failed': n_failed, 'n_at_fit_bound': n_at_bound,
                'package_versions': _versions(),
                'prediction': ('D_label is nu-independent, so nu_offset_abs should be '
                               'roughly flat in nu while nu_offset_rel and D_label/nu '
                               'grow as nu falls'),
                'seed_formulas': ('per (nu, seed_idx): rng_label = default_rng('
                                  f'SeedSequence([{SEED_LABEL_BASE}, round(nu*1e6), '
                                  'seed_idx])); rng_brownian = default_rng(SeedSequence'
                                  f'([{SEED_BROWNIAN_BASE}, round(nu*1e6), seed_idx])); '
                                  'keyed by the viscosity VALUE so a cell is invariant '
                                  'to nu-list order and subset selection; shared across '
                                  'arms within a viscosity, distinct per nu'),
                'pairing': ('paired only within a viscosity: at fixed nu all arms share '
                            'the Brownian generator; label uniforms are a separate '
                            'stream unused by the control. Not paired across nu, and '
                            'once arms diverge the shared draws align by sorted rank, '
                            'not by persistent particle identity.'),
                'fit_bound': ('tanh fit uses nu_lo = nu/50, nu_hi = max(2, 20*nu); '
                              'at_bound flags runs whose nu_hat sits within 2% of a '
                              'bound'),
                'window': (f'evaluation window = XC +/- min(L/2, '
                           f'{WINDOW_SHOCK_WIDTHS:g}*2*nu); particles evolve on the '
                           'whole line, only the reconstruction points narrow'),
                'label': ('internal conditional-mean transport control; not Roberts '
                          'method or a competing solver'),
                'command': 'python reproduce.py multinu'},
           'summary': summary, 'paired_nu_contrasts': paired}
    json.dump(out, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=2)
    print(f"n_failed={n_failed}, n_at_fit_bound={n_at_bound}")
    print(f"Wrote summary.csv, summary.json, per_run.csv to {out_dir}")
    return summary


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=None)
    ap.add_argument('--nu', type=float, nargs='*', default=None)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    run_study(S_override=args.seeds, out_base=args.out, nu_seq=args.nu)
