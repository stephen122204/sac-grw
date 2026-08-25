"""Conditional-mean transport control (ablation) for Paper 2.

Purpose.  The a-by-dt matrix shows the recovered shock width increases with the
relaxation speed a and decreases under time-step refinement, consistent with the
conditional label-variance scale D_label. That is an association. This ablation
tests attribution: it replaces the sampled two-speed label V_i in {-a,+a} with
the exact conditional mean V_i = u_i (Burgers f'(u)=u), which removes label-
sampling variance while keeping the same Brownian diffusion. Archived outcome
(see summary.json and the manuscript's ablation section): the control removes
most of the a-dependent broadening, and all six paired two-speed-minus-control
intervals resolve away from zero, so the manuscript reports a paired
attribution to sampled-label transport in the tested setting.

Design (separate experiment, split RNG streams).
  N = 6400, M = 400, S = 50
  dt in {0.01, 0.0025, 0.000625}
  arms at each dt: two-speed a=2, two-speed a=4, one conditional-mean control.
At a fixed dt every arm and seed uses the SAME Brownian generator, so the arms
receive identical Brownian increments and the comparison is paired. Label
uniforms come from a SEPARATE generator (unused by the control). The production
single-stream path is untouched. This control is an internal diagnostic; it is
NOT Roberts' method or a competing solver, and comparisons are not made against
the single-stream a-by-dt matrix as though paired.
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
    A, NU, T, L, XC, N_OUT, _fit_tanh, u_exact_fn, _savez_deterministic,
)

OUT_BASE = os.path.join(
    'output', 'final_prepublication_tests', 'gbmc_conditional_mean_ablation')
DT_SEQ = [0.01, 0.0025, 0.000625]
N_FIXED = 6400
S = 50
SEED_LABEL_BASE = 810000
SEED_BROWNIAN_BASE = 820000
# (arm_id, transport, a): the control's a only sets the subcharacteristic
# margin; its transport velocity is u_i regardless.
ARMS = [('two_speed_a2', 'two_speed', 2.0),
        ('two_speed_a4', 'two_speed', 4.0),
        ('cond_mean', 'conditional_mean', 2.0)]


def _versions():
    import platform, scipy
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__}


def _fingerprint(S_use):
    """Configuration fingerprint stored in the manifest. A resume is refused
    when it differs, so an incompatible S / dt design / seed scheme cannot
    silently reuse cells."""
    return {'N': N_FIXED, 'M': N_OUT, 'S': int(S_use), 'nu': NU, 'T': T,
            'A': A, 'XC': XC, 'L': L, 'dt_seq': list(DT_SEQ),
            'arms': [a[0] for a in ARMS],
            'seed_label_base': SEED_LABEL_BASE,
            'seed_brownian_base': SEED_BROWNIAN_BASE,
            'fit_bounds': 'nu_lo=0.05, nu_hi=2.0 (production defaults)',
            'seed_design': 'default_rng(base + seed_idx), shared across arms'}


def _run_arm(dt, seed_idx, transport, a_rel, x_out, u_ref, dx):
    x0, m0, u_left = initialize_tanh_shock_particles(N_FIXED, NU, A, XC)
    n_steps = int(round(T / dt))
    rng_label = np.random.default_rng(SEED_LABEL_BASE + seed_idx)
    rng_brownian = np.random.default_rng(SEED_BROWNIAN_BASE + seed_idx)
    t0 = time.perf_counter()
    run = advance_rbgbmc_particles(
        x0, m0, u_left, NU, a_rel, dt, n_steps, rng_label,
        rng_brownian=rng_brownian,
        conditional_mean_transport=(transport == 'conditional_mean'))
    u_out = reconstruct_cumulative_field(run['x'], run['m'], u_left, x_out)
    l2 = float(np.sqrt(np.sum((u_out - u_ref) ** 2) * dx))
    xc_fit, nu_fit, A_fit = _fit_tanh(x_out, u_out, A, XC, NU, strict=True)
    return {'u_out': u_out, 'l2': l2, 'nu_fit': nu_fit,
            'runtime_s': time.perf_counter() - t0}


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


def run_study(S_override=None, out_base=None):
    S_use = int(S_override) if S_override else S
    out_dir = out_base or OUT_BASE
    os.makedirs(out_dir, exist_ok=True)
    x_out = np.linspace(0.0, L, N_OUT)
    u_ref = u_exact_fn(x_out)
    dx = float(x_out[1] - x_out[0])

    # resumable by cell: a cell is (dt, arm_id). A cell is only treated as done
    # if BOTH its scalar rows and its profiles were checkpointed, so a resumed
    # run can still recompute ensemble bias/spread/total.
    manifest_path = os.path.join(out_dir, 'manifest.json')
    done = set()
    per_run = []
    nu_by = {}   # (dt, arm_id) -> {seed_idx: nu_fit}
    prof = {}    # (dt, arm_id) -> (S, N_OUT) profiles

    def _cell_npz(dt_c, arm_c):
        return os.path.join(out_dir,
                            f'cell_{arm_c}_{dt_c:g}'.replace('.', 'p') + '.npz')

    fp = _fingerprint(S_use)
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
            nu_by.setdefault((float(row['dt']), row['arm']), {})[
                int(row['seed_idx'])] = float(row['nu_fit'])
        for (dt_c, arm_c) in claimed:
            if not os.path.exists(_cell_npz(dt_c, arm_c)):
                continue
            arr = np.load(_cell_npz(dt_c, arm_c))['profiles']
            if arr.shape != (S_use, N_OUT):   # reject stale/incompatible shapes
                continue
            seeds_seen = sorted(nu_by.get((dt_c, arm_c), {}))
            if seeds_seen != list(range(S_use)):   # reject incomplete scalar rows
                continue
            prof[(dt_c, arm_c)] = arr
            done.add((dt_c, arm_c))
        # drop scalar rows and nu for cells whose profiles are missing (rerun)
        per_run = [r for r in per_run if (float(r['dt']), r['arm']) in done]
        nu_by = {k: v for k, v in nu_by.items() if k in done}
        print(f"resuming: {len(done)}/{len(claimed)} claimed cells are complete")

    print(f"Conditional-mean ablation: dt={DT_SEQ}, arms={[a[0] for a in ARMS]}, "
          f"N={N_FIXED}, M={N_OUT}, S={S_use}")
    for dt in DT_SEQ:
        for arm_id, transport, a_rel in ARMS:
            if (dt, arm_id) in done:
                continue
            profiles, nus = [], {}
            for s in range(S_use):
                r = _run_arm(dt, s, transport, a_rel, x_out, u_ref, dx)
                profiles.append(r['u_out'])
                nus[s] = r['nu_fit']
                per_run.append({'dt': dt, 'arm': arm_id, 'transport': transport,
                                'a': a_rel, 'seed_idx': s, 'l2': r['l2'],
                                'nu_fit': r['nu_fit'], 'runtime_s': r['runtime_s']})
            prof[(dt, arm_id)] = np.asarray(profiles)
            nu_by[(dt, arm_id)] = nus
            done.add((dt, arm_id))
            # checkpoint after each cell: scalar rows AND profiles
            with open(os.path.join(out_dir, 'per_run.csv'), 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(per_run[0]))
                w.writeheader(); w.writerows(per_run)
            _savez_deterministic(_cell_npz(dt, arm_id),
                                 profiles=prof[(dt, arm_id)])
            json.dump({'config': fp, 'done': sorted(done)},
                      open(manifest_path, 'w'))
            em = _ensemble(profiles, u_ref, dx)
            print(f"  dt={dt:<9g} {arm_id:<13s} bias={em['E_bias']:.5f} "
                  f"spread={em['E_spread']:.5f} nu={np.mean(list(nus.values())):.5f}")

    # Profiles for every cell are available: freshly-run cells stay in memory,
    # resumed cells were reloaded from their per-cell npz above.
    summary, paired = [], []
    rng = np.random.default_rng(20260821)
    for dt in DT_SEQ:
        for arm_id, transport, a_rel in ARMS:
            if (dt, arm_id) not in prof:
                continue
            em = _ensemble(prof[(dt, arm_id)], u_ref, dx)
            nus = np.array(list(nu_by[(dt, arm_id)].values()))
            off = nus - NU
            off_boot = np.array([off[rng.integers(0, len(off), len(off))].mean()
                                 for _ in range(5000)])
            ci_lo, ci_hi = np.percentile(off_boot, [2.5, 97.5])
            identity = abs(em['E_total'] ** 2 - em['E_bias'] ** 2
                           - em['E_spread'] ** 2)
            summary.append({'dt': dt, 'arm': arm_id, 'a': a_rel, 'S': S_use,
                            **em, 'identity_error': identity,
                            'nu_mean': float(nus.mean()),
                            'nu_offset': float(off.mean()),
                            'nu_offset_ci_lo': float(ci_lo),
                            'nu_offset_ci_hi': float(ci_hi),
                            'nu_offset_ci_includes_zero': bool(ci_lo <= 0 <= ci_hi),
                            'nu_std': float(nus.std())})
        # paired nu_fit differences at this dt (shared Brownian => paired)
        def dnu(arm_a, arm_b):
            a_map, b_map = nu_by.get((dt, arm_a)), nu_by.get((dt, arm_b))
            if not a_map or not b_map:
                return None
            seeds = sorted(set(a_map) & set(b_map))
            d = np.array([a_map[s] - b_map[s] for s in seeds])
            return {'dt': dt, 'contrast': f'{arm_a} - {arm_b}',
                    'n_pairs': len(seeds), **_paired_ci(d, rng)}
        for pair in [('two_speed_a2', 'cond_mean'),
                     ('two_speed_a4', 'cond_mean'),
                     ('two_speed_a4', 'two_speed_a2')]:
            iv = dnu(*pair)
            if iv:
                paired.append(iv)

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader()
        w.writerows(summary)
    # decision metric: how much of each two-speed arm's width offset the control removes
    decision = {}
    for dt in DT_SEQ:
        row = {s['arm']: s for s in summary if s['dt'] == dt}
        if {'two_speed_a2', 'two_speed_a4', 'cond_mean'} <= set(row):
            off_c = row['cond_mean']['nu_offset']
            decision[f'dt={dt:g}'] = {
                'cond_mean_nu_offset': off_c,
                'two_speed_a2_nu_offset': row['two_speed_a2']['nu_offset'],
                'two_speed_a4_nu_offset': row['two_speed_a4']['nu_offset'],
                'frac_a2_offset_removed_by_control':
                    1.0 - off_c / row['two_speed_a2']['nu_offset'],
                'frac_a4_offset_removed_by_control':
                    1.0 - off_c / row['two_speed_a4']['nu_offset'],
            }
    n_failed = sum(1 for r in per_run
                   if str(r.get('l2', '')).strip() in ('', 'nan'))
    out = {'metadata': {'purpose': 'conditional-mean transport control (ablation)',
                        'dt_seq': DT_SEQ, 'arms': [a[0] for a in ARMS],
                        'N': N_FIXED, 'M': N_OUT, 'S': S_use, 'n_failed': n_failed,
                        'package_versions': _versions(),
                        'seed_formulas': (f'rng_label = default_rng({SEED_LABEL_BASE} '
                                          f'+ seed_idx); rng_brownian = '
                                          f'default_rng({SEED_BROWNIAN_BASE} + '
                                          'seed_idx); shared across arms at each dt'),
                        'pairing': ('split streams aligned by sorted particle rank: '
                                    'at fixed dt all arms share the Brownian '
                                    'generator (paired); label uniforms are a '
                                    'separate stream, unused by the control. Not '
                                    'paired across dt, and once arms diverge the '
                                    'shared draws align by rank, not by persistent '
                                    'particle identity.'),
                        'spread_note': ('stochastic spread stays at a similar scale '
                                        '(~0.015-0.0175) across arms; the coarse '
                                        'a=4 spread is modestly larger. It is not '
                                        'exactly flat.'),
                        'label': ('internal conditional-mean transport control; '
                                  'not Roberts method or a competing solver'),
                        'command': 'python reproduce.py ablation'},
           'summary': summary, 'paired_nu_contrasts': paired,
           'decision_metric': decision}
    json.dump(out, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=2)
    if prof:
        _savez_deterministic(os.path.join(out_dir, 'profiles.npz'),
                             x=x_out, u_exact=u_ref,
                             **{f'{a}_{d:g}'.replace('.', 'p'): prof[(d, a)]
                                for (d, a) in prof})
    print("decision metric (fraction of two-speed width offset removed by control):")
    for k, v in decision.items():
        print(f"  {k}: a2 removed {v['frac_a2_offset_removed_by_control']*100:.0f}%, "
              f"a4 removed {v['frac_a4_offset_removed_by_control']*100:.0f}% "
              f"(control offset {v['cond_mean_nu_offset']:+.5f})")
    print(f"Wrote summary.csv, summary.json, per_run.csv to {out_dir}")
    return summary


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=None)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    run_study(S_override=args.seeds, out_base=args.out)
