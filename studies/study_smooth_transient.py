"""Smooth nonstationary transient study for Paper 2 (RB-GBMC).

Manuscript map: `sec:gbmc-transient` (initial condition `eq:transient-ic`).
Production study; outputs are archived under
output/final_prepublication_tests/gbmc_smooth_transient/, including the
reference and its documented tolerance (reference.npz). The reference is a
high-accuracy evaluation of the exact Cole-Hopf formula; no tanh-profile fit
is used anywhere in this study.

Purpose.  Every stationary-shock diagnostic above rests on the equal-mass tanh
representation, whose sorted reconstruction is time-independent.  This study
tests whether the relaxation-speed- and time-step-dependent label mechanism
persists on a genuinely time-dependent smooth solution with signed masses of
BOTH signs, where the reconstruction evolves and D_label becomes a trajectory
quantity.

Problem.  Whole-line viscous Burgers with the Gaussian hump

    u0(x) = B exp(-(x - x0)^2 / (2 sigma^2)),   B = 1, x0 = 2, sigma = 0.5,

at nu = 0.1 up to T = 1.  The inviscid steepening time is sigma*sqrt(e)/B
(about 0.82), so by T = 1 the right flank has formed a viscous front and the
state is far from the initial condition.  u0' changes sign: the particle
initialization carries equal-magnitude masses +-TV/N with TV = 2B, placed at
the quantiles of |u0'|/TV, and u(-inf) = 0.  max|u| <= B = 1 for the
reconstruction of this representation, so a = 2 and a = 4 keep a strict
subcharacteristic margin.

Reference.  The exact whole-line solution via the Cole--Hopf representation,

    u(x,t) = [ int (x-y)/t exp(-G/(2 nu)) dy ] / [ int exp(-G/(2 nu)) dy ],
    G(y; x, t) = int_0^y u0(s) ds + (x - y)^2 / (2 t),

with the inner integral analytic (an erf) and the outer integrals evaluated by
adaptive quadrature after subtracting min G for stability.  The reference is
INDEPENDENT of RB-GBMC.  Its accuracy is documented in the archive: quadrature
self-consistency (epsrel 1e-10 versus 1e-8) and agreement with an independent
Crank--Nicolson finite-difference solve; the stored reference tolerance is the
larger of the two.

Design.  dt in {0.005, 0.0025, 0.00125}; arms {two-speed a=2, two-speed a=4,
conditional-mean control}; N = 6400 particles; S = 50 seeds; 400 reconstruction
points on [0, 6].  Within one dt all arms share the Brownian stream per seed
(value-keyed SeedSequence), so arm comparisons are paired through common
random numbers; comparisons across dt are not paired.  There is no profile
fit: the primary mechanism quantity is the paired seed-level difference of the
L^2 errors, two-speed minus control, and the ensemble decomposition is
reported per arm.  Resumable by cell with a validated fingerprint.
"""

import contextlib
import csv
import io
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxation_gbmc import (
    advance_rbgbmc_particles,
    reconstruct_cumulative_field,
)
from studies.study_gbmc_production_n_refinement import _savez_deterministic

OUT_BASE = os.path.join(
    'output', 'final_prepublication_tests', 'gbmc_smooth_transient')
B_AMP = 1.0
X0 = 2.0
SIGMA = 0.5
NU = 0.1
T_FINAL = 1.0
DT_SEQ = [0.005, 0.0025, 0.00125]
N_FIXED = 6400
S = 50
N_OUT = 400
X_LO, X_HI = 0.0, 6.0
SEED_LABEL_BASE = 930000
SEED_BROWNIAN_BASE = 940000
ARMS = [('two_speed_a2', 'two_speed', 2.0),
        ('two_speed_a4', 'two_speed', 4.0),
        ('cond_mean', 'conditional_mean', 2.0)]


def u0(x):
    return B_AMP * np.exp(-(x - X0) ** 2 / (2.0 * SIGMA ** 2))


def u0_antiderivative(y):
    """int_0^y u0(s) ds, analytic."""
    from scipy.special import erf
    c = B_AMP * SIGMA * math.sqrt(math.pi / 2.0)
    return c * (erf((y - X0) / (SIGMA * math.sqrt(2.0)))
                - erf((0.0 - X0) / (SIGMA * math.sqrt(2.0))))


def initialize_gaussian_gradient_particles(N):
    """Equal-|mass| quantile representation of w = u0' (signs from u0').

    |u0'| / TV is a probability density with TV = 2B; particle i sits at the
    (i - 1/2)/N quantile of that density and carries mass sign(u0'(X_i))*TV/N.
    The reconstruction from the left state u(-inf) = 0 then reproduces u0 in
    the particle limit.
    """
    y = np.linspace(X0 - 8.0 * SIGMA, X0 + 8.0 * SIGMA, 200001)
    du = np.abs(-(y - X0) / SIGMA ** 2 * u0(y))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (du[1:] + du[:-1])
                                           * np.diff(y))])
    cdf /= cdf[-1]
    r = (np.arange(1, N + 1) - 0.5) / N
    x_p = np.interp(r, cdf, y)
    tv = 2.0 * B_AMP
    m_p = np.where(x_p < X0, tv / N, -tv / N)
    return x_p, m_p, 0.0


def reference_solution(x_out, t, epsrel=1e-10):
    """Cole--Hopf whole-line solution by stabilized adaptive quadrature."""
    from scipy.integrate import quad
    u = np.empty_like(x_out, dtype=float)
    for k, x in enumerate(np.asarray(x_out, dtype=float)):
        def G(y):
            return u0_antiderivative(y) + (x - y) ** 2 / (2.0 * t)
        yg = np.linspace(x - 30.0 * math.sqrt(t), x + 30.0 * math.sqrt(t), 4001)
        Gmin = float(np.min(G(yg)))
        def num(y):
            return (x - y) / t * math.exp(-(G(y) - Gmin) / (2.0 * NU))
        def den(y):
            return math.exp(-(G(y) - Gmin) / (2.0 * NU))
        lo, hi = float(yg[0]), float(yg[-1])
        n_val, _ = quad(num, lo, hi, epsabs=0.0, epsrel=epsrel, limit=400)
        d_val, _ = quad(den, lo, hi, epsabs=0.0, epsrel=epsrel, limit=400)
        u[k] = n_val / d_val
    return u


def crank_nicolson_check(x_out, t, dx=0.004, dt=2.0e-4):
    """Independent finite-difference solve (diffusion implicit, advection
    explicit) on a padded interval; used only to document the reference
    tolerance."""
    from scipy.linalg import solve_banded
    lo, hi = X0 - 10.0, X0 + 12.0
    m = int(round((hi - lo) / dx)) + 1
    xg = np.linspace(lo, hi, m)
    u = u0(xg)
    steps = int(round(t / dt))
    lam = NU * dt / dx ** 2
    ab = np.zeros((3, m))
    ab[0, 1:] = -0.5 * lam
    ab[1, :] = 1.0 + lam
    ab[2, :-1] = -0.5 * lam
    ab[1, 0] = ab[1, -1] = 1.0
    ab[0, 1] = ab[2, -2] = 0.0
    for _ in range(steps):
        flux = 0.5 * u ** 2
        dflux = np.zeros_like(u)
        dflux[1:-1] = (flux[2:] - flux[:-2]) / (2.0 * dx)
        rhs = u - dt * dflux
        rhs[1:-1] += 0.5 * lam * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        rhs[0] = 0.0
        rhs[-1] = 0.0
        u = solve_banded((1, 1), ab, rhs)
    return np.interp(x_out, xg, u)


def _rngs(dt, seed_idx):
    dk = int(round(dt * 1_000_000))
    rng_label = np.random.default_rng(
        np.random.SeedSequence([SEED_LABEL_BASE, dk, seed_idx]))
    rng_brownian = np.random.default_rng(
        np.random.SeedSequence([SEED_BROWNIAN_BASE, dk, seed_idx]))
    return rng_label, rng_brownian


def _fingerprint(S_use):
    return {'N': N_FIXED, 'nu': NU, 'T': T_FINAL, 'N_OUT': N_OUT,
            'B': B_AMP, 'x0': X0, 'sigma': SIGMA,
            'window': [X_LO, X_HI], 'S': int(S_use),
            'dt_seq': list(DT_SEQ), 'arms': [a[0] for a in ARMS],
            'seed_label_base': SEED_LABEL_BASE,
            'seed_brownian_base': SEED_BROWNIAN_BASE,
            'seed_design': 'SeedSequence([base, round(dt*1e6), seed_idx])',
            'diagnostic': 'no profile fit; paired L2 differences vs control'}


def _paired_ci(diffs, rng, n_boot=5000):
    diffs = np.asarray(diffs)
    boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'mean': float(diffs.mean()), 'ci_lo': float(lo), 'ci_hi': float(hi),
            'resolves_sign': bool(lo > 0 or hi < 0)}


def run_study(S_override=None, out_base=None):
    S_use = int(S_override) if S_override else S
    fp = _fingerprint(S_use)
    out_dir = out_base or OUT_BASE
    os.makedirs(out_dir, exist_ok=True)
    x_out = np.linspace(X_LO, X_HI, N_OUT)
    dx = float(x_out[1] - x_out[0])

    # Reference: computed once, archived, with a documented tolerance.
    ref_path = os.path.join(out_dir, 'reference.npz')
    if os.path.exists(ref_path):
        z = np.load(ref_path)
        u_ref = z['u_ref']
        ref_meta = json.loads(str(z['meta']))
    else:
        print('computing Cole-Hopf quadrature reference ...')
        t0 = time.perf_counter()
        u_ref = reference_solution(x_out, T_FINAL, epsrel=1e-10)
        u_ref_coarse = reference_solution(x_out, T_FINAL, epsrel=1e-8)
        self_consistency = float(np.max(np.abs(u_ref - u_ref_coarse)))
        print(f'  quadrature self-consistency (1e-10 vs 1e-8): '
              f'{self_consistency:.3e}')
        print('computing independent Crank-Nicolson check ...')
        u_cn = crank_nicolson_check(x_out, T_FINAL)
        cn_agreement = float(np.max(np.abs(u_ref - u_cn)))
        print(f'  quadrature vs Crank-Nicolson: {cn_agreement:.3e} '
              f'({time.perf_counter() - t0:.1f}s total)')
        ref_meta = {'epsrel': 1e-10,
                    'self_consistency_max_abs': self_consistency,
                    'crank_nicolson_max_abs': cn_agreement,
                    'reference_tolerance': max(self_consistency, cn_agreement),
                    'cn_grid': {'dx': 0.004, 'dt': 2.0e-4,
                                'interval': [X0 - 10.0, X0 + 12.0]}}
        _savez_deterministic(ref_path, u_ref=u_ref, x=x_out,
                             meta=np.array(json.dumps(ref_meta)))

    manifest_path = os.path.join(out_dir, 'manifest.json')
    done = set()
    per_run = []
    l2_by = {}
    prof = {}

    def _cell_npz(dt_c, arm_c):
        return os.path.join(out_dir,
                            f'cell_{arm_c}_dt{dt_c:g}'.replace('.', 'p') + '.npz')

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
            l2_by.setdefault((float(row['dt']), row['arm']), {})[
                int(row['seed_idx'])] = float(row['l2'])
        for (dt_c, arm_c) in claimed:
            if not os.path.exists(_cell_npz(dt_c, arm_c)):
                continue
            arr = np.load(_cell_npz(dt_c, arm_c))['profiles']
            if arr.shape != (S_use, N_OUT):
                continue
            if sorted(l2_by.get((dt_c, arm_c), {})) != list(range(S_use)):
                continue
            prof[(dt_c, arm_c)] = arr
            done.add((dt_c, arm_c))
        per_run = [r for r in per_run if (float(r['dt']), r['arm']) in done]
        l2_by = {k: v for k, v in l2_by.items() if k in done}
        print(f"resuming: {len(done)}/{len(claimed)} claimed cells are complete")

    print(f"Smooth transient: nu={NU}, T={T_FINAL}, dt={DT_SEQ}, "
          f"arms={[a[0] for a in ARMS]}, N={N_FIXED}, M={N_OUT}, S={S_use}, "
          f"reference tolerance {ref_meta['reference_tolerance']:.2e}")
    for dt in DT_SEQ:
        n_steps = int(round(T_FINAL / dt))
        for arm_id, transport, a_rel in ARMS:
            if (dt, arm_id) in done:
                continue
            profiles, l2s = [], {}
            for s in range(S_use):
                x0p, m0p, u_left = initialize_gaussian_gradient_particles(N_FIXED)
                rng_label, rng_brownian = _rngs(dt, s)
                t0 = time.perf_counter()
                run = advance_rbgbmc_particles(
                    x0p, m0p, u_left, NU, a_rel, dt, n_steps, rng_label,
                    rng_brownian=rng_brownian,
                    collect_label_diagnostics=True,
                    conditional_mean_transport=(transport == 'conditional_mean'))
                u_out = reconstruct_cumulative_field(run['x'], run['m'],
                                                     u_left, x_out)
                l2 = float(np.sqrt(np.sum((u_out - u_ref) ** 2) * dx))
                mass = float(run['m'].sum())
                if not np.all(np.isfinite(u_out)) or abs(mass) > 1e-12:
                    raise RuntimeError(
                        f"health failure dt={dt} arm={arm_id} seed={s}: "
                        f"mass={mass!r}")
                profiles.append(u_out)
                l2s[s] = l2
                per_run.append({'dt': dt, 'arm': arm_id, 'transport': transport,
                                'a': a_rel, 'seed_idx': s, 'l2': l2,
                                'mass_final': mass,
                                'label_excess_mean': run['label_excess_mean'],
                                'D_label_run': 0.5 * dt * run['label_excess_mean'],
                                'runtime_s': time.perf_counter() - t0})
            prof[(dt, arm_id)] = np.asarray(profiles)
            l2_by[(dt, arm_id)] = l2s
            done.add((dt, arm_id))
            with open(os.path.join(out_dir, 'per_run.csv'), 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(per_run[0]))
                w.writeheader(); w.writerows(per_run)
            _savez_deterministic(_cell_npz(dt, arm_id), profiles=prof[(dt, arm_id)])
            json.dump({'config': fp, 'done': sorted(done)},
                      open(manifest_path, 'w'))
            arr = np.array(list(l2s.values()))
            print(f"  dt={dt:<8g} {arm_id:<13s} mean L2={arr.mean():.5f} "
                  f"(std {arr.std():.5f})")

    # ---------------------------------------------------------------- analysis
    rng = np.random.default_rng(20260826)
    summary, paired = [], []
    for dt in DT_SEQ:
        for arm_id, transport, a_rel in ARMS:
            if (dt, arm_id) not in prof:
                continue
            u_arr = prof[(dt, arm_id)]
            um = u_arr.mean(axis=0)
            E_bias = float(np.sqrt(np.sum((um - u_ref) ** 2 * dx)))
            E_spread = float(np.sqrt(np.mean(
                np.sum((u_arr - um[None, :]) ** 2 * dx, axis=1))))
            E_total = float(np.sqrt(np.mean(
                np.sum((u_arr - u_ref[None, :]) ** 2 * dx, axis=1))))
            d_runs = [float(r['D_label_run']) for r in per_run
                      if abs(float(r['dt']) - dt) < 1e-15 and r['arm'] == arm_id]
            summary.append({'dt': dt, 'arm': arm_id, 'a': a_rel, 'S': S_use,
                            'E_bias': E_bias, 'E_spread': E_spread,
                            'E_total': E_total,
                            'identity_error': abs(E_total ** 2 - E_bias ** 2
                                                  - E_spread ** 2),
                            'D_label_run_mean': float(np.mean(d_runs))})
        def dl2(arm_a):
            a_map, b_map = l2_by.get((dt, arm_a)), l2_by.get((dt, 'cond_mean'))
            if not a_map or not b_map:
                return None
            seeds = sorted(set(a_map) & set(b_map))
            d = np.array([a_map[s] - b_map[s] for s in seeds])
            return {'dt': dt, 'contrast': f'{arm_a} - cond_mean',
                    'n_pairs': len(seeds), **_paired_ci(d, rng)}
        for arm_a in ('two_speed_a2', 'two_speed_a4'):
            iv = dl2(arm_a)
            if iv:
                paired.append(iv)

    out = {'metadata': {
                'purpose': ('smooth nonstationary transient: does the a- and '
                            'dt-dependent label mechanism persist away from '
                            'the equal-mass stationary reconstruction?'),
                'problem': (f'u0 = {B_AMP:g}*exp(-(x-{X0:g})^2/(2*{SIGMA:g}^2)), '
                            f'nu = {NU:g}, T = {T_FINAL:g}, whole line, '
                            f'window [{X_LO:g}, {X_HI:g}] with {N_OUT} points'),
                'reference': ref_meta,
                'dt_seq': DT_SEQ, 'arms': [a[0] for a in ARMS],
                'N': N_FIXED, 'S': S_use,
                'n_failed': 0,
                'pairing': ('arms share Brownian streams per (dt, seed); '
                            'paired within dt only, aligned by sorted rank'),
                'command': 'python studies/study_smooth_transient.py'},
           'summary': summary, 'paired_l2_contrasts': paired}
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
    args = ap.parse_args()
    run_study(S_override=args.seeds, out_base=args.out)
