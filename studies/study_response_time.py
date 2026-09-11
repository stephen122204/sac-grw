"""Time development of the paired transport effect at fixed viscosity, and
held-out validation of the derived modified-equation prediction.

Motivation.  The archived multi-viscosity sweeps vary the layer width 2*nu/A
and the dimensionless evolution time T*A^2/nu together, so the reported growth
of the paired fitted-viscosity difference toward D_vel has two candidate
explanations.  The implemented update is exactly invariant under the scaling
    X = A x / nu,   tau = A^2 t / nu,   U = u / A,
so the stationary-shock problem depends only on
    at = a/A,   dtt = dt A^2 / nu,   taut = T A^2 / nu,   N,
and on the evaluation window measured in shock widths.  Layer width is not an
independent parameter.  This driver therefore varies taut alone at fixed nu.

Predictions under test (fixed before any run here; see ANALYTICAL_EXTENSION.md):
  M2 (derived)  u_t + f(u)_x = d/dx[(nu + D(u)) u_x],  D(u) = (dt/2)(a^2-f'(u)^2)
  M1 (naive)    u_t + f(u)_x = (nu + D_vel) u_xx,      D_vel = (dt/2)<a^2-f'(u)^2>
Each prediction is evolved from the exact nu-shock to T and passed through the
SAME three-parameter tanh fit on the SAME window as the particle profiles, so
the predicted paired difference carries no fitted coefficient.

Modes:
  A  time development at fixed nu (the primary study)
  B  exact-scaling equivalence check between two physically different (A, nu)
  C  held-out dimensionless configurations not used to develop the model

Usage:
  python studies/study_response_time.py --mode A --seeds 50
  python studies/study_response_time.py --mode B
  python studies/study_response_time.py --mode C --seeds 50
"""

import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxation_gbmc import (advance_rbgbmc_particles,
                             initialize_tanh_shock_particles,
                             reconstruct_cumulative_field)

OUT_BASE = os.path.join('output', 'reevaluation_2026_09', 'response_time')

# ---- frozen design constants -------------------------------------------------
XC = 2.0                 # shock center
N_FIXED = 6400           # production particle count (paper convention)
N_OUT = 400              # reconstruction points (paper convention)
HALF_WIDTHS = 12.0       # evaluation half-window in shock widths (2*nu/A)
S_DEFAULT = 50
SEED_LABEL_BASE = 930000
SEED_BROWNIAN_BASE = 940000

# Study A: nu fixed, dimensionless step fixed, dimensionless time varied.
A_MODE = dict(A=1.0, nu=0.5, dtt=0.005, at_list=(2.0, 4.0),
              taut_list=(0.5, 1.0, 2.0, 3.5, 5.0, 10.0, 20.0))

# Study C: held-out dimensionless configurations (at, dtt, taut).  None of
# these triples appears in the archived studies or in Study A.
C_CASES = [
    dict(name='C1', A=1.0, nu=0.5, at=3.0, dtt=0.005, taut=3.5),
    dict(name='C2', A=1.0, nu=0.5, at=2.0, dtt=0.020, taut=5.0),
    dict(name='C3', A=1.0, nu=0.5, at=1.5, dtt=0.010, taut=2.0),
    dict(name='C4', A=1.0, nu=0.5, at=4.0, dtt=0.020, taut=1.0),
]

# Study B: two physically different configurations that are dimensionless-equal.
B_PAIR = [dict(name='B1', A=1.0, nu=0.5, at=2.0, dtt=0.005, taut=10.0),
          dict(name='B2', A=2.0, nu=0.16, at=2.0, dtt=0.005, taut=10.0)]


# ---- helpers -----------------------------------------------------------------
def _versions():
    import scipy
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__}


def _key(*vals):
    return [int(round(float(v) * 1_000_000)) for v in vals]


def _rngs(tag, seed_idx):
    """Label and Brownian generators keyed by a configuration tag and seed
    index only, never by the arm, so all arms in a cell share the Brownian
    stream and are paired.  Brownian normals are drawn for the sorted particle
    array, so the pairing is by sorted rank, matching the archived studies."""
    ss = list(tag) + [int(seed_idx)]
    return (np.random.default_rng(np.random.SeedSequence([SEED_LABEL_BASE] + ss)),
            np.random.default_rng(np.random.SeedSequence([SEED_BROWNIAN_BASE] + ss)))


def _window(A, nu):
    half = HALF_WIDTHS * 2.0 * nu / A
    x_out = np.linspace(XC - half, XC + half, N_OUT)
    return x_out, float(x_out[1] - x_out[0])


def d_vel(a, A, dt, N=N_FIXED):
    """Particle-averaged one-step diffusion scale for the equal-mass shock."""
    return 0.5 * dt * (a ** 2 - A ** 2 * (1.0 / 3.0 + 2.0 / (3.0 * N ** 2)))


def fit_tanh3(x, u, A, nu, x_lo, x_hi):
    """Three-parameter tanh fit, same model and estimator as the archived
    studies.  Bounds follow the multi-viscosity convention, with the center
    bounded by the evaluation window.  Returns (A_hat, xc_hat, nu_hat, rms)."""
    from scipy.optimize import curve_fit

    def model(x_, A_, xc_, nu_):
        return -A_ * np.tanh(A_ * (x_ - xc_) / (2.0 * nu_))

    lo = [0.5 * A, x_lo, nu / 50.0]
    hi = [2.0 * A, x_hi, max(2.0 * A ** 2, 20.0 * nu)]
    popt, _ = curve_fit(model, x, u, p0=[A, XC, nu], bounds=(lo, hi), maxfev=20000)
    res = u - model(x, *popt)
    at_bound = bool(popt[2] <= 1.02 * lo[2] or popt[2] >= 0.98 * hi[2])
    return (float(popt[0]), float(popt[1]), float(popt[2]),
            float(np.sqrt(np.mean(res ** 2))), at_bound)


def run_cell(A, nu, at, dtt, taut_list, seeds, tag, arms=('cond_mean', 'two_speed')):
    """Advance one configuration for every seed and arm, recording a snapshot
    at each requested dimensionless time from a single trajectory."""
    a = at * A
    dt = dtt * nu / A ** 2
    steps = {t: int(round(t / dtt)) for t in taut_list}
    n_max = max(steps.values())
    x_out, dx = _window(A, nu)
    u_ref = -A * np.tanh(A * (x_out - XC) / (2.0 * nu))
    rows, profiles = [], {}
    for arm in arms:
        for s in range(seeds):
            x0, m0, u_left = initialize_tanh_shock_particles(N_FIXED, nu, A, XC)
            rl, rb = _rngs(tag, s)
            t0 = time.perf_counter()
            run = advance_rbgbmc_particles(
                x0, m0, u_left, nu, a, dt, n_max, rl, rng_brownian=rb,
                snapshot_steps=set(steps.values()),
                conditional_mean_transport=(arm == 'cond_mean'))
            el = time.perf_counter() - t0
            for taut, k in steps.items():
                xs, ms = run['snapshots'][k]
                u = reconstruct_cumulative_field(xs, ms, u_left, x_out)
                Ah, xch, nuh, rms, ab = fit_tanh3(x_out, u, A, nu,
                                                  x_out[0], x_out[-1])
                rows.append(dict(A=A, nu=nu, a=a, at=at, dt=dt, dtt=dtt,
                                 taut=taut, n_steps=k, arm=arm, seed=s,
                                 A_hat=Ah, xc_hat=xch, nu_hat=nuh,
                                 fit_rms=rms, at_bound=int(ab),
                                 l2_vs_exact=float(np.sqrt(np.sum((u - u_ref) ** 2) * dx)),
                                 runtime_s=el / len(steps)))
                profiles.setdefault((arm, taut), []).append(u)
    return rows, {k: np.asarray(v) for k, v in profiles.items()}, x_out, dx


def paired_stats(rows, profiles, x_out, dx, A, nu, a, dt, n_boot=5000, seed=20260908):
    """Paired differences with seed-level bootstrap.  Whole seeds are resampled,
    so a seed's group of time snapshots stays together and the correlation
    between times is preserved."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by[(r['arm'], r['taut'], r['seed'])] = r
    tauts = sorted({r['taut'] for r in rows})
    seeds = sorted({r['seed'] for r in rows})
    D = d_vel(a, A, dt)
    idx = np.array([rng.integers(0, len(seeds), len(seeds)) for _ in range(n_boot)])
    out = []
    for t in tauts:
        d = np.array([by[('two_speed', t, s)]['nu_hat'] - by[('cond_mean', t, s)]['nu_hat']
                      for s in seeds])
        dA = np.array([by[('two_speed', t, s)]['A_hat'] - by[('cond_mean', t, s)]['A_hat']
                       for s in seeds])
        dxc = np.array([by[('two_speed', t, s)]['xc_hat'] - by[('cond_mean', t, s)]['xc_hat']
                        for s in seeds])
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p2, pc = profiles[('two_speed', t)], profiles[('cond_mean', t)]
        mean_diff = p2.mean(axis=0) - pc.mean(axis=0)
        per_real = np.sqrt(np.sum((p2 - pc) ** 2, axis=1) * dx)
        out.append(dict(
            taut=t, D_vel=D, n_pairs=len(seeds),
            d_nu=float(d.mean()), d_nu_lo=float(lo), d_nu_hi=float(hi),
            d_nu_sd=float(d.std(ddof=1)),
            g=float(d.mean() / D), g_lo=float(lo / D), g_hi=float(hi / D),
            d_A=float(dA.mean()), d_xc=float(dxc.mean()),
            norm_of_mean_diff=float(np.sqrt(np.sum(mean_diff ** 2) * dx)),
            mean_of_per_real_diff_norm=float(per_real.mean()),
            nu_hat_control=float(np.mean([by[('cond_mean', t, s)]['nu_hat'] for s in seeds])),
            nu_hat_two_speed=float(np.mean([by[('two_speed', t, s)]['nu_hat'] for s in seeds])),
            fit_rms_two_speed=float(np.mean([by[('two_speed', t, s)]['fit_rms'] for s in seeds])),
            n_at_bound=int(sum(by[(arm, t, s)]['at_bound']
                               for arm in ('two_speed', 'cond_mean') for s in seeds)),
        ))
    return out


def _write(out_dir, name, rows, summary, meta, profiles=None, x_out=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{name}_per_run.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({'metadata': meta, 'summary': summary},
              open(os.path.join(out_dir, f'{name}_summary.json'), 'w'), indent=1)
    if profiles is not None:
        np.savez(os.path.join(out_dir, f'{name}_profiles.npz'), x_out=x_out,
                 **{f'{arm}_taut{t:g}'.replace('.', 'p'): v
                    for (arm, t), v in profiles.items()})


def mode_A(seeds, out_dir):
    cfg = A_MODE
    A, nu, dtt = cfg['A'], cfg['nu'], cfg['dtt']
    for at in cfg['at_list']:
        tag = _key(A, nu, at, dtt)
        print(f"[A] A={A} nu={nu} a/A={at} dt~={dtt} times={cfg['taut_list']} S={seeds}")
        rows, prof, x_out, dx = run_cell(A, nu, at, dtt, cfg['taut_list'], seeds, tag)
        summ = paired_stats(rows, prof, x_out, dx, A, nu, at * A, dtt * nu / A ** 2)
        meta = dict(mode='A', A=A, nu=nu, at=at, dtt=dtt, N=N_FIXED, N_OUT=N_OUT,
                    half_widths=HALF_WIDTHS, S=seeds, XC=XC,
                    taut_list=list(cfg['taut_list']), versions=_versions(),
                    pairing='arms share the Brownian stream within (config, seed); '
                            'normals are assigned by sorted particle rank',
                    bootstrap='whole seeds resampled; a seed keeps all of its '
                              'time snapshots together')
        _write(out_dir, f'A_at{at:g}'.replace('.', 'p'), rows, summ, meta, prof, x_out)
        for r in summ:
            print(f"   T~={r['taut']:6g}  d_nu={r['d_nu']:.6f} "
                  f"[{r['d_nu_lo']:.6f},{r['d_nu_hi']:.6f}]  g={r['g']:.4f}")


def mode_C(seeds, out_dir):
    for c in C_CASES:
        A, nu, at, dtt, taut = c['A'], c['nu'], c['at'], c['dtt'], c['taut']
        tag = _key(A, nu, at, dtt, taut) + [7]
        print(f"[C] {c['name']}: a/A={at} dt~={dtt} T~={taut} S={seeds}")
        rows, prof, x_out, dx = run_cell(A, nu, at, dtt, (taut,), seeds, tag)
        summ = paired_stats(rows, prof, x_out, dx, A, nu, at * A, dtt * nu / A ** 2)
        meta = dict(mode='C', case=c['name'], A=A, nu=nu, at=at, dtt=dtt, taut=taut,
                    N=N_FIXED, N_OUT=N_OUT, half_widths=HALF_WIDTHS, S=seeds,
                    XC=XC, versions=_versions())
        _write(out_dir, c['name'], rows, summ, meta, prof, x_out)
        r = summ[0]
        print(f"   d_nu={r['d_nu']:.6f} [{r['d_nu_lo']:.6f},{r['d_nu_hi']:.6f}]  "
              f"g={r['g']:.4f} [{r['g_lo']:.4f},{r['g_hi']:.4f}]")


def mode_B(seeds, out_dir):
    """Exact-scaling equivalence check.  B1 and B2 are different physical
    problems with identical dimensionless parameters, so their dimensionless
    statistics must agree; with matched random streams they agree to round-off.
    This is an implementation and scaling check, not independent validation."""
    res = {}
    for c in B_PAIR:
        A, nu, at, dtt, taut = c['A'], c['nu'], c['at'], c['dtt'], c['taut']
        tag = [11, 22]                                    # identical streams
        rows, prof, x_out, dx = run_cell(A, nu, at, dtt, (taut,), seeds, tag)
        summ = paired_stats(rows, prof, x_out, dx, A, nu, at * A, dtt * nu / A ** 2)
        res[c['name']] = dict(cfg=c, summary=summ[0],
                              U=(prof[('two_speed', taut)] / A).tolist()[:1],
                              X=((x_out - XC) * A / nu).tolist())
        print(f"[B] {c['name']}: A={A} nu={nu} -> d_nu/nu={summ[0]['d_nu']/nu:.8f} "
              f"g={summ[0]['g']:.6f}")
    X1 = np.array(res['B1']['X']); X2 = np.array(res['B2']['X'])
    U1 = np.array(res['B1']['U']); U2 = np.array(res['B2']['U'])
    dg = res['B1']['summary']['g'] - res['B2']['summary']['g']
    print(f"[B] max|X1-X2|={np.max(np.abs(X1-X2)):.3e}  "
          f"max|U1-U2|={np.max(np.abs(U1-U2)):.3e}  g1-g2={dg:.3e}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump({'metadata': dict(mode='B', S=seeds, versions=_versions(),
                                purpose='exact dimensionless equivalence check'),
               'g_B1': res['B1']['summary']['g'], 'g_B2': res['B2']['summary']['g'],
               'max_abs_X_diff': float(np.max(np.abs(X1 - X2))),
               'max_abs_U_diff': float(np.max(np.abs(U1 - U2))),
               'g_difference': float(dg)},
              open(os.path.join(out_dir, 'B_scaling_check.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['A', 'B', 'C'], required=True)
    ap.add_argument('--seeds', type=int, default=S_DEFAULT)
    ap.add_argument('--out', default=OUT_BASE)
    args = ap.parse_args()
    t0 = time.perf_counter()
    {'A': mode_A, 'B': mode_B, 'C': mode_C}[args.mode](args.seeds, args.out)
    print(f"done in {time.perf_counter() - t0:.1f} s")
