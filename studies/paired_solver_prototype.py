"""Paired advance of two complete simulations, the prototype of coupled_gradient_particles.py (Section 3.2)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradient_particles import (advance_particles,
                             initialize_tanh_shock_particles,
                             reconstruct_cumulative_field)
from studies.smooth_transient import initialize_gaussian_gradient_particles
from studies import twopulse_reference as TP

OUT = ROOT / 'output/paired_solver_prototype'
PK = {'shock': 101, 'gaussian': 202, 'twopulse': 303}
COUPLINGS = ['raw_rank', 'sign_adjusted', 'within_sign', 'independent']
SEEDS = 64
CASES = [('shock', 1600), ('gaussian', 1600), ('gaussian', 6400),
         ('twopulse', 1600), ('twopulse', 6400)]
PAR = {'shock':    dict(nu=.5, a=2., T=2.5, h=.005),
       'gaussian': dict(nu=.1, a=4., T=1.,  h=.005),
       'twopulse': dict(nu=.1, a=2., T=1.,  h=.005)}


def setup(problem, N):
    p = PAR[problem]
    if problem == 'shock':
        x = np.linspace(-10., 14., 400)
        return x, -np.tanh((x - 2.) / (2 * p['nu'])), initialize_tanh_shock_particles(N, p['nu'], 1., 2.), None
    if problem == 'gaussian':
        z = np.load(ROOT / 'output/final_prepublication_tests/'
                           'gbmc_smooth_transient/reference.npz')
        return z['x'], z['u_ref'], initialize_gaussian_gradient_particles(N), None
    cache = OUT / 'twopulse_reference.npz'
    x = np.linspace(-5., 5., 400)
    if cache.exists():
        d = np.load(cache); ref = d['u_ref']
    else:
        ref = TP.cole_hopf(x, p['T'], p['nu'])
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(cache, x=x, u_ref=ref, nu=p['nu'], T=p['T'])
    xi, mi, ul, info = TP.initialize(N)
    return x, ref, (xi, mi, ul), info


def _noise(coupling, mA, mB, rng, sd, N):
    """Returns (zA, zB). Signs/permutation are predictable: they depend only on
    the pre-diffusion masses at matched spatial ranks, never on the drawn Z."""
    z = rng.standard_normal(N)
    if coupling == 'independent':
        return sd * z, sd * rng.standard_normal(N)
    if coupling == 'raw_rank':
        return sd * z, -sd * z
    sA, sB = np.sign(mA), np.sign(mB)
    if coupling == 'sign_adjusted':
        return sd * z, -sd * (sA * sB) * z
    if coupling == 'within_sign':
        zB = np.empty(N)
        for sgn in (1.0, -1.0):
            iA = np.flatnonzero(sA == sgn); iB = np.flatnonzero(sB == sgn)
            n = min(len(iA), len(iB))
            zB[iB[:n]] = -z[iA[:n]]
            if len(iB) > n:                       # counts are equal here, but be safe
                zB[iB[n:]] = rng.standard_normal(len(iB) - n)
        return sd * z, sd * zB
    raise ValueError(coupling)


def advance_pair(initial, nu, dt, K, rng, coupling, record_disagree=False):
    x0, m0, u_left = np.asarray(initial[0], float), np.asarray(initial[1], float), initial[2]
    o = np.argsort(x0, kind='stable')
    xA, mA = x0[o].copy(), m0[o].copy(); xB, mB = xA.copy(), mA.copy()
    N = len(xA); sd = np.sqrt(2.0 * nu * dt)
    vA = u_left + np.cumsum(mA); vB = vA.copy()
    dis = []
    for k in range(K):
        xA = xA + vA * dt
        o = np.argsort(xA, kind='stable'); xA, mA = xA[o], mA[o]
        vA = u_left + np.cumsum(mA)
        xB = xB + vB * dt
        o = np.argsort(xB, kind='stable'); xB, mB = xB[o], mB[o]
        vB = u_left + np.cumsum(mB)
        if record_disagree:
            dis.append(float(np.mean(np.sign(mA) != np.sign(mB))))
        zA, zB = _noise(coupling, mA, mB, rng, sd, N)
        xA = xA + zA; xB = xB + zB
    return (xA, mA), (xB, mB), u_left, np.array(dis)


def run_case(problem, N, seeds, do_disagree=True):
    p = PAR[problem]
    x_out, ref, init, info = setup(problem, N)
    dx = float(x_out[1] - x_out[0]); K = round(p['T'] / p['h'])
    phi = np.exp(-(x_out - 0.0) ** 2 / 2.0)          # smooth field observable
    Jref = float(dx * np.sum(ref * phi))
    rows, store = [], {}
    for coup in COUPLINGS:
        t0 = time.perf_counter(); PM, FM, J, DIS, secs = [], [], [], [], []
        for s in range(seeds):
            r = np.random.default_rng(np.random.SeedSequence([PK[problem], N, 51, s]))
            ts = time.perf_counter()
            A, B, ul, dis = advance_pair(init, p['nu'], p['h'], K, r, coup,
                                         record_disagree=(do_disagree and s == 0))
            secs.append(time.perf_counter() - ts)
            fa = reconstruct_cumulative_field(A[0], A[1], ul, x_out)
            fb = reconstruct_cumulative_field(B[0], B[1], ul, x_out)
            pm = (fa + fb) / 2
            PM.append(pm)
            FM.append((float(np.sum(A[1] * A[0])) + float(np.sum(B[1] * B[0]))) / 2)
            J.append(float(dx * np.sum(pm * phi)))
            if len(dis): DIS = dis
        cost = (time.perf_counter() - t0) / seeds
        PM = np.array(PM); FM = np.array(FM); J = np.array(J)
        mse = float(np.mean(dx * np.sum((PM - ref) ** 2, axis=1)))
        var = float(dx * np.sum(PM.var(axis=0, ddof=1)))
        b2 = float(dx * np.sum((PM.mean(0) - ref) ** 2)) - var / seeds
        rows.append(dict(problem=problem, N=N, coupling=coup, seeds=seeds,
                         mse=mse, var=var, bias2_signed=b2, cost_s=cost,
                         K_coef=var * cost, mse_cost=mse * cost,
                         J_err=float(np.mean(np.abs(J - Jref))),
                         first_moment_mean=float(FM.mean()),
                         first_moment_sd=float(FM.std(ddof=1)),
                         median_seed_s=float(np.median(secs))))
        store[coup] = PM
        if len(DIS): store[f'disagree_{coup}'] = DIS
    store['ref'] = ref; store['x'] = x_out
    return rows, store


def validate():
    print("VALIDATION"); print("-" * 74)
    # (a) one-sign: sign_adjusted must be pathwise identical to raw_rank
    x, m, ul = initialize_tanh_shock_particles(400, .5, 1., 2.)
    A1, B1, _, _ = advance_pair((x, m, ul), .5, .005, 30,
                                np.random.default_rng(5), 'raw_rank')
    A2, B2, _, _ = advance_pair((x, m, ul), .5, .005, 30,
                                np.random.default_rng(5), 'sign_adjusted')
    print(f"  one-sign pathwise: max|xA diff|={np.max(np.abs(A1[0]-A2[0])):.2e}  "
          f"max|xB diff|={np.max(np.abs(B1[0]-B2[0])):.2e}   (must be 0)")
    # (b) supplied-noise witness: arm A of a pair vs the production update
    K = 40
    r = advance_particles(x, m, ul, .5, 2., .005, K,
                                 np.random.default_rng(1),
                                 rng_brownian=np.random.default_rng(77),
                                 conditional_mean_transport=True)
    class Supply:                     # replays production's normals into arm A
        def __init__(s): s.g = np.random.default_rng(77)
        def standard_normal(s, n): return s.g.normal(0., 1., size=n)
    A3, _, _, _ = advance_pair((x, m, ul), .5, .005, K, Supply(), 'raw_rank')
    print(f"  marginal witness vs production: max|x diff| = "
          f"{np.max(np.abs(np.sort(A3[0])-np.sort(r['x']))):.2e}   (must be 0)")
    # (c) pair-mean Brownian first-moment increment
    xi, mi, uli, _ = TP.initialize(200)
    for coup in ('raw_rank', 'sign_adjusted', 'within_sign'):
        rng = np.random.default_rng(3)
        sA = np.sign(mi); sB = np.roll(sA, 1)      # force sign disagreement
        zA, zB = _noise(coup, mi, mi * np.where(sB == sA, 1, -1), rng,
                        1.0, len(mi))
        inc = 0.5 * (np.sum(mi * zA) + np.sum(mi * np.where(sB == sA, 1, -1) * zB))
        print(f"  pair-mean first-moment increment, {coup:>14}: {inc:+.3e}")
    print()


def main(seeds=SEEDS):
    OUT.mkdir(parents=True, exist_ok=True)
    validate()
    all_rows, fields = [], {}
    for problem, N in CASES:
        rows, store = run_case(problem, N, seeds)
        all_rows += rows
        for k, v in store.items():
            fields[f'{problem}_{N}_{k}'] = v
        base = [r for r in rows if r['coupling'] == 'raw_rank'][0]
        d = store.get('disagree_raw_rank')
        print(f"{problem} N={N}" + (f"   sign-disagreement fraction: "
              f"start {d[0]:.3f} mid {d[len(d)//2]:.3f} end {d[-1]:.3f}"
              if d is not None and len(d) else ""))
        print("-" * 92)
        for r in rows:
            print(f"  {r['coupling']:>14} mse={r['mse']:.4e} ({r['mse']/base['mse']:5.3f}) "
                  f"var={r['var']:.4e} bias2={r['bias2_signed']:+.2e} "
                  f"t={r['cost_s']:.4f}s  K={r['K_coef']:.3e} "
                  f"({r['K_coef']/base['K_coef']:5.3f})  |dJ|={r['J_err']:.3e}")
        print()
    np.savez_compressed(OUT / 'fields.npz', **fields)
    with (OUT / 'sign_coupling.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    (OUT / 'sign_coupling.json').write_text(json.dumps(dict(
        rows=all_rows, params=PAR, cases=[list(c) for c in CASES], seeds=seeds,
        solver_sha=hashlib.sha256((ROOT/'gradient_particles.py').read_bytes()).hexdigest()[:16],
        driver_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        refhelper_sha=hashlib.sha256((ROOT/'studies/twopulse_reference.py').read_bytes()).hexdigest()[:16],
        estimator='pair-mean field; every arm costs two solver paths'), indent=1))
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=SEEDS)
    main(ap.parse_args().seeds)
