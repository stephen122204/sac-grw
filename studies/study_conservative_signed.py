"""Round-3 part 2: conservative signed Brownian gradient dynamics.

Screens whether enforcing the PDE's physical first-moment balance inside each
realization gives a useful FULL-FIELD improvement on a mixed-sign transient,
after removing the known drift repair and against a strong antithetic baseline.

Known ingredients, not claimed novel: the secant drift is a standard
conservative rank drift; the centred projection is the known centred rank
diffusion with an elementary marginal-variance restoration (Reygner eqs 10-11);
conservation plus variance reduction is not a new principle (Degond-Dimarco-
Pareschi).

Physics.  With inclusive-rank reconstruction, sum m_i u_right_i =
[u+^2-u-^2]/2 + sum m_i^2/2, so inclusive drift carries a deterministic O(1/N)
first-moment defect.  Secant drift v_i=[f(u_r)-f(u_l)]/m_i = (u_l+u_r)/2 for
Burgers telescopes to f(u+)-f(u-) exactly.  Independent noise adds a random
walk to sum m_i X_i of variance 2 nu T sum m_i^2; the signed projection
Zc = (Z - s (s.Z)/N)/sqrt(1-1/N) removes it while keeping unit marginals.
Since integral (u - u_-) dx = L sum m_i - sum m_i X_i, that is exactly the
physical area/centroid balance.

All arms use the SAME splitting schedule and the SAME initial representation,
in production's sorted frame, so arm 1 is production mean transport.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import (advance_rbgbmc_particles,
                             initialize_tanh_shock_particles,
                             reconstruct_cumulative_field)
from studies.study_smooth_transient import initialize_gaussian_gradient_particles

OUT = ROOT / 'output/conservative_signed_2026_09_09'
PK = {'shock': 101, 'gaussian': 202}
CFG = {'shock':    dict(a=2., nu=.5, T=2.5, h=.005),
       'gaussian': dict(a=4., nu=.1, T=1.,  h=.005)}
NS = (1600, 6400)
SEEDS = 64
ARMS = ['inclusive_indep', 'secant_indep', 'secant_centered_novar',
        'secant_centered', 'rank_anti_inclusive', 'rank_anti_secant']
PAIR_ARMS = {'rank_anti_inclusive', 'rank_anti_secant'}


def setup(problem, nu, N):
    if problem == 'shock':
        x = np.linspace(-10., 14., 400)
        return x, -np.tanh((x - 2.) / (2 * nu)), initialize_tanh_shock_particles(N, nu, 1., 2.)
    z = np.load(ROOT / 'output/final_prepublication_tests/'
                       'gbmc_smooth_transient/reference.npz')
    return z['x'], z['u_ref'], initialize_gaussian_gradient_particles(N)


def advance(initial, nu, dt, K, rng, drift='inclusive', noise='independent',
            anti=False):
    """One arm. Sorted frame, production schedule: transport with the carried
    velocity, sort, reconstruct, set next velocity, diffuse."""
    x, m, u_left = np.array(initial[0]), np.array(initial[1]), initial[2]
    o = np.argsort(x, kind='stable'); x, m = x[o], m[o]
    N = len(x)
    u_r = u_left + np.cumsum(m); u_l = u_r - m
    v = u_r if drift == 'inclusive' else 0.5 * (u_l + u_r)
    sd = np.sqrt(2.0 * nu * dt)
    for _ in range(K):
        x = x + v * dt
        o = np.argsort(x, kind='stable'); x, m = x[o], m[o]
        u_r = u_left + np.cumsum(m); u_l = u_r - m
        v = u_r if drift == 'inclusive' else 0.5 * (u_l + u_r)
        z = rng.normal(0.0, sd, size=N)
        if anti:
            z = -z
        if noise != 'independent':
            s = np.sign(m)
            z = z - s * (s @ z) / N            # centred: sum m_i z_i = 0
            if noise == 'centered':
                z = z / np.sqrt(1.0 - 1.0 / N)  # restore unit marginal variance
        x = x + z
    return x, m, u_left


def diagnostics(x, m, x_out, u_left, ref, dx, ref_shape):
    """Full-field error plus whole-line conservation and shape diagnostics."""
    f = reconstruct_cumulative_field(x, m, u_left, x_out)
    first = float(np.sum(m * x))                    # whole line, not truncated
    aw = np.abs(m); xbar = float(np.sum(aw * x) / np.sum(aw))
    shape = float(np.sum(aw * (x - xbar) ** 2) / np.sum(aw))
    return f, first, shape, xbar


def ref_shape_moment(x_out, ref):
    g = np.abs(np.gradient(ref, x_out))
    if g.sum() == 0:
        return np.nan
    xb = float(np.sum(g * x_out) / np.sum(g))
    return float(np.sum(g * (x_out - xb) ** 2) / np.sum(g))


def main(seeds=SEEDS):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, fields = [], {}
    for problem, c in CFG.items():
        for N in NS:
            x_out, ref, init = setup(problem, c['nu'], N)
            dx = float(x_out[1] - x_out[0]); K = round(c['T'] / c['h'])
            rshape = ref_shape_moment(x_out, ref)
            first0 = float(np.sum(init[1] * init[0]))
            # initialization audit: represented area vs exact, at t=0
            f0 = reconstruct_cumulative_field(init[0], init[1], init[2], x_out)
            init_area_err = float(dx * np.sum(f0 - ref)) if problem == 'gaussian' else np.nan
            print(f"\n{problem} N={N} h={c['h']} K={K} seeds={seeds}"
                  f"   init sum m X = {first0:.6f}"
                  + (f"   init area err = {init_area_err:+.3e}" if problem == 'gaussian' else ""))
            print("-" * 96)
            for arm in ARMS:
                drift = 'inclusive' if 'inclusive' in arm else 'secant'
                noise = ('centered' if arm == 'secant_centered' else
                         'centered_raw' if arm == 'secant_centered_novar' else 'independent')
                is_pair = arm in PAIR_ARMS
                t0 = time.perf_counter(); F, FM, SH = [], [], []
                for s in range(seeds):
                    if is_pair:
                        key = np.random.SeedSequence([PK[problem], N, 71, s])
                        acc = []
                        for anti in (False, True):
                            r = np.random.default_rng(key)
                            st = advance(init, c['nu'], c['h'], K, r, drift,
                                         'independent', anti=anti)
                            acc.append(st)
                        f1, m1, sh1, _ = diagnostics(*acc[0][:2], x_out, init[2], ref, dx, rshape)
                        f2, m2, sh2, _ = diagnostics(*acc[1][:2], x_out, init[2], ref, dx, rshape)
                        F.append((f1 + f2) / 2); FM.append((m1 + m2) / 2); SH.append((sh1 + sh2) / 2)
                    else:
                        r = np.random.default_rng(
                            np.random.SeedSequence([PK[problem], N, 61, s]))
                        st = advance(init, c['nu'], c['h'], K, r, drift, noise)
                        f, fm, sh, _ = diagnostics(*st[:2], x_out, init[2], ref, dx, rshape)
                        F.append(f); FM.append(fm); SH.append(sh)
                cost = (time.perf_counter() - t0) / seeds
                F = np.array(F); FM = np.array(FM); SH = np.array(SH)
                mse = float(np.mean(dx * np.sum((F - ref) ** 2, axis=1)))
                var = float(dx * np.sum(F.var(axis=0, ddof=1)))
                raw_b2 = float(dx * np.sum((F.mean(0) - ref) ** 2))
                b2 = raw_b2 - var / seeds                      # signed, debiased
                rows.append(dict(problem=problem, N=N, arm=arm, seeds=seeds,
                                 pair=is_pair, mse=mse, bias2_signed=b2, var=var,
                                 K_coef=var * cost, cost_s=cost,
                                 first_moment_drift=float(FM.mean() - first0),
                                 first_moment_sd=float(FM.std(ddof=1)),
                                 shape=float(SH.mean()), shape_ref=rshape))
                q = rows[-1]
                print(f"  {arm:>22} mse={mse:.4e} bias2={b2:+.3e} var={var:.4e} "
                      f"K={q['K_coef']:.3e} t={cost:.4f}s | "
                      f"d(sum mX)={q['first_moment_drift']:+.3e} "
                      f"sd={q['first_moment_sd']:.3e}")
                fields[f'{problem}_{N}_{arm}'] = F
            fields[f'{problem}_{N}_ref'] = ref
            fields[f'{problem}_{N}_x'] = x_out
    np.savez_compressed(OUT / 'fields.npz', **fields)
    meta = dict(rows=rows, config={k: dict(v) for k, v in CFG.items()}, Ns=list(NS),
                source_sha=hashlib.sha256((ROOT/'relaxation_gbmc.py').read_bytes()).hexdigest()[:16],
                driver_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                estimator='full-field L2 vs reference; pair arms average 2 runs',
                note='bias2 is signed and finite-ensemble debiased (raw - var/S)')
    (OUT / 'conservative_signed.json').write_text(json.dumps(meta, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
