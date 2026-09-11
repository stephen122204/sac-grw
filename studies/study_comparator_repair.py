"""Round-3 part 1: repair the antithetic comparator and the equivalence check.

Two defects in Round 2 are addressed.

(a) The antithetic comparator used IDENTITY-based noise, while the production
    method negates the normals drawn FOR THE SORTED ARRAY (rank-based). Both
    give the same single-run marginal law but different PAIRED laws, so rho and
    K_anti are not comparable. This reruns the production rank-antithetic
    baseline at exactly the pilot configuration with fresh seeds.

(b) The isolated-stepper check compared two noisy mean norms, which is weak.
    Here a PATHWISE witness is built instead: the isolated stepper is run with
    production's rank-assignment convention and the same normal stream, so the
    two must agree to floating point. That isolates the noise-assignment
    convention as the only difference between the two implementations.
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

OUT = ROOT / 'output/comparator_repair_2026_09_09'
PK = {'shock': 101, 'gaussian': 202}
CFG = {'shock':    dict(a=2., nu=.5, T=2.5, h=.0025, N=1600),
       'gaussian': dict(a=4., nu=.1, T=1.,  h=.0025, N=1600)}
SEEDS = 64


def setup(problem, nu, N):
    if problem == 'shock':
        x = np.linspace(-10., 14., 400)
        return x, -np.tanh((x - 2.) / (2 * nu)), initialize_tanh_shock_particles(N, nu, 1., 2.)
    z = np.load(ROOT / 'output/final_prepublication_tests/'
                       'gbmc_smooth_transient/reference.npz')
    return z['x'], z['u_ref'], initialize_gaussian_gradient_particles(N)


def isolated_rank_convention(initial, nu, dt, K, rng, anti=False):
    """Isolated stepper using production's RANK noise convention, for the
    pathwise witness. Arrays are kept sorted exactly as production keeps them."""
    x, m, u_left = np.array(initial[0]), np.array(initial[1]), initial[2]
    o = np.argsort(x, kind='stable'); x, m = x[o], m[o]
    v = u_left + np.cumsum(m)
    sd = np.sqrt(2.0 * nu * dt)
    for _ in range(K):
        x = x + v * dt
        o = np.argsort(x, kind='stable'); x, m = x[o], m[o]
        v = u_left + np.cumsum(m)
        z = rng.normal(0.0, sd, size=len(x))
        x = x + (-z if anti else z)
    return x, m, u_left


def ivar(F, dx):
    return float(dx * np.sum(np.asarray(F).var(axis=0, ddof=1)))


def main(seeds=SEEDS):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    print("PATHWISE WITNESS: isolated stepper (rank convention) vs production")
    print("-" * 74)
    for p, c in CFG.items():
        x_out, ref, init = setup(p, c['nu'], c['N'])
        K = round(c['T'] / c['h'])
        s1 = isolated_rank_convention(init, c['nu'], c['h'], K,
                                      np.random.default_rng(12345))
        r = advance_rbgbmc_particles(*init, c['nu'], c['a'], c['h'], K,
                                     np.random.default_rng(999),
                                     rng_brownian=np.random.default_rng(12345),
                                     conditional_mean_transport=True)
        f1 = reconstruct_cumulative_field(*s1, x_out)
        f2 = reconstruct_cumulative_field(r['x'], r['m'], init[2], x_out)
        print(f"  {p:>9}: max|field diff| = {np.max(np.abs(f1 - f2)):.3e}   "
              f"max|position diff| = {np.max(np.abs(np.sort(s1[0]) - np.sort(r['x']))):.3e}")
        res.setdefault(p, {})['pathwise_max_field_diff'] = float(np.max(np.abs(f1 - f2)))

    print("\nPRODUCTION RANK-ANTITHETIC at the pilot configuration")
    print("-" * 74)
    for p, c in CFG.items():
        x_out, ref, init = setup(p, c['nu'], c['N'])
        dx = float(x_out[1] - x_out[0]); K = round(c['T'] / c['h'])
        # single-level reference group (independent seeds)
        t0 = time.perf_counter()
        P = []
        for s in range(seeds):
            rb = np.random.default_rng(np.random.SeedSequence([PK[p], 31, s]))
            rl = np.random.default_rng(np.random.SeedSequence([PK[p], 32, s]))
            rr = advance_rbgbmc_particles(*init, c['nu'], c['a'], c['h'], K, rl,
                                          rng_brownian=rb,
                                          conditional_mean_transport=True)
            P.append(reconstruct_cumulative_field(rr['x'], rr['m'], init[2], x_out))
        C_sl = (time.perf_counter() - t0) / seeds
        P = np.array(P); V_sl = ivar(P, dx)
        # rank-antithetic pairs (independent seeds); one sample = one pair
        t0 = time.perf_counter(); A, FA, FB = [], [], []
        for s in range(seeds):
            key = np.random.SeedSequence([PK[p], 41, s])
            ra = np.random.default_rng(key); rb2 = np.random.default_rng(key)
            rl1 = np.random.default_rng(np.random.SeedSequence([PK[p], 42, s]))
            rl2 = np.random.default_rng(np.random.SeedSequence([PK[p], 42, s]))
            r1 = advance_rbgbmc_particles(*init, c['nu'], c['a'], c['h'], K, rl1,
                                          rng_brownian=ra, antithetic=False,
                                          conditional_mean_transport=True)
            r2 = advance_rbgbmc_particles(*init, c['nu'], c['a'], c['h'], K, rl2,
                                          rng_brownian=rb2, antithetic=True,
                                          conditional_mean_transport=True)
            fa = reconstruct_cumulative_field(r1['x'], r1['m'], init[2], x_out)
            fb = reconstruct_cumulative_field(r2['x'], r2['m'], init[2], x_out)
            FA.append(fa); FB.append(fb); A.append((fa + fb) / 2)
        C_an = (time.perf_counter() - t0) / seeds
        A, FA, FB = np.array(A), np.array(FA), np.array(FB)
        V_an = ivar(A, dx)
        ca, cb = FA - FA.mean(0), FB - FB.mean(0)
        rho = float((ca * cb).sum() / np.sqrt((ca ** 2).sum() * (cb ** 2).sum()))
        vA, vB = ivar(FA, dx), ivar(FB, dx)
        cov = float(dx * np.sum(((FA - FA.mean(0)) * (FB - FB.mean(0))).sum(0) / (seeds - 1)))
        ident = (vA + vB + 2 * cov) / 4
        K_SL, K_AN = V_sl * C_sl, V_an * C_an
        print(f"  {p:>9}: V_SL={V_sl:.4e} C={C_sl:.4f}s | rank-anti V={V_an:.4e} "
              f"C={C_an:.4f}s rho={rho:+.4f}")
        print(f"             K_SL={K_SL:.4e}  K_rank-anti={K_AN:.4e}  ratio={K_AN/K_SL:.4f}")
        print(f"             identity check Var((A+B)/2)={V_an:.6e} vs "
              f"(vA+vB+2cov)/4={ident:.6e}  rel={abs(ident/V_an-1):.2e}")
        print(f"             partner var / standalone var = "
              f"{(vA+vB)/2/V_sl:.4f}")
        np.savez_compressed(OUT / f'{p}_rank_antithetic.npz', x_out=x_out, ref=ref,
                            P=P, A=A, FA=FA, FB=FB, seed_ids=np.arange(seeds))
        res[p].update(dict(V_SL=V_sl, C_SL=C_sl, V_rank_anti=V_an, C_rank_anti=C_an,
                           rho_rank=rho, K_SL=K_SL, K_rank_anti=K_AN,
                           var_identity_lhs=V_an, var_identity_rhs=ident,
                           partner_over_standalone_var=float((vA + vB) / 2 / V_sl),
                           N=c['N'], h=c['h'], T=c['T'], seeds=seeds))
    res['source_sha'] = hashlib.sha256((ROOT / 'relaxation_gbmc.py').read_bytes()).hexdigest()[:16]
    res['driver_sha'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    (OUT / 'comparator_repair.json').write_text(json.dumps(res, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
