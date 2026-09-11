"""Round-2 pilot: is an identity-consistent Brownian time coupling worth it?

Question. Can a coupling across TIME resolutions of the mean-transport gradient
solver give a complete multilevel estimator whose variance-cost coefficient
beats the same finest-level single-level estimator AND its cheap antithetic
competitor?  Feasibility screen only: no novelty claim, no PDE-accuracy claim.

Why an isolated stepper.  The production solver sorts the particle arrays every
step and draws its Brownian normals FOR THE SORTED ARRAY, i.e. noise is assigned
by current rank.  Coarse and fine runs exchange ranks, so sharing noise by array
index would silently couple by rank rather than by particle.  This module keeps
the arrays in IDENTITY order and computes ranks by argsort, so a shared normal
always reaches the same physical particle.  The two conventions are
distributionally identical (i.i.d. normals, permutation-invariant) but not
pathwise identical; `--verify` checks that equivalence before any data are used.

Coupling.  With h_c = 2 h_f, the coarse increment is the exact SUM of its two
fine increments:
    fine:   sqrt(2 nu h_f) Z_a ,  sqrt(2 nu h_f) Z_b
    coarse: sqrt(2 nu h_f) (Z_a + Z_b)     ~ N(0, 2 nu h_c)   (correct marginal)

Estimator figure of merit.  For per-sample variance V and per-sample cost C the
work to reach variance eps^2 is V*C/eps^2, so K = V*C is compared:
    K_SL   = V(P_L) C(P_L)
    K_ML   = ( sqrt(V0 C0) + sum_l sqrt(V_l C_l) )^2      (independent groups)
    K_anti = V(pair mean) * 2 C(P_L)  = V(P_L) C(P_L) (1 + rho)
Levels use INDEPENDENT seed groups, so no fictitiously free coarse base.
"""
import argparse
import csv
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

OUT = ROOT / 'output/time_coupling_pilot_2026_09_09'
# Stable integer keys; no hash(). Chosen before any run.
PROBLEM_KEY = {'shock': 101, 'gaussian': 202}
PROBLEMS = {
    # h0 = twice the existing dt0, so all three levels nest exactly in T.
    'shock':    dict(a=2., nu=.5, T=2.5, h0=.01, N=1600),
    'gaussian': dict(a=4., nu=.1, T=1.,  h0=.01, N=1600),
}
SEEDS = 64


def setup(problem, nu, N):
    if problem == 'shock':
        x = np.linspace(-10., 14., 400)
        return x, -np.tanh((x - 2.) / (2 * nu)), initialize_tanh_shock_particles(N, nu, 1., 2.)
    z = np.load(ROOT / 'output/final_prepublication_tests/'
                       'gbmc_smooth_transient/reference.npz')
    return z['x'], z['u_ref'], initialize_gaussian_gradient_particles(N)


def step_identity(x, m, v, u_left, dt, dz):
    """One mean-transport step with arrays held in IDENTITY order.

    Mirrors the production schedule exactly: transport with the velocity carried
    in, sort/reconstruct, set the next velocity to the reconstructed state, then
    diffuse.  ``dz`` is the already-scaled Brownian increment in identity order.
    """
    x = x + v * dt
    order = np.argsort(x, kind='stable')
    u_sorted = u_left + np.cumsum(m[order])
    u = np.empty_like(u_sorted)
    u[order] = u_sorted                      # reconstructed state, identity order
    return x + dz, u                         # next v is u (mean transport)


def run_single(initial, nu, dt, n_steps, rng):
    """Standalone run at one resolution; noise drawn in identity order."""
    x, m, u_left = np.array(initial[0]), np.array(initial[1]), initial[2]
    order = np.argsort(x, kind='stable')
    u_s = u_left + np.cumsum(m[order]); v = np.empty_like(u_s); v[order] = u_s
    sd = np.sqrt(2.0 * nu * dt)
    for _ in range(n_steps):
        x, v = step_identity(x, m, v, u_left, dt, sd * rng.standard_normal(len(x)))
    return x, m, u_left


def run_single_z(initial, nu, dt, n_steps, Zs):
    """Standalone run consuming a supplied unit-normal sequence (for antithetic)."""
    x, m, u_left = np.array(initial[0]), np.array(initial[1]), initial[2]
    order = np.argsort(x, kind='stable')
    u_s = u_left + np.cumsum(m[order]); v = np.empty_like(u_s); v[order] = u_s
    sd = np.sqrt(2.0 * nu * dt)
    for k in range(n_steps):
        x, v = step_identity(x, m, v, u_left, dt, sd * Zs[k])
    return x, m, u_left


def run_coupled(initial, nu, h_f, n_coarse, rng):
    """Lockstep fine (h_f) and coarse (2 h_f) runs sharing Brownian increments.

    The coarse increment is the exact sum of the two fine increments it spans,
    so both marginals are correct and the particle identity is shared."""
    xf, m, u_left = np.array(initial[0]), np.array(initial[1]), initial[2]
    xc = xf.copy()
    order = np.argsort(xf, kind='stable')
    u_s = u_left + np.cumsum(m[order]); v0 = np.empty_like(u_s); v0[order] = u_s
    vf, vc = v0.copy(), v0.copy()
    sd_f = np.sqrt(2.0 * nu * h_f)
    for _ in range(n_coarse):
        za = rng.standard_normal(len(xf))
        zb = rng.standard_normal(len(xf))
        xf, vf = step_identity(xf, m, vf, u_left, h_f, sd_f * za)
        xf, vf = step_identity(xf, m, vf, u_left, h_f, sd_f * zb)
        xc, vc = step_identity(xc, m, vc, u_left, 2.0 * h_f, sd_f * (za + zb))
    return (xf, m, u_left), (xc, m, u_left)


def field(state, x_out):
    return reconstruct_cumulative_field(state[0], state[1], state[2], x_out)


def ivar(F, dx):
    """Integrated field variance E||F - E F||_h^2 (unbiased over samples)."""
    F = np.asarray(F)
    return float(dx * np.sum(F.var(axis=0, ddof=1)))


def verify(problem):
    """Coupling and marginal checks, before any data are interpreted."""
    cfg = PROBLEMS[problem]; nu, N = cfg['nu'], cfg['N']
    x_out, ref, initial = setup(problem, nu, N)
    print(f"  [{problem}] coupling and marginal checks")
    # (a) coarse increment marginal equals sqrt(2 nu h_c)
    rng = np.random.default_rng(0); h_f = 1e-3
    za, zb = rng.standard_normal(200000), rng.standard_normal(200000)
    got = (np.sqrt(2*nu*h_f)*(za+zb)).std(); want = np.sqrt(2*nu*2*h_f)
    print(f"     coarse increment sd  {got:.6e}  vs exact {want:.6e}  "
          f"rel {abs(got/want-1):.2e}")
    # (b) coarse arm of a coupled pair matches a standalone coarse run in law
    K = 40
    a1 = np.array([field(run_coupled(initial, nu, h_f, K,
                    np.random.default_rng(1000+s))[1], x_out) for s in range(60)])
    a2 = np.array([field(run_single(initial, nu, 2*h_f, K,
                    np.random.default_rng(5000+s)), x_out) for s in range(60)])
    dx = float(x_out[1]-x_out[0])
    print(f"     coupled-coarse vs standalone-coarse: ||mean diff||_h = "
          f"{np.sqrt(dx*np.sum((a1.mean(0)-a2.mean(0))**2)):.3e}  "
          f"(pooled sd/sqrt(S) ~ {np.sqrt((ivar(a1,dx)+ivar(a2,dx))/60):.3e})")
    # (c) identity stepper vs production solver, in law
    b1 = np.array([field(run_single(initial, nu, 2*h_f, K,
                    np.random.default_rng(7000+s)), x_out) for s in range(60)])
    b2 = []
    for s in range(60):
        r = advance_rbgbmc_particles(*initial, nu, cfg['a'], 2*h_f, K,
                                     np.random.default_rng(8000+s),
                                     rng_brownian=np.random.default_rng(9000+s),
                                     conditional_mean_transport=True)
        b2.append(reconstruct_cumulative_field(r['x'], r['m'], initial[2], x_out))
    b2 = np.array(b2)
    print(f"     identity stepper vs production: ||mean diff||_h = "
          f"{np.sqrt(dx*np.sum((b1.mean(0)-b2.mean(0))**2)):.3e}  "
          f"(pooled sd/sqrt(S) ~ {np.sqrt((ivar(b1,dx)+ivar(b2,dx))/60):.3e}); "
          f"var ratio = {ivar(b1,dx)/ivar(b2,dx):.4f}")


def main(seeds=SEEDS, verify_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    src = hashlib.sha256((ROOT/'relaxation_gbmc.py').read_bytes()).hexdigest()[:16]
    print("verification\n" + "-"*72)
    for p in PROBLEMS:
        verify(p)
    if verify_only:
        return
    results = {}
    for problem, cfg in PROBLEMS.items():
        nu, N, T, h0 = cfg['nu'], cfg['N'], cfg['T'], cfg['h0']
        x_out, ref, initial = setup(problem, nu, N)
        dx = float(x_out[1] - x_out[0])
        hs = [h0, h0/2, h0/4]
        for h in hs:
            assert abs(T/h - round(T/h)) < 1e-9, "horizon must nest exactly"
        pk = PROBLEM_KEY[problem]
        print(f"\n{problem}: N={N} h={hs} T={T} seeds={seeds}\n" + "-"*72)

        # level 0, independent group
        t0 = time.perf_counter()
        P0 = np.array([field(run_single(initial, nu, hs[0], round(T/hs[0]),
                      np.random.default_rng(np.random.SeedSequence([pk,0,s]))), x_out)
                       for s in range(seeds)])
        C0 = (time.perf_counter()-t0)/seeds
        V0 = ivar(P0, dx)

        # corrections, independent groups
        Vl, Cl, Dall = [], [], []
        for l in (1, 2):
            t0 = time.perf_counter(); D = []
            for s in range(seeds):
                rng = np.random.default_rng(np.random.SeedSequence([pk, l, s]))
                fine, coarse = run_coupled(initial, nu, hs[l], round(T/hs[l-1]), rng)
                D.append(field(fine, x_out) - field(coarse, x_out))
            Cl.append((time.perf_counter()-t0)/seeds)
            D = np.array(D); Vl.append(ivar(D, dx)); Dall.append(D)
            nd = float(dx*np.sum(D.mean(0)**2) - ivar(D, dx)/seeds)   # noise-debiased
            print(f"  correction l={l}: V={Vl[-1]:.4e} C={Cl[-1]:.4f}s  "
                  f"||E D||^2 (debiased) = {nd:+.3e}")

        # single level at the finest resolution, independent group
        t0 = time.perf_counter()
        P2 = np.array([field(run_single(initial, nu, hs[2], round(T/hs[2]),
                      np.random.default_rng(np.random.SeedSequence([pk,9,s]))), x_out)
                       for s in range(seeds)])
        C2 = (time.perf_counter()-t0)/seeds
        V2 = ivar(P2, dx)

        # antithetic single level at the finest resolution: one sample = one pair
        t0 = time.perf_counter(); A, rho_n = [], []
        K2 = round(T/hs[2])
        for s in range(seeds):
            rng = np.random.default_rng(np.random.SeedSequence([pk, 8, s]))
            Z = rng.standard_normal((K2, N))
            fa = field(run_single_z(initial, nu, hs[2], K2, Z), x_out)
            fb = field(run_single_z(initial, nu, hs[2], K2, -Z), x_out)
            A.append((fa+fb)/2); rho_n.append((fa, fb))
        Canti = (time.perf_counter()-t0)/seeds
        A = np.array(A); Vanti = ivar(A, dx)
        fa = np.array([r[0] for r in rho_n]); fb = np.array([r[1] for r in rho_n])
        ca, cb = fa-fa.mean(0), fb-fb.mean(0)
        rho = float((ca*cb).sum()/np.sqrt((ca**2).sum()*(cb**2).sum()))

        K_SL = V2*C2
        K_ML = (np.sqrt(V0*C0) + sum(np.sqrt(v*c) for v, c in zip(Vl, Cl)))**2
        K_AN = Vanti*Canti
        print(f"  V0={V0:.4e} C0={C0:.4f}s | V(P2)={V2:.4e} C={C2:.4f}s | "
              f"antithetic V={Vanti:.4e} C={Canti:.4f}s rho={rho:+.4f}")
        print(f"  K_SL={K_SL:.4e}   K_ML={K_ML:.4e}   K_anti={K_AN:.4e}")
        print(f"  K_ML/K_SL={K_ML/K_SL:.3f}   K_ML/K_anti={K_ML/K_AN:.3f}   "
              f"K_anti/K_SL={K_AN/K_SL:.3f}  (<1 favours the numerator)")
        # exploratory paired outer-seed bootstrap on the variance-cost ratios;
        # costs held at their measured values so only sampling variability enters
        bs = np.random.default_rng(4242); B = 4000; r_sl, r_an = [], []
        for _ in range(B):
            i = bs.integers(0, seeds, seeds)
            v0 = ivar(P0[i], dx); v1 = ivar(Dall[0][i], dx); v2 = ivar(Dall[1][i], dx)
            vf = ivar(P2[i], dx); va = ivar(A[i], dx)
            kml = (np.sqrt(v0*C0) + np.sqrt(v1*Cl[0]) + np.sqrt(v2*Cl[1]))**2
            r_sl.append(kml/(vf*C2)); r_an.append(kml/(va*Canti))
        ci_sl = np.percentile(r_sl, [2.5, 97.5]); ci_an = np.percentile(r_an, [2.5, 97.5])
        print(f"  exploratory 95% bootstrap: K_ML/K_SL in "
              f"[{ci_sl[0]:.3f},{ci_sl[1]:.3f}]   K_ML/K_anti in "
              f"[{ci_an[0]:.3f},{ci_an[1]:.3f}]")
        np.savez_compressed(OUT/f'{problem}_fields.npz', x_out=x_out, ref=ref,
                            P0=P0, P2=P2, A=A, D1=Dall[0], D2=Dall[1],
                            fa=fa, fb=fb, seed_ids=np.arange(seeds))
        results[problem] = dict(N=N, T=T, hs=hs, seeds=seeds, nu=nu, a=cfg['a'],
                                V0=V0, C0=C0, V1=Vl[0], C1=Cl[0], V2c=Vl[1],
                                C2c=Cl[1], V_fine=V2, C_fine=C2, V_anti=Vanti,
                                C_anti=Canti, rho=rho, K_SL=K_SL, K_ML=K_ML,
                                K_anti=K_AN, source_sha=src,
                                estimator='ensemble-mean field, integrated variance',
                                replicate_unit='one outer seed; antithetic sample = 1 pair = 2 runs',
                                ci_KML_over_KSL=list(map(float, ci_sl)),
                                ci_KML_over_Kanti=list(map(float, ci_an)),
                                bootstrap='exploratory paired outer-seed, costs fixed at measured values')
    (OUT/'pilot.json').write_text(json.dumps(results, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=SEEDS)
    ap.add_argument('--verify-only', action='store_true')
    a = ap.parse_args()
    main(a.seeds, a.verify_only)
