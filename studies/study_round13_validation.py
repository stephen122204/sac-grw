"""Round-13: independent validation of the round-12 work-to-target prediction.

SELECTION AND EVALUATION ARE SEPARATED. The plans below are FROZEN from the
round-12 pilot estimates (seed key 1) and are not re-fitted here. Evaluation uses
a fresh, disjoint seed key (1313). The round-12 numbers were oracle plug-in
predictions: bias, variance and cost were estimated, the cheapest cell and
replicate count were selected from those same estimates, and no plan was ever
executed. This run executes them.

TARGET. tol = 5e-6 is an EXPECTED INTEGRATED SQUARED ERROR of the reconstructed
field over the output window [-5, 5]:  int (u_hat - u_ref)^2 dx <= tol. Its
square root, 2.24e-3, is not an RMS field error; dividing by the window length
gives mean square 5e-7 and RMS 7.07e-4, against a field of size max|u| ~ 0.55.
The claim is about the EXPECTED error of the plan, not a probability that any
single realisation lands below tol.

FROZEN PLANS (round-12 pilot, cell N=6400, h=0.005):
    SINGLE  74 independent runs      RAW  18 pairs      FULL  3 pairs

MATCHED TIMING BOUNDARY. All three arms are timed over the SAME requested output:
advance + reconstruct the field on the output grid. SINGLE uses one arm of the
identical lean loop, so no implementation-overhead asymmetry remains; the
production-stepper cost is reported separately for reference.

BUDGET. 32 disjoint blocks per arm: 96 FULL pairs, 576 RAW pairs, 2368 single
runs. Estimated ~5 minutes.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import advance_rbgbmc_particles, reconstruct_cumulative_field
from studies.study_round06_cubic import advance_pair_flux, XG
from studies.round06_spectral_reference import solve, evaluate, FPRIME
from studies import twopulse_reference as TP

OUT = ROOT / 'output/round13_validation_2026_09_10'
NU, T, A_REL = 0.1, 1.0, 2.0
N_PART, H_STEP, TOL = 6400, 0.005, 5e-6
PLAN = {'SINGLE': 74, 'RAW': 18, 'FULL': 3}
BLOCKS, EVAL_KEY = 32, 1313


def advance_single_flux(initial, nu, dt, K, rng, flux='burgers'):
    """Exactly one arm of advance_pair_flux: identical loop, identical cost basis."""
    fp = FPRIME[flux]
    x0 = np.asarray(initial[0], float)
    m = np.asarray(initial[1], float)
    ul = float(initial[2])
    o = np.argsort(x0, kind='stable')
    x, m = x0[o].copy(), m[o].copy()
    n = len(x)
    sd = np.sqrt(2.0 * nu * dt)
    v = fp(ul + np.cumsum(m))
    for _ in range(K):
        x = x + v * dt
        o = np.argsort(x, kind='stable'); x, m = x[o], m[o]
        v = fp(ul + np.cumsum(m))
        x = x + sd * rng.standard_normal(n)
    return x, m, ul


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, uh, kk = solve('burgers', T, NU, M=2048, dt=1e-4)
    ref = evaluate(uh, kk, XG, 2048)
    dx = float(XG[1] - XG[0])
    x0, m0, ul, _ = TP.initialize(N_PART)
    K = round(T / H_STEP)
    t_start = time.perf_counter()
    print(f"target tol = {TOL:.1e} expected integrated squared error on [-5,5]")
    print(f"  sqrt(tol) = {np.sqrt(TOL):.3e} (NOT an RMS field error); "
          f"window RMS = {np.sqrt(TOL/10):.3e}; max|u_ref| = {np.max(np.abs(ref)):.3f}")
    print(f"frozen plans (round-12 pilot, key 1): {PLAN}")
    print(f"evaluation: {BLOCKS} disjoint blocks, fresh key {EVAL_KEY}\n")

    res, costs = {}, {}
    for arm, B in PLAN.items():
        errs, secs = [], []
        for b in range(BLOCKS):
            acc = np.zeros(len(XG))
            t0 = time.perf_counter()
            for r in range(B):
                rng = np.random.default_rng(
                    np.random.SeedSequence([EVAL_KEY, hash(arm) % 9973, b, r]))
                if arm == 'SINGLE':
                    x, m, u_l = advance_single_flux((x0, m0, ul), NU, H_STEP, K, rng)
                    acc += reconstruct_cumulative_field(x, m, u_l, XG)
                else:
                    A, Bp, u_l = advance_pair_flux((x0, m0, ul), NU, H_STEP, K,
                                                   rng, arm, 'burgers')
                    fa = reconstruct_cumulative_field(A[0], A[1], u_l, XG)
                    fb = reconstruct_cumulative_field(Bp[0], Bp[1], u_l, XG)
                    acc += 0.5 * (fa + fb)
            secs.append(time.perf_counter() - t0)
            est = acc / B
            errs.append(float(dx * np.sum((est - ref) ** 2)))
        errs = np.array(errs)
        res[arm] = errs
        costs[arm] = float(np.median(secs))
        se = float(errs.std(ddof=1) / np.sqrt(BLOCKS))
        print(f"  {arm:>6} B={B:3d}: realised expected error = {errs.mean():.4e} "
              f"+- {se:.2e}   target {TOL:.1e}   "
              f"{'MET' if errs.mean() + 1.96*se < TOL else ('met (point)' if errs.mean() < TOL else 'NOT met')}"
              f"   blocks below tol {int(np.sum(errs < TOL))}/{BLOCKS}"
              f"   measured work {costs[arm]:.3f}s")

    # production-stepper reference cost, for the record
    tp = []
    for r in range(8):
        t0 = time.perf_counter()
        o = advance_rbgbmc_particles(x0, m0, ul, NU, A_REL, H_STEP, K,
                                     np.random.default_rng(0),
                                     rng_brownian=np.random.default_rng(900 + r),
                                     conditional_mean_transport=True)
        reconstruct_cumulative_field(o['x'], o['m'], ul, XG)
        tp.append(time.perf_counter() - t0)
    print(f"\n  production stepper, single run + reconstruction: "
          f"{np.median(tp):.4f}s vs lean single arm "
          f"{costs['SINGLE']/PLAN['SINGLE']:.4f}s "
          f"(ratio {np.median(tp)/(costs['SINGLE']/PLAN['SINGLE']):.2f}x)")
    print(f"\n  MEASURED work ratios at matched output boundary:")
    print(f"    FULL/RAW    {costs['FULL']/costs['RAW']:.3f}")
    print(f"    FULL/SINGLE {costs['FULL']/costs['SINGLE']:.3f}")
    print(f"    RAW/SINGLE  {costs['RAW']/costs['SINGLE']:.3f}")
    print(f"\nelapsed {time.perf_counter()-t_start:.0f}s")
    json.dump(dict(tol=TOL, tol_is='expected integrated squared error on [-5,5]',
                   sqrt_tol=float(np.sqrt(TOL)),
                   window_rms=float(np.sqrt(TOL / 10)),
                   plan=PLAN, blocks=BLOCKS, eval_key=EVAL_KEY,
                   cell=dict(N=N_PART, h=H_STEP, nu=NU, T=T),
                   realised={a: dict(mean=float(v.mean()),
                                     se=float(v.std(ddof=1) / np.sqrt(BLOCKS)),
                                     blocks_below_tol=int(np.sum(v < TOL)),
                                     per_block=v.tolist()) for a, v in res.items()},
                   measured_work_s=costs,
                   production_single_s=float(np.median(tp)),
                   note='selection frozen from round-12 pilot (key 1); evaluation '
                        'on disjoint key 1313. Claim is about expected error, not '
                        'a probability that a realisation lands below tol.'),
              open(OUT / 'validation.json', 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
