"""Round-10 part 3b: is the candidate statistic well defined?

The per-pair finite differences of phi split into three behaviours: some are
stable in eps, some collapse to ~1e-13 once eps is small, and some GROW like
1/eps. The last pattern is a gradient kink, not a second derivative: phi is
continuous but its gradient jumps where two particles exchange order, because
the per-particle reconstructed velocity carried into the next transport is
reassigned there. That is the same mechanism as the round-7 counterexample.

So the naive statistic 2 s^2 sum d2phi/da db is NOT well defined in the eps -> 0
limit, and a prediction built from a median over a few eps values is an artefact
of the eps grid. This script quantifies that, then tests a statistic that needs
no limit:

  FINITE-SCALE MIXED SENSITIVITY, evaluated at the physical noise scale s,
      D(s) = sum_{i in unlike} [ phi(a+s e_i, b+s e_i) - phi(a+s e_i, b-s e_i)
                               - phi(a-s e_i, b+s e_i) + phi(a-s e_i, b-s e_i) ] / 2,
  i.e. the two-point Gauss-Hermite analogue of the Gaussian expectation
  difference, costing four continuation evaluations per unlike pair and assuming
  no smoothness at all.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.study_round10_telescoping import (det_stage, step, initial_state,
                                               SIGMA, H_STEP, N_PART)
from studies.study_round10_tractability import phi

OUT = ROOT / 'output/round10_telescoping_2026_09_10'
M_OUT, M_IN = 16, 2000
EPS_GRID = (4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4)


def mixed_sum(a, b, mA, uA, mB, uB, ul, unlike, eps):
    tot = 0.0
    for i in unlike:
        pa, pb = [], []
        for sa in (+1, -1):
            for sb in (+1, -1):
                aa = a.copy(); aa[i] += sa * eps
                bb = b.copy(); bb[i] += sb * eps
                pa.append(aa); pb.append(bb)
        v = phi(np.array(pa), np.array(pb), mA, uA, mB, uB, ul, SIGMA)
        tot += (v[0] - v[1] - v[2] + v[3]) / (4 * eps * eps)
    return tot


def finite_scale(a, b, mA, uA, mB, uB, ul, unlike, s):
    """Two-point statistic at the physical noise scale; no limit, no smoothness."""
    tot = 0.0
    for i in unlike:
        pa, pb = [], []
        for sa in (+1, -1):
            for sb in (+1, -1):
                aa = a.copy(); aa[i] += sa * s
                bb = b.copy(); bb[i] += sb * s
                pa.append(aa); pb.append(bb)
        v = phi(np.array(pa), np.array(pb), mA, uA, mB, uB, ul, SIGMA)
        tot += (v[0] - v[1] - v[2] + v[3]) / 2.0
    return tot


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S0, ul = initial_state()
    rng = np.random.default_rng(2020)      # same stream as the tractability run
    t0 = time.perf_counter()
    naive = {e: [] for e in EPS_GRID}
    fs, meas, se = [], [], []
    for o in range(M_OUT):
        z1 = rng.standard_normal(N_PART)
        S1 = step(S0, ul, H_STEP, SIGMA, z1, 'RAW')
        a, mA, uA = det_stage(*S1[0], ul, H_STEP)
        b, mB, uB = det_stage(*S1[1], ul, H_STEP)
        unlike = np.flatnonzero(np.sign(mA) * np.sign(mB) < 0)
        if len(unlike) == 0:
            continue
        for e in EPS_GRID:
            naive[e].append(2 * SIGMA * SIGMA
                            * mixed_sum(a, b, mA, uA, mB, uB, ul, unlike, e))
        fs.append(finite_scale(a, b, mA, uA, mB, uB, ul, unlike, SIGMA))
        Z2 = rng.standard_normal((M_IN, N_PART))
        epsv = -np.sign(mA) * np.sign(mB)
        d = (phi(a[None, :] + SIGMA * Z2, b[None, :] + epsv[None, :] * SIGMA * Z2,
                 mA, uA, mB, uB, ul, SIGMA)
             - phi(a[None, :] + SIGMA * Z2, b[None, :] - SIGMA * Z2,
                   mA, uA, mB, uB, ul, SIGMA))
        meas.append(float(d.mean())); se.append(float(d.std(ddof=1) / np.sqrt(M_IN)))
    meas = np.array(meas); fs = np.array(fs)
    M = meas.mean()
    print(f"measured earlier-step difference at s = sigma: {M:+.5e} "
          f"+- {meas.std(ddof=1)/np.sqrt(len(meas)):.1e}\n")
    print("A. NAIVE second-derivative statistic 2 s^2 sum d2phi/da db, by eps:")
    print(f"   {'eps':>10} {'prediction':>14} {'ratio to measured':>19}")
    for e in EPS_GRID:
        p = float(np.mean(naive[e]))
        print(f"   {e:10.1e} {p:14.5e} {p/M:19.3f}")
    vals = [float(np.mean(naive[e])) for e in EPS_GRID]
    print(f"   -> prediction varies by a factor "
          f"{max(vals)/min(vals) if min(vals) else float('inf'):.2f} across eps: "
          f"NOT a well-defined derivative")
    print(f"\nB. FINITE-SCALE statistic at s = sigma (no limit, no smoothness):")
    P = float(fs.mean())
    print(f"   prediction {P:+.5e}   ratio to measured {P/M:.3f}")
    r = fs / meas
    print(f"   per-state ratio: median {np.median(r):.3f}, "
          f"IQR [{np.percentile(r,25):.3f}, {np.percentile(r,75):.3f}]")
    print(f"   sign agreement per state: "
          f"{int(np.sum(np.sign(fs)==np.sign(meas)))}/{len(fs)}")
    print(f"\nelapsed {time.perf_counter()-t0:.0f}s")
    json.dump(dict(measured_mean=float(M),
                   naive_by_eps={str(e): float(np.mean(naive[e])) for e in EPS_GRID},
                   naive_spread_factor=float(max(vals) / min(vals)) if min(vals) else None,
                   finite_scale_mean=float(P), finite_scale_ratio=float(P / M),
                   per_state_ratio_median=float(np.median(r)),
                   sign_agreement=int(np.sum(np.sign(fs) == np.sign(meas))),
                   n_states=len(fs)),
              open(OUT / 'epsilon_audit.json', 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
