"""Round-10 part 3: is the earlier-step term expressible through an accessible,
non-circular statistic?

THE CANDIDATE. Given the paired state S_k, both transitions apply the SAME
deterministic transport/sort/velocity stage and then differ only in the joint
law of the diffusion: writing (a, b) for the post-transport paired positions,
RAW uses (a + sZ, b - sZ) and FULL uses (a + sZ, b + eps sZ) with
eps_i = -sign(m_A,i) sign(m_B,i), so the two coincide except at UNLIKE-SIGN
matched pairs, where the perturbation direction flips from (+,-) to (+,+).

For a continuation value phi that is smooth near (a, b), second order in s gives

    E[phi(a+sZ, b+sZ)] - E[phi(a+sZ, b-sZ)]  =  2 s^2 sum_{i in unlike} d2phi/da_i db_i
                                                 + O(s^4).

So the CANDIDATE ACCESSIBLE STATISTIC is the MIXED SENSITIVITY of the RAW
continuation value at unlike-sign matched pairs. This is not circular: it is a
property of the RAW dynamics alone, it does not presuppose the sign of the
terminal covariance advantage, and it is separately computable.

WHY SMOOTHNESS IS NOT AUTOMATIC. phi contains an intermediate sort. When two
particles exchange order their per-particle reconstructed velocities swap, and
that velocity assignment is what the next transport uses. The remaining Gaussian
stages smooth the terminal field, but they do not obviously smooth a velocity
reassignment that happens BEFORE them. That is the same mechanism as the
round-7 counterexample, so whether the expansion is valid is exactly the
question, and it is settled here numerically rather than assumed.

TEST. Vary ONLY the step-2 noise scale s2 (the expansion parameter), holding the
step-3 scale fixed inside the continuation, and compare the measured difference
with 2 s2^2 sum d2phi/da db. If phi were C^4 near (a,b) the ratio would tend to 1
as s2 -> 0. Predeclared: 16 outer states, 2000 paired inner samples, s2 in
{sigma, sigma/2, sigma/4, sigma/8}, finite-difference eps in {2e-3,1e-3,5e-4}.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.study_round10_telescoping import (det_stage, step, initial_state,
                                               SIGMA, H_STEP, N_PART, XG)

OUT = ROOT / 'output/round10_telescoping_2026_09_10'
M_OUT, M_IN = 16, 2000
EPS_LIST = (2e-3, 1e-3, 5e-4)
CHUNK = 512


def EG_vec(A, MA, VA, B, MB, VB, ul, h, sigma, policy, xg):
    """Vectorised exact one-step E[int_I U_A U_B dx] over a batch of states."""
    n = len(A)
    out = np.empty(n)
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        a, mA, b, mB = A[lo:hi], MA[lo:hi], B[lo:hi], MB[lo:hi]
        va, vb = VA[lo:hi], VB[lo:hi]
        xa = a + va * h
        o = np.argsort(xa, axis=1, kind='stable')
        xa = np.take_along_axis(xa, o, 1); ma = np.take_along_axis(mA, o, 1)
        xb = b + vb * h
        o = np.argsort(xb, axis=1, kind='stable')
        xb = np.take_along_axis(xb, o, 1); mb = np.take_along_axis(mB, o, 1)
        tA = (xg[None, None, :] - xa[:, :, None]) / sigma
        tB = (xg[None, None, :] - xb[:, :, None]) / sigma
        PA, PB = ndtr(tA), ndtr(tB)
        UA = ul + (ma[:, :, None] * PA).sum(1)
        UB = ul + (mb[:, :, None] * PB).sum(1)
        eps = (-np.sign(ma) * np.sign(mb)) if policy == 'FULL' else -np.ones_like(ma)
        Pj = np.where(eps[:, :, None] > 0, ndtr(np.minimum(tA, tB)),
                      np.clip(PA - ndtr(-tB), 0.0, None))
        integ = UA * UB + ((ma * mb)[:, :, None] * (Pj - PA * PB)).sum(1)
        out[lo:hi] = np.trapz(integ, xg, axis=1)
    return out


def phi(alpha, beta, mA, uA, mB, uB, ul, sigma3):
    """RAW continuation value as a function of the paired post-noise positions."""
    n = len(alpha)
    return EG_vec(alpha, np.tile(mA, (n, 1)), np.tile(uA, (n, 1)),
                  beta, np.tile(mB, (n, 1)), np.tile(uB, (n, 1)),
                  ul, H_STEP, sigma3, 'RAW', XG)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S0, ul = initial_state()
    rng = np.random.default_rng(2020)
    scales = [SIGMA, SIGMA / 2, SIGMA / 4, SIGMA / 8]
    rows = []
    t0 = time.perf_counter()
    print(f"expansion test: measured difference vs 2 s2^2 * sum_unlike d2phi/da db")
    print(f"(step-3 scale held at sigma={SIGMA:.4f}; only s2 varies)\n")
    print(f"{'s2/sigma':>9} {'measured':>13} {'se':>10} {'predicted':>13} "
          f"{'ratio':>8} {'FD spread':>10}")
    agg = {s: dict(meas=[], pred=[], se=[]) for s in range(len(scales))}
    for o in range(M_OUT):
        z1 = rng.standard_normal(N_PART)
        S1 = step(S0, ul, H_STEP, SIGMA, z1, 'RAW')
        a, mA, uA = det_stage(*S1[0], ul, H_STEP)
        b, mB, uB = det_stage(*S1[1], ul, H_STEP)
        unlike = np.flatnonzero(np.sign(mA) * np.sign(mB) < 0)
        if len(unlike) == 0:
            continue
        # mixed partials of phi at (a, b), finite differences, several eps
        mixed = {}
        for eps in EPS_LIST:
            tot = 0.0
            for i in unlike:
                pts_a, pts_b = [], []
                for sa in (+1, -1):
                    for sb in (+1, -1):
                        aa = a.copy(); aa[i] += sa * eps
                        bb = b.copy(); bb[i] += sb * eps
                        pts_a.append(aa); pts_b.append(bb)
                v = phi(np.array(pts_a), np.array(pts_b), mA, uA, mB, uB, ul, SIGMA)
                tot += (v[0] - v[1] - v[2] + v[3]) / (4 * eps * eps)
            mixed[eps] = tot
        med = float(np.median(list(mixed.values())))
        spread = float((max(mixed.values()) - min(mixed.values())) / abs(med)) \
            if med != 0 else np.nan
        Z2 = rng.standard_normal((M_IN, N_PART))
        for si, s2 in enumerate(scales):
            zA = s2 * Z2
            epsv = -np.sign(mA) * np.sign(mB)
            aF, bF = a[None, :] + zA, b[None, :] + epsv[None, :] * s2 * Z2
            aR, bR = a[None, :] + zA, b[None, :] - s2 * Z2
            d = (phi(aF, bF, mA, uA, mB, uB, ul, SIGMA)
                 - phi(aR, bR, mA, uA, mB, uB, ul, SIGMA))
            agg[si]['meas'].append(float(d.mean()))
            agg[si]['se'].append(float(d.std(ddof=1) / np.sqrt(M_IN)))
            agg[si]['pred'].append(2 * s2 * s2 * med)
        rows.append(dict(outer=o, n_unlike=int(len(unlike)),
                         mixed_median=med, fd_spread=spread))
    for si, s2 in enumerate(scales):
        me = float(np.mean(agg[si]['meas'])); pr = float(np.mean(agg[si]['pred']))
        s = float(np.std(agg[si]['meas'], ddof=1) / np.sqrt(len(agg[si]['meas'])))
        fd = float(np.mean([r['fd_spread'] for r in rows]))
        print(f"{s2/SIGMA:9.3f} {me:13.5e} {s:10.2e} {pr:13.5e} "
              f"{me/pr if pr else np.nan:8.3f} {fd:10.3f}")
    print(f"\nmean unlike-sign matched pairs: "
          f"{np.mean([r['n_unlike'] for r in rows]):.2f} of {N_PART}")
    print(f"finite-difference spread across eps (relative): "
          f"{np.mean([r['fd_spread'] for r in rows]):.3f}")
    print(f"elapsed {time.perf_counter()-t0:.0f}s")
    json.dump(dict(scales=[float(s) for s in scales], sigma=SIGMA,
                   outer=M_OUT, inner=M_IN, eps_list=list(EPS_LIST),
                   per_scale=[dict(s2=float(scales[i]),
                                   measured_mean=float(np.mean(agg[i]['meas'])),
                                   measured_se=float(np.std(agg[i]['meas'], ddof=1)
                                                     / np.sqrt(len(agg[i]['meas']))),
                                   predicted_mean=float(np.mean(agg[i]['pred'])))
                              for i in range(len(scales))],
                   per_outer=rows),
              open(OUT / 'tractability.json', 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
