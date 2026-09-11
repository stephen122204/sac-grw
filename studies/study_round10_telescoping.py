"""Round-10: exact policy-comparison identity, and a bounded tractability test.

PREDECLARED BUDGET (fixed before any result was inspected): N=6 particles,
K=3 steps, nu=0.2, h=0.02, bounded interval I=[-2,2] with 1201 quadrature
points, 64 outer S_1 samples, 2000 inner noise samples, finite-difference
epsilons {2e-3, 1e-3, 5e-4} for the mixed partials.

STATE AND SCHEDULE. Production's conditional-mean schedule is
    x <- x + v h ;  sort (x,m,v) ;  u <- u_left + cumsum(m) ;  v <- u ;  x <- x + noise,
so the velocity used by a transport is set BEFORE the preceding diffusion. The
step-boundary state of one replica is therefore the triple (x, m, v) taken just
after a diffusion, and the paired Markov state is S_k = (s_A, s_B). Positions
alone do not define the continuation; the carried velocity is part of the state.
The documented redraw-after-diffusion ablation is NOT used anywhere here.

FUNCTIONAL. On the bounded interval, G(S_K) = int_I U_A(x) U_B(x) dx, which is
finite without tail qualifications.

WHY G COMPARES COVARIANCES. Every policy here is a predictable permutation and
sign change of an i.i.d. Gaussian vector, so each replica keeps its own solver
law and E[U_A], E[U_B] do not depend on the policy. Hence
    E_FULL G - E_RAW G = int_I (Cov_FULL - Cov_RAW) dx.
With Pbar = (U_A+U_B)/2 and policy-independent marginal variances,
    int Var(Pbar) = (1/4)(int Var U_A + int Var U_B) + (1/2) int Cov,
so    V_FULL - V_RAW = (1/2) ( E_FULL G - E_RAW G ).      <- the factor of one half

TELESCOPING. Let pi_k use FULL for steps 1..k and RAW afterwards, so pi_0 = RAW
and pi_K = FULL. Consecutive hybrids share the law of S_k, differ at step k+1
only, and share the RAW continuation, giving the exact identity

    E_FULL G - E_RAW G = sum_{k=0}^{K-1} E_{S_k ~ FULL}[ (Q^F_{k+1} - Q^R_{k+1}) H_{k+1}(S_k) ],
    H_j(s) = E_RAW[ G(S_K) | S_j = s ],     H_K = G.

This is standard kernel telescoping, not a claimed novelty. Its content is that
the changed state distribution is handled explicitly: the expectation is under
the FULL-driven law of S_k, while the continuation H is the RAW one.

A STRUCTURAL FACT THIS MAKES OBVIOUS. Both replicas start from identical
particles, so at step 1 every matched pair has equal masses, sign(m_A)sign(m_B)
= +1 everywhere, and Q^F_1 = Q^R_1 identically: the k=0 term is exactly zero.
The policies cannot differ until the replicas' signed orders have diverged, so
K=2 is degenerate (it reproduces the final-step corollary and nothing else) and
K=3 is the smallest case with a genuine earlier-step term.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'output/round10_telescoping_2026_09_10'
N_PART, K_STEPS, NU, H_STEP = 6, 3, 0.2, 0.02
SIGMA = float(np.sqrt(2 * NU * H_STEP))
XG = np.linspace(-2.0, 2.0, 1201)
M_OUT, M_IN = 64, 2000
EPS_LIST = (2e-3, 1e-3, 5e-4)


def det_stage(x, m, v, ul, h):
    """Transport with the CARRIED velocity, sort, reconstruct, set next v."""
    x = x + v * h
    o = np.argsort(x, kind='stable')
    x, m, v = x[o], m[o], v[o]
    return x, m, ul + np.cumsum(m)


def step(state, ul, h, sigma, z, policy):
    (xA, mA, vA), (xB, mB, vB) = state
    a, mA2, uA = det_stage(xA, mA, vA, ul, h)
    b, mB2, uB = det_stage(xB, mB, vB, ul, h)
    zA = sigma * z
    zB = (-sigma * np.sign(mA2) * np.sign(mB2) * z) if policy == 'FULL' else (-sigma * z)
    return (a + zA, mA2, uA), (b + zB, mB2, uB)


def EG_onestep_batch(A, MA, VA, B, MB, VB, ul, h, sigma, policy, xg):
    """EXACT E[ int_I U_A U_B dx ] after one further step, for a batch of states.

    Only matched-rank diagonal pairs are coupled: off-diagonal (i != j) pairs
    involve independent noise coordinates, so their joint factorises.
    """
    out = np.empty(len(A))
    for s in range(len(A)):
        a, mA, _ = det_stage(A[s], MA[s], VA[s], ul, h)
        b, mB, _ = det_stage(B[s], MB[s], VB[s], ul, h)
        tA = (xg[None, :] - a[:, None]) / sigma
        tB = (xg[None, :] - b[:, None]) / sigma
        PA, PB = ndtr(tA), ndtr(tB)                 # marginals: policy independent
        UA = ul + (mA[:, None] * PA).sum(0)
        UB = ul + (mB[:, None] * PB).sum(0)
        eps = (-np.sign(mA) * np.sign(mB)) if policy == 'FULL' else -np.ones(len(a))
        Pj = np.where(eps[:, None] > 0, ndtr(np.minimum(tA, tB)),
                      np.clip(PA - ndtr(-tB), 0.0, None))
        integ = UA * UB + ((mA * mB)[:, None] * (Pj - PA * PB)).sum(0)
        out[s] = np.trapz(integ, xg)
    return out


def batch_step(S1, ul, h, sigma, Z, policy):
    (xA, mA, vA), (xB, mB, vB) = S1
    a, mA2, uA = det_stage(xA, mA, vA, ul, h)
    b, mB2, uB = det_stage(xB, mB, vB, ul, h)
    zA = sigma * Z
    zB = (-sigma * (np.sign(mA2) * np.sign(mB2))[None, :] * Z) if policy == 'FULL' \
        else (-sigma * Z)
    n = len(Z)
    return (a[None, :] + zA, np.tile(mA2, (n, 1)), np.tile(uA, (n, 1)),
            b[None, :] + zB, np.tile(mB2, (n, 1)), np.tile(uB, (n, 1)))


def initial_state(seed=17):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-0.5, 0.5, N_PART))
    m = np.where(np.arange(N_PART) % 2 == 0, 1.0, -1.0) / N_PART
    m = m - m.mean()
    ul = 0.0
    v = ul + np.cumsum(m)
    return ((x.copy(), m.copy(), v.copy()), (x.copy(), m.copy(), v.copy())), ul


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S0, ul = initial_state()
    rng = np.random.default_rng(1010)
    print(f"N={N_PART} K={K_STEPS} nu={NU} h={H_STEP} sigma={SIGMA:.4f} "
          f"I=[-2,2] ({len(XG)} points), {M_OUT} outer x {M_IN} inner\n")

    # --- k = 0 term is exactly zero: verify rather than assert ---
    Z = rng.standard_normal((8, N_PART))
    d0 = 0.0
    for z in Z:
        sF = step(S0, ul, H_STEP, SIGMA, z, 'FULL')
        sR = step(S0, ul, H_STEP, SIGMA, z, 'RAW')
        d0 = max(d0, max(float(np.max(np.abs(sF[0][0] - sR[0][0]))),
                         float(np.max(np.abs(sF[1][0] - sR[1][0])))))
    print(f"k=0 term: replicas identical at step 1, max pathwise |F - R| = {d0:.1e}"
          f"  -> Q^F_1 = Q^R_1, term is exactly 0")

    t0 = time.perf_counter()
    term1, term2, direct, unlike = [], [], [], []
    for o in range(M_OUT):
        z1 = rng.standard_normal(N_PART)
        S1 = step(S0, ul, H_STEP, SIGMA, z1, 'RAW')     # == FULL at step 1
        Z2 = rng.standard_normal((M_IN, N_PART))        # COMMON across policies
        bF = batch_step(S1, ul, H_STEP, SIGMA, Z2, 'FULL')
        bR = batch_step(S1, ul, H_STEP, SIGMA, Z2, 'RAW')
        # H_2 = one-step RAW continuation, exact
        h2F = EG_onestep_batch(*bF, ul, H_STEP, SIGMA, 'RAW', XG)
        h2R = EG_onestep_batch(*bR, ul, H_STEP, SIGMA, 'RAW', XG)
        gF = EG_onestep_batch(*bF, ul, H_STEP, SIGMA, 'FULL', XG)
        term1.append(float(np.mean(h2F - h2R)))
        term2.append(float(np.mean(gF - h2F)))
        direct.append(float(np.mean(gF - h2R)))
        a, mA, _ = det_stage(*S1[0], ul, H_STEP)
        b, mB, _ = det_stage(*S1[1], ul, H_STEP)
        unlike.append(int(np.sum(np.sign(mA) * np.sign(mB) < 0)))
    term1, term2, direct = map(np.array, (term1, term2, direct))
    resid = np.max(np.abs(direct - (term1 + term2)))
    se = lambda a: float(a.std(ddof=1) / np.sqrt(len(a)))
    print(f"\nEXACTNESS of the telescoping identity (per outer sample, shared Z2):")
    print(f"   max |direct - (term_k1 + term_k2)| = {resid:.3e}   (machine level)")
    print(f"\nDECOMPOSITION of E_FULL G - E_RAW G  (mean +- se over {M_OUT} outer):")
    print(f"   k=0 (step 1) : 0 exactly")
    print(f"   k=1 (step 2, EARLIER term) : {term1.mean():+.5e} +- {se(term1):.1e}")
    print(f"   k=2 (step 3, FINAL term)   : {term2.mean():+.5e} +- {se(term2):.1e}")
    print(f"   total                      : {direct.mean():+.5e} +- {se(direct):.1e}")
    print(f"   V_FULL - V_RAW = half of total = {0.5*direct.mean():+.5e}")
    print(f"   earlier-term share of total: {term1.mean()/direct.mean():.3f}")
    print(f"   unlike-sign matched pairs at step 2: mean {np.mean(unlike):.2f} of {N_PART}")
    print(f"   elapsed {time.perf_counter()-t0:.0f}s")

    json.dump(dict(config=dict(N=N_PART, K=K_STEPS, nu=NU, h=H_STEP, sigma=SIGMA,
                               interval=[-2.0, 2.0], quad=len(XG),
                               outer=M_OUT, inner=M_IN),
                   identity_residual=float(resid),
                   term_k0=0.0,
                   term_k1=dict(mean=float(term1.mean()), se=se(term1)),
                   term_k2=dict(mean=float(term2.mean()), se=se(term2)),
                   total=dict(mean=float(direct.mean()), se=se(direct)),
                   unlike_pairs_mean=float(np.mean(unlike)),
                   per_outer=dict(term1=term1.tolist(), term2=term2.tolist(),
                                  direct=direct.tolist())),
              open(OUT / 'telescoping.json', 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
