"""Round-11 part 1: independent check of the continuation audit.

MY OWN DERIVATION (not Codex's route, so agreement is a real cross-check).

For step fields with ZERO total signed mass and zero background,
U_A(x) = sum_i m_Ai 1{X_Ai <= x} tends to 0 at both ends, and for any constant c
the zero-sum property makes sum_ij m_Ai m_Bj c vanish. Writing
int 1{u<=x} 1{v<=x} dx = int 1{x >= max(u,v)} dx and dropping the (cancelling)
constant gives

    int_R U_A U_B dx = - sum_ij m_Ai m_Bj max(X_Ai, X_Bj)
                     = - (1/2) sum_ij m_Ai m_Bj |X_Ai - X_Bj|,

the second step using max = (u+v+|u-v|)/2 and zero-sum again. Taking
expectations over one further diffusion, X_Ai - X_Bj is Gaussian with mean
a_i - b_j and standard deviation s_ij: sqrt(2) sigma off the matched diagonal
(independent noise coordinates), 2 sigma on it under RAW, and 0 on it under FULL
at unlike-sign pairs (the increments cancel exactly). Hence

    H(state) = - (1/2) sum_ij m_Ai m_Bj g_{s_ij}(a_i - b_j),
    g_s(d) = E|d + s Z| = |d| (2 Phi(|d|/s) - 1) + 2 s phi(|d|/s),   g_0(d) = |d|.

NO SPATIAL GRID APPEARS. The round-10 driver instead evaluated exact Gaussian
probabilities at 1201 fixed probes and applied the trapezoidal rule, and its RAW
joint probability carries a positive-part kink whose location moves with the
perturbed positions. Differentiating that quadrature at stencils comparable to
the probe spacing (dx = 4/1200 = 3.3e-3) exposes probe-scale quadrature error,
not a property of the integrated continuation.

ANALYTIC MIXED PARTIAL, also derived here. Perturbing alpha_i moves a'_{rA(i)}
and perturbing beta_i moves b'_{rB(i)}, where rA, rB are the continuation's sort
permutations, so only the (rA(i), rB(i)) term survives:

    d2H/dalpha_i dbeta_i = (1/2) m_A'{rA} m_B'{rB} g''_{s}(d),  g''_s(d) = (2/s) phi(d/s),

with s = 2 sigma when rA(i) = rB(i) and sqrt(2) sigma otherwise. Note the
off-diagonal case is NOT zero; round 10 implicitly assumed only the diagonal.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.study_round10_telescoping import (det_stage, step, initial_state,
                                               SIGMA, H_STEP, N_PART, XG)
from studies.study_round10_tractability import phi as phi_grid

OUT = ROOT / 'output/round11_continuation_2026_09_10'
EPS_GRID = (4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5)
M_OUT, M_IN = 16, 2000          # same shape as the round-10 epsilon audit


def g(s, d):
    d = np.abs(d)
    if np.isscalar(s):
        if s <= 0:
            return d
        return d * (2 * ndtr(d / s) - 1) + 2 * s * norm.pdf(d / s)
    out = np.where(s > 0,
                   d * (2 * ndtr(d / np.where(s > 0, s, 1)) - 1)
                   + 2 * s * norm.pdf(d / np.where(s > 0, s, 1)), d)
    return out


def H_exact(alpha, beta, mA0, uA, mB0, uB, ul, sigma, policy='RAW'):
    """Whole-line E[int U_A U_B dx] after one further step. No spatial grid."""
    a, mA, _ = det_stage(alpha, mA0, uA, ul, H_STEP)
    b, mB, _ = det_stage(beta, mB0, uB, ul, H_STEP)
    D = a[:, None] - b[None, :]
    S = np.full(D.shape, np.sqrt(2.0) * sigma)
    diag = np.arange(len(a))
    if policy == 'FULL':
        unlike = np.sign(mA) * np.sign(mB) < 0
        S[diag, diag] = np.where(unlike, 0.0, 2.0 * sigma)
    else:
        S[diag, diag] = 2.0 * sigma
    return float(-0.5 * np.sum(np.outer(mA, mB) * g(S, D)))


def H_exact_mixed(alpha, beta, mA0, uA, mB0, uB, ul, sigma, pairs):
    """Closed-form sum of d2H/dalpha_i dbeta_i over the given index pairs."""
    a, mA, _ = det_stage(alpha, mA0, uA, ul, H_STEP)
    b, mB, _ = det_stage(beta, mB0, uB, ul, H_STEP)
    rA = np.argsort(np.argsort(alpha + uA * H_STEP, kind='stable'), kind='stable')
    rB = np.argsort(np.argsort(beta + uB * H_STEP, kind='stable'), kind='stable')
    tot = 0.0
    for i in pairs:
        k, l = rA[i], rB[i]
        s = 2.0 * sigma if k == l else np.sqrt(2.0) * sigma
        d = a[k] - b[l]
        tot += 0.5 * mA[k] * mB[l] * (2.0 / s) * norm.pdf(d / s)
    return tot


def bounded_adaptive(alpha, beta, mA0, uA, mB0, uB, ul, sigma, L=2.0):
    """Adaptive integration of E[U_A U_B] on [-L, L], split at the kinks."""
    a, mA, _ = det_stage(alpha, mA0, uA, ul, H_STEP)
    b, mB, _ = det_stage(beta, mB0, uB, ul, H_STEP)

    def integrand(x):
        tA = (x - a) / sigma
        tB = (x - b) / sigma
        PA, PB = ndtr(tA), ndtr(tB)
        UA = ul + np.sum(mA * PA)
        UB = ul + np.sum(mB * PB)
        Pj = np.clip(PA - ndtr(-tB), 0.0, None)
        return UA * UB + np.sum(mA * mB * (Pj - PA * PB))

    pts = sorted(set(np.clip((a + b) / 2.0, -L, L).tolist()))
    tot, err = 0.0, 0.0
    edges = [-L] + pts + [L]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            v, e = quad(integrand, lo, hi, limit=200, epsabs=1e-14, epsrel=1e-12)
            tot += v; err += e
    return tot, err


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S0, ul = initial_state()
    rng = np.random.default_rng(2020)      # SAME stream as the round-10 audit
    states = []
    for o in range(M_OUT):
        z1 = rng.standard_normal(N_PART)
        S1 = step(S0, ul, H_STEP, SIGMA, z1, 'RAW')
        a, mA, uA = det_stage(*S1[0], ul, H_STEP)
        b, mB, uB = det_stage(*S1[1], ul, H_STEP)
        unlike = np.flatnonzero(np.sign(mA) * np.sign(mB) < 0)
        rng.standard_normal((M_IN, N_PART))      # consume, as round 10 did
        if len(unlike):
            states.append((a, mA, uA, b, mB, uB, unlike))
    print(f"reproduced {len(states)} outer states from the round-10 stream\n")

    # --- cross-check the two continuation evaluations on one state ---
    a, mA, uA, b, mB, uB, unlike = states[0]
    hx = H_exact(a, b, mA, uA, mB, uB, ul, SIGMA)
    hb, err = bounded_adaptive(a, b, mA, uA, mB, uB, ul, SIGMA)
    hg = float(phi_grid(a[None, :], b[None, :], mA, uA, mB, uB, ul, SIGMA)[0])
    print("continuation value on representative state 0:")
    print(f"   whole-line closed form        {hx:.16e}")
    print(f"   bounded adaptive on [-2,2]    {hb:.16e}  (quad err {err:.1e})")
    print(f"   |difference|                  {abs(hx-hb):.2e}")
    print(f"   round-10 fixed-grid trapezoid {hg:.16e}   error {abs(hg-hx):.2e}\n")

    # --- finite differences of both, plus the analytic mixed partial ---
    print(f"{'eps':>10} {'grid FD':>15} {'exact FD':>15} {'analytic':>15} {'sortchg':>8}")
    rows = []
    for eps in EPS_GRID:
        pg = pe = 0.0
        nsort = 0
        for (a, mA, uA, b, mB, uB, unlike) in states:
            base_rA = np.argsort(a + uA * H_STEP, kind='stable')
            base_rB = np.argsort(b + uB * H_STEP, kind='stable')
            sg = se = 0.0
            for i in unlike:
                pa, pb = [], []
                for sa in (+1, -1):
                    for sb in (+1, -1):
                        aa = a.copy(); aa[i] += sa * eps
                        bb = b.copy(); bb[i] += sb * eps
                        pa.append(aa); pb.append(bb)
                        if not (np.array_equal(np.argsort(aa + uA * H_STEP, kind='stable'), base_rA)
                                and np.array_equal(np.argsort(bb + uB * H_STEP, kind='stable'), base_rB)):
                            nsort += 1
                v = phi_grid(np.array(pa), np.array(pb), mA, uA, mB, uB, ul, SIGMA)
                sg += (v[0] - v[1] - v[2] + v[3]) / (4 * eps * eps)
                w = [H_exact(pa[k], pb[k], mA, uA, mB, uB, ul, SIGMA) for k in range(4)]
                se += (w[0] - w[1] - w[2] + w[3]) / (4 * eps * eps)
            pg += 2 * SIGMA * SIGMA * sg
            pe += 2 * SIGMA * SIGMA * se
        an = sum(2 * SIGMA * SIGMA
                 * H_exact_mixed(a, b, mA, uA, mB, uB, ul, SIGMA, unlike)
                 for (a, mA, uA, b, mB, uB, unlike) in states)
        pg /= len(states); pe /= len(states); an /= len(states)
        print(f"{eps:10.1e} {pg:15.8f} {pe:15.8f} {an:15.8f} {nsort:8d}")
        rows.append(dict(eps=eps, grid_fd=pg, exact_fd=pe, analytic=an,
                         sort_changes=nsort))
    print(f"\nround-10 archived physical-scale measured mean: -0.00203688")
    print(f"limiting exact prediction:                     {rows[-1]['exact_fd']:.8f}")
    json.dump(dict(state0=dict(whole_line=hx, bounded_adaptive=hb,
                               quad_err=err, grid=hg, grid_error=abs(hg - hx)),
                   rows=rows, n_states=len(states),
                   measured_physical_scale=-0.00203688),
              open(OUT / 'continuation_check.json', 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
