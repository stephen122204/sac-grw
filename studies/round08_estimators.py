"""Round-8 estimator toolkit: instrumented paired solver, identity-tracked heat
control, and final-step conditional expectation. Shared by the study driver.

IDENTITY TRACKING. The solver sorts by position every step, so array index does
NOT identify a particle. A stable identity vector is permuted alongside (x, m)
and the drawn displacement is accumulated as dW[identity] += z, never by rank
label. Identity i keeps mass m0[i] and initial position x0[i] throughout.

WHY THE CONTROL'S MEAN IS EXACT. At step k the permutation and any sign
multipliers are functions of the pre-diffusion configuration, hence measurable
with respect to F_{k-1}; the noise z_k is iid N(0,1) independent of F_{k-1}. A
predictable permutation of an iid Gaussian vector is iid Gaussian, and a
predictable +-1 multiplier preserves N(0, sd^2). So conditionally on F_{k-1}
each identity's increment is N(0, sd^2) with DETERMINISTIC variance. If
X_k | F_{k-1} ~ N(0, sd^2) with sd^2 deterministic then
E[e^{iuS_K}] = E[e^{iuS_{K-1}}] e^{-u^2 sd^2/2}, so by induction the accumulated
displacement of every identity is exactly N(0, K sd^2) = N(0, 2 nu T). Only
marginals are needed for the mean of a step field, hence

    mu_H(x) = u_left + sum_i m_i Phi( (x - x_i(0)) / sqrt(2 nu T) )

is exact for the SAME DISCRETE initial particles. It is not the continuum
profile and carries no PDE information.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import reconstruct_cumulative_field
from studies.round06_spectral_reference import FPRIME


def advance_pair_instrumented(initial, nu, dt, K, rng, policy, flux,
                              track=True, transport=True):
    """Paired inclusive-schedule update with identity tracking and pre-final
    state capture. With track=False and transport=True the trajectories are
    bit-identical to studies.study_round06_cubic.advance_pair_flux."""
    fp = FPRIME[flux]
    x0 = np.asarray(initial[0], float)
    m0 = np.asarray(initial[1], float)
    ul = float(initial[2])
    n = len(x0)
    o = np.argsort(x0, kind='stable')
    xA, mA, iA = x0[o].copy(), m0[o].copy(), o.copy()
    xB, mB, iB = xA.copy(), mA.copy(), iA.copy()
    sd = np.sqrt(2.0 * nu * dt)
    zero = np.zeros(n)
    vA = fp(ul + np.cumsum(mA)) if transport else zero
    vB = vA.copy()
    dWA = np.zeros(n) if track else None
    dWB = np.zeros(n) if track else None
    pre = {}
    for k in range(K):
        xA = xA + vA * dt
        o = np.argsort(xA, kind='stable'); xA, mA, iA = xA[o], mA[o], iA[o]
        vA = fp(ul + np.cumsum(mA)) if transport else zero
        xB = xB + vB * dt
        o = np.argsort(xB, kind='stable'); xB, mB, iB = xB[o], mB[o], iB[o]
        vB = fp(ul + np.cumsum(mB)) if transport else zero
        if k == K - 1:                       # state entering the final diffusion
            pre = dict(YA=xA.copy(), pre_mA=mA.copy(), YB=xB.copy(), pre_mB=mB.copy())
        z = rng.standard_normal(n)
        if policy == 'WITHIN':
            sA, sB = np.sign(mA), np.sign(mB)
            zB = np.empty(n)
            for sgn in (1.0, -1.0):
                ia = np.flatnonzero(sA == sgn); ib = np.flatnonzero(sB == sgn)
                q = min(len(ia), len(ib))
                zB[ib[:q]] = -z[ia[:q]]
                if len(ib) > q:
                    zB[ib[q:]] = rng.standard_normal(len(ib) - q)
            zA, zB = sd * z, sd * zB
        elif policy == 'FULL':
            zA = sd * z
            zB = -sd * (np.sign(mA) * np.sign(mB)) * z
        else:                                 # RAW
            zA, zB = sd * z, -sd * z
        if track:
            dWA[iA] += zA                     # accumulate BY IDENTITY
            dWB[iB] += zB
        xA = xA + zA; xB = xB + zB
    return dict(xA=xA, mA=mA, iA=iA, xB=xB, mB=mB, iB=iB,
                dWA=dWA, dWB=dWB, u_left=ul, **pre)


def pair_field(out, xg):
    """Terminal pair-mean reconstructed field: the estimator every arm shares."""
    fa = reconstruct_cumulative_field(out['xA'], out['mA'], out['u_left'], xg)
    fb = reconstruct_cumulative_field(out['xB'], out['mB'], out['u_left'], xg)
    return 0.5 * (fa + fb)


def heat_pair_field(out, x0, m0, xg):
    """Heat surrogate: each identity's own initial position plus its own
    accumulated production increments. Same masses, same identities."""
    ha = reconstruct_cumulative_field(x0 + out['dWA'], m0, out['u_left'], xg)
    hb = reconstruct_cumulative_field(x0 + out['dWB'], m0, out['u_left'], xg)
    return 0.5 * (ha + hb)


def mu_heat(x0, m0, u_left, nu, T, xg):
    """Exact expectation of the heat surrogate for these DISCRETE particles."""
    s = np.sqrt(2.0 * nu * T)
    return u_left + (m0[:, None] * ndtr((xg[None, :] - x0[:, None]) / s)).sum(0)


def rb_pair_field(out, nu, dt, xg):
    """Final-step conditional expectation: integrate out the last Gaussian stage.

    Given the pre-final state (Y, m), the terminal field is
    U(x) = u_left + sum_i m_i 1{Y_i + sd Z_i <= x} with Z iid N(0,1), so
    E[U(x) | pre-final] = u_left + sum_i m_i Phi((x - Y_i)/sd) exactly, for
    EITHER arm and under ANY coupling of the final noise. Hence the pair mean of
    these fields is the conditional expectation of the terminal pair estimator,
    E[Ubar | S], so E[U_RB] = E[Ubar] and, by the law of total variance,
    Var(U_RB) = Var(Ubar) - E[Var(Ubar | S)] <= Var(Ubar) pointwise. The mean is
    unchanged, so MSE cannot increase either.
    """
    sd = np.sqrt(2.0 * nu * dt)
    fa = out['u_left'] + (out['pre_mA'][:, None]
                          * ndtr((xg[None, :] - out['YA'][:, None]) / sd)).sum(0)
    fb = out['u_left'] + (out['pre_mB'][:, None]
                          * ndtr((xg[None, :] - out['YB'][:, None]) / sd)).sum(0)
    return 0.5 * (fa + fb)
