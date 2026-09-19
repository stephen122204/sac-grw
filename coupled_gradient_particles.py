"""Coupled advance of two simulations of Algorithm 1 under the admissible couplings of Section 3.2: RAW reflects at equal rank, FULL switches by the mass-sign product, WITHIN reflects within each sign class, FINAL switches only at the last stage. Multipliers and matching are fixed before each draw, so each replica keeps the single-simulation law (Section 3.3)."""

import numpy as np


def advance_pair(
    initial, nu, dt, K, rng, policy, flux_derivative, *, pre_diffusion=None
):
    """Paired general-flux inclusive-schedule update. Sign multipliers and the
    matching are functions of the pre-diffusion configuration, hence predictable
    before Z is drawn, so each arm keeps its exact marginal law."""
    if policy not in {"RAW", "FULL", "FINAL", "WITHIN"}:
        raise ValueError("unknown coupling policy")
    if not (np.isfinite(nu) and nu >= 0 and np.isfinite(dt) and dt > 0):
        raise ValueError("nu must be nonnegative and dt positive")
    if not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError("K must be a nonnegative integer")
    fp = flux_derivative
    x0, m0, ul = (
        np.asarray(initial[0], float),
        np.asarray(initial[1], float),
        initial[2],
    )
    if x0.ndim != 1 or x0.size == 0 or m0.shape != x0.shape:
        raise ValueError("positions and masses must be nonempty equal-sized vectors")
    if not (np.isfinite(x0).all() and np.isfinite(m0).all() and np.isfinite(ul)):
        raise ValueError("initial state must be finite")
    if (m0 == 0).any():
        raise ValueError("this implementation excludes zero masses")
    o = np.argsort(x0, kind="stable")
    xA, mA = x0[o].copy(), m0[o].copy()
    xB, mB = xA.copy(), mA.copy()
    n = len(xA)
    sd = np.sqrt(2.0 * nu * dt)
    vA = fp(ul + np.cumsum(mA))
    vB = vA.copy()
    for k in range(K):
        xA = xA + vA * dt
        o = np.argsort(xA, kind="stable")
        xA, mA = xA[o], mA[o]
        vA = fp(ul + np.cumsum(mA))
        xB = xB + vB * dt
        o = np.argsort(xB, kind="stable")
        xB, mB = xB[o], mB[o]
        vB = fp(ul + np.cumsum(mB))
        if pre_diffusion is not None:
            # Copies make instrumentation unable to modify the simulation state.
            pre_diffusion(
                k, xA.copy(), mA.copy(), vA.copy(), xB.copy(), mB.copy(), vB.copy()
            )
        z = rng.standard_normal(n)
        sign_aware = (policy == "FULL") or (policy == "FINAL" and k == K - 1)
        if policy == "WITHIN":
            sA, sB = np.sign(mA), np.sign(mB)
            zB = np.empty(n)
            for sgn in (1.0, -1.0):
                iA = np.flatnonzero(sA == sgn)
                iB = np.flatnonzero(sB == sgn)
                if len(iA) != len(iB):
                    raise ValueError("within-sign matching requires equal sign counts")
                zB[iB] = -z[iA]
            zA, zB = sd * z, sd * zB
        elif sign_aware:
            zA = sd * z
            zB = -sd * (np.sign(mA) * np.sign(mB)) * z
        else:
            zA, zB = sd * z, -sd * z
        xA = xA + zA
        xB = xB + zB
    return (xA, mA), (xB, mB), ul
