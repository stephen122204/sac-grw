"""Transport-aware known-mean control variate (Section 4.3, Appendix D)."""
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradient_particles import reconstruct_cumulative_field
from studies.spectral_reference import FPRIME


def initial_velocity_by_identity(x0, m0, u_left, flux):
    """v_i^0 indexed by IDENTITY (position in the caller's x0/m0 arrays)."""
    o = np.argsort(np.asarray(x0, float), kind='stable')
    v_sorted = FPRIME[flux](u_left + np.cumsum(np.asarray(m0, float)[o]))
    v0 = np.empty(len(o))
    v0[o] = v_sorted
    return v0


def frozen_pair_field(out, x0, m0, v0, T, xg):
    """Pair-mean frozen-velocity surrogate from the run's own increments."""
    fa = reconstruct_cumulative_field(x0 + v0 * T + out['dWA'], m0, out['u_left'], xg)
    fb = reconstruct_cumulative_field(x0 + v0 * T + out['dWB'], m0, out['u_left'], xg)
    return 0.5 * (fa + fb)


def mu_frozen(x0, m0, v0, u_left, nu, T, xg):
    """Exact mean of the frozen-velocity surrogate for these DISCRETE particles."""
    s = np.sqrt(2.0 * nu * T)
    return u_left + (m0[:, None]
                     * ndtr((xg[None, :] - x0[:, None] - v0[:, None] * T) / s)).sum(0)


def advance_pair_frozen_transport(initial, nu, dt, K, rng, policy, flux):
    """FIXTURE ONLY: the same paired update with the velocity frozen at v_i^0 and
    carried with the identity. Terminal position of identity i is then exactly
    x_i^0 + v_i^0 T + sum_k dW_{i,k}, so U must equal F pathwise."""
    x0 = np.asarray(initial[0], float)
    m0 = np.asarray(initial[1], float)
    ul = float(initial[2])
    n = len(x0)
    o = np.argsort(x0, kind='stable')
    xA, mA, iA = x0[o].copy(), m0[o].copy(), o.copy()
    xB, mB, iB = xA.copy(), mA.copy(), iA.copy()
    v0 = initial_velocity_by_identity(x0, m0, ul, flux)
    vA, vB = v0[iA].copy(), v0[iB].copy()
    sd = np.sqrt(2.0 * nu * dt)
    dWA, dWB = np.zeros(n), np.zeros(n)
    for _ in range(K):
        xA = xA + vA * dt
        o = np.argsort(xA, kind='stable'); xA, mA, iA, vA = xA[o], mA[o], iA[o], vA[o]
        xB = xB + vB * dt
        o = np.argsort(xB, kind='stable'); xB, mB, iB, vB = xB[o], mB[o], iB[o], vB[o]
        z = rng.standard_normal(n)
        if policy == 'FULL':
            zA = sd * z
            zB = -sd * (np.sign(mA) * np.sign(mB)) * z
        else:
            zA, zB = sd * z, -sd * z
        dWA[iA] += zA; dWB[iB] += zB
        xA = xA + zA; xB = xB + zB
    return dict(xA=xA, mA=mA, xB=xB, mB=mB, dWA=dWA, dWB=dWB, u_left=ul)
