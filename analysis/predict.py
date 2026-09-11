"""Parameter-free predictions of the paired fitted-viscosity difference.

For each dimensionless configuration (at, dtt, taut, window) the two candidate
models are evolved from the exact stationary nu-shock and passed through the
SAME three-parameter tanh fit used for the particle profiles:

  M2 (derived)      U_tau + U U_X = d/dX[(1 + delta(U)) U_X],
                    delta(U) = (dtt/2)(at^2 - U^2)
  M1 (naive)        U_tau + U U_X = (1 + Dvel/nu) U_XX,   evaluated exactly by
                    Cole-Hopf

The control arm's prediction is the exact stationary shock itself, so the
predicted paired difference is  nu_hat[model] - nu.  Nothing is fitted to the
particle data.
"""

import json
import os

import numpy as np
from scipy.optimize import curve_fit

from .effective_equation import (solve_modified, cole_hopf_constant_nu,
                                 steady_profile_modified)


def fit_ratio(X, U, r_hi=20.0):
    """Three-parameter tanh fit in dimensionless variables; returns
    (U_hat, Xc_hat, r_hat) with r_hat = nu_hat/nu."""
    def model(x, Uh, Xc, r):
        return -Uh * np.tanh(Uh * (x - Xc) / (2.0 * r))
    p, _ = curve_fit(model, X, U, p0=[1.0, 0.0, 1.0],
                     bounds=([0.5, X[0], 0.02], [2.0, X[-1], r_hi]), maxfev=20000)
    return float(p[0]), float(p[1]), float(p[2])


def d_vel_over_nu(at, dtt, N=6400):
    return 0.5 * dtt * (at ** 2 - (1.0 / 3.0 + 2.0 / (3.0 * N ** 2)))


_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'output', 'reevaluation_2026_09',
                           'prediction_cache.json')
_CACHE = None


def _cache():
    """Predictions are deterministic functions of (at, dtt, taut, window, M, N),
    so they are memoised on disk.  Delete the cache file to force recomputation."""
    global _CACHE
    if _CACHE is None:
        try:
            with open(_CACHE_PATH) as fh:
                _CACHE = json.load(fh)
        except (OSError, ValueError):
            _CACHE = {}
    return _CACHE


def predict(at, dtt, taut, half_widths=12.0, M=400, r_hi=20.0,
            dX=0.01, Xmax=None, N=6400):
    """Return the predicted paired fitted-viscosity difference, in units of nu
    and of D_vel, for both models."""
    key = f'{at:.10g}|{dtt:.10g}|{taut:.10g}|{half_widths:.10g}|{M}|{N}|{dX:.10g}'
    cache = _cache()
    if key in cache:
        return dict(cache[key])
    halfX = 2.0 * half_widths                    # shock width = 2 in X units
    Xw = np.linspace(-halfX, halfX, M)
    D = d_vel_over_nu(at, dtt, N)
    Xmax = Xmax or max(140.0, 6.0 * halfX)
    X, U = solve_modified(taut, dtt=dtt, at=at, Xmax=Xmax, dX=dX)
    r2 = fit_ratio(Xw, np.interp(Xw, X, U), r_hi)[2]
    r1 = fit_ratio(Xw, cole_hopf_constant_nu(Xw, taut, 1.0 + D), r_hi)[2]
    rs = fit_ratio(Xw, steady_profile_modified(Xw, dtt, at), r_hi)[2]
    out = dict(at=at, dtt=dtt, taut=taut, half_widths=half_widths,
               D_vel_over_nu=D,
               M2_dnu_over_nu=r2 - 1.0, M2_g=(r2 - 1.0) / D,
               M1_dnu_over_nu=r1 - 1.0, M1_g=(r1 - 1.0) / D,
               M2_steady_g=(rs - 1.0) / D)
    cache[key] = out
    os.makedirs(os.path.dirname(os.path.abspath(_CACHE_PATH)), exist_ok=True)
    with open(_CACHE_PATH, 'w') as fh:
        json.dump(cache, fh)
    return dict(out)
