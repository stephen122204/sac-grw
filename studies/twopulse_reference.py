"""Reference solution and particle initialization for the two-pulse profile (Appendix C)."""
import numpy as np
from scipy.special import erf
from scipy.integrate import quad

PULSES = ((0.8, -0.7, 0.5), (-0.5, 0.4, 0.3))   # (amplitude, centre, width)


def u0(x):
    x = np.asarray(x, dtype=float)
    return sum(A * np.exp(-(x - c) ** 2 / (2 * s * s)) for A, c, s in PULSES)


def du0(x):
    x = np.asarray(x, dtype=float)
    return sum(A * (-(x - c) / (s * s)) * np.exp(-(x - c) ** 2 / (2 * s * s))
               for A, c, s in PULSES)


def F(y):
    """int_0^y u0(s) ds, analytic."""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    for A, c, s in PULSES:
        k = A * s * np.sqrt(np.pi / 2)
        out = out + k * (erf((y - c) / (s * np.sqrt(2))) - erf((0 - c) / (s * np.sqrt(2))))
    return out


def exact_area():
    """int u0 dx on the whole line, analytic."""
    return sum(A * s * np.sqrt(2 * np.pi) for A, c, s in PULSES)


def cole_hopf(x_out, t, nu, half_width=40.0, tol=1e-11):
    """Stabilised Cole-Hopf evaluation at the requested points."""
    vals = np.empty_like(np.asarray(x_out, dtype=float))
    for i, x in enumerate(np.atleast_1d(x_out)):
        lo, hi = x - half_width * max(np.sqrt(2 * nu * t), 1.0), \
                 x + half_width * max(np.sqrt(2 * nu * t), 1.0)
        yy = np.linspace(lo, hi, 4001)
        G = F(yy) + (x - yy) ** 2 / (2 * t)
        Gm = G.min()
        num = quad(lambda y: ((x - y) / t) * np.exp(-(F(y) + (x - y) ** 2 / (2 * t) - Gm) / (2 * nu)),
                   lo, hi, epsabs=tol, epsrel=tol, limit=400)[0]
        den = quad(lambda y: np.exp(-(F(y) + (x - y) ** 2 / (2 * t) - Gm) / (2 * nu)),
                   lo, hi, epsabs=tol, epsrel=tol, limit=400)[0]
        vals[i] = num / den
    return vals


def initialize(N):
    """Deterministic signed-quantile particles; equal counts and weight per sign."""
    assert N % 2 == 0
    g = np.linspace(-12.0, 12.0, 400001)
    d = du0(g)
    tv = np.trapz(np.abs(d), g)
    xs, ms = [], []
    for sgn in (+1.0, -1.0):
        part = np.where(np.sign(d) == sgn, np.abs(d), 0.0)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (part[1:] + part[:-1]) * np.diff(g))])
        total = cdf[-1]
        r = (np.arange(1, N // 2 + 1) - 0.5) / (N // 2)
        xs.append(np.interp(r * total, cdf, g))
        ms.append(np.full(N // 2, sgn * tv / N))
    x = np.concatenate(xs); m = np.concatenate(ms)
    o = np.argsort(x); x, m = x[o], m[o]
    return x, m, 0.0, dict(total_variation=float(tv),
                           positive_weight=float(tv / 2), particle_mass=float(tv / N))
