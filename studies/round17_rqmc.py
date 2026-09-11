"""Round-17: corrected randomized rank-based diffusion adaptation (Lecot Eqs. 44-45).

Supersedes studies/round16_rqmc.py, which is retained unchanged for provenance.

RANDOMIZATION ACTUALLY IMPLEMENTED -- stated precisely.
  ONE scipy.stats.qmc.Sobol engine is created per replicate with one scramble;
  `increments` then calls `engine.random(N)` repeatedly, so successive TIME STEPS
  consume SUCCESSIVE BLOCKS of that single scrambled sequence. The net is NOT
  independently re-scrambled at each step. This matches Lecot Eq. (45), which
  consumes points nN <= j < (n+1)N from one sequence. Within-run temporal
  dependence is therefore part of the adaptation, and the statistical units are
  INDEPENDENT COMPLETE REPLICATES (one scramble each), never steps or points.

  SciPy documents `scramble=True` for Sobol as a LEFT-MATRIX SCRAMBLE (LMS)
  COMBINED WITH A DIGITAL RANDOM SHIFT -- not fully nested Owen scrambling. The
  installed version and the `bits` setting are recorded at run time.

  We do NOT claim this adaptation preserves the interacting iid law. A one-step
  expectation check says nothing about multistep conditional Brownian laws. The
  construction is stated as it is and its TOTAL error is measured independently.

TIMING FAIRNESS. The bijection assertion (`np.sort(rank)`) is a validation cost
absent from the other arms. It is confined to `debug=True`, which is used in
fixtures only; timed runs use `debug=False`. Equivalence of the two paths on the
same input stream is demonstrated in the round-17 driver.

ENDPOINTS. Sobol returns values in [0,1); Phi^{-1}(0) = -inf. p2 is clipped to
[2^-53, 1-2^-53]; clipped values are counted and reported, never discarded.
"""
import numpy as np
from scipy.stats import qmc, norm
import scipy

EPS = 2.0 ** -53
SOBOL_BITS = 30


class RQMCDiffusion:
    def __init__(self, N, seed, debug=False):
        m = int(np.log2(N))
        if 2 ** m != N:
            raise ValueError(f'N must be a power of two for a base-2 net; got {N}')
        self.N, self.m, self.debug = N, m, debug
        self.engine = qmc.Sobol(d=2, scramble=True, bits=SOBOL_BITS, seed=seed)
        self.clipped = 0
        self.blocks = 0
        self.t_gen = self.t_rank = self.t_icdf = 0.0

    @staticmethod
    def provenance():
        return dict(scipy_version=scipy.__version__,
                    engine='scipy.stats.qmc.Sobol(d=2, scramble=True)',
                    scramble='LMS + digital random shift (per SciPy docs); '
                             'NOT fully nested Owen scrambling',
                    bits=SOBOL_BITS,
                    sequence_use='one engine per replicate; successive time steps '
                                 'consume successive blocks of N=2^m points',
                    endpoint_handling=f'p2 clipped to [{EPS}, 1-{EPS}], count reported')

    def increments(self, sigma, profile=False):
        import time as _t
        t0 = _t.perf_counter() if profile else 0.0
        p = self.engine.random(self.N)
        t1 = _t.perf_counter() if profile else 0.0
        rank = np.floor(self.N * p[:, 0]).astype(np.int64)
        np.clip(rank, 0, self.N - 1, out=rank)
        if self.debug and not np.array_equal(np.sort(rank), np.arange(self.N)):
            raise RuntimeError('rank assignment is not a bijection')
        t2 = _t.perf_counter() if profile else 0.0
        u = p[:, 1]
        self.clipped += int(np.sum((u < EPS) | (u > 1.0 - EPS)))
        u = np.clip(u, EPS, 1.0 - EPS)
        z = np.empty(self.N)
        z[rank] = sigma * norm.ppf(u)
        if profile:
            t3 = _t.perf_counter()
            self.t_gen += t1 - t0; self.t_rank += t2 - t1; self.t_icdf += t3 - t2
        self.blocks += 1
        return z


def advance_rank_diffusion(initial, nu, dt, K, source, fprime, profile=False):
    """Production carried-velocity transport, rank-assigned diffusion.

    Returns (x, m, u_left, v) -- the CARRIED VELOCITY is returned so the full
    production state can be compared, not only positions and masses.
    """
    import time as _t
    x0 = np.asarray(initial[0], float)
    m = np.asarray(initial[1], float)
    ul = float(initial[2])
    o = np.argsort(x0, kind='stable')
    x, m = x0[o].copy(), m[o].copy()
    sd = np.sqrt(2.0 * nu * dt)
    v = fprime(ul + np.cumsum(m))
    t_tr = 0.0
    for _ in range(K):
        t0 = _t.perf_counter() if profile else 0.0
        x = x + v * dt
        o = np.argsort(x, kind='stable')
        x, m = x[o], m[o]
        v = fprime(ul + np.cumsum(m))
        if profile:
            t_tr += _t.perf_counter() - t0
        x = x + source.increments(sd, profile=profile)
    return (x, m, ul, v, t_tr) if profile else (x, m, ul, v)
