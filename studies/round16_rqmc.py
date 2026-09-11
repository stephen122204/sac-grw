"""Round-16: randomized rank-based diffusion adaptation inspired by Lecot.

SOURCE MAPPING. Lecot, RIMS Kokyuroku 1240 (2001), Eqs. (44)-(45): particles are
sorted by position at the half step,

    x_0^{(n+1/2)} <= ... <= x_{N-1}^{(n+1/2)},                                (44)

and the diffusion step is

    x_{floor(N p_{j,1})}^{(n+1)} = x_{floor(N p_{j,1})}^{(n+1/2)}
                                   + sqrt(2 nu dt) Phi^{-1}(p_{j,2}),
    nN <= j < (n+1)N,                                                        (45)

with {p_j} a (0,2)-sequence of Faure in base 2 and N a power of two. So the FIRST
coordinate selects the spatial RANK and the SECOND supplies the Gaussian
increment through the inverse normal CDF; one consecutive block of N points is
consumed per time step.

WHAT WE FOLLOW: Eqs. (44)-(45) exactly -- rank selection by floor(N p_1),
increment by Phi^{-1}(p_2), one block of N per step, N a power of two.

WHAT IS OUR ADAPTATION, and must be labelled as such:
  * a scrambled Sobol' (0,m,2)-net replaces Faure's (0,2)-sequence. Both are
    digital nets in base 2 with the same equidistribution property used above.
  * the net is RANDOMIZED (Owen scrambling), one independent scramble per
    replicate, so the arm has a distribution over runs and can be compared on
    work-to-error with the stochastic arms. Lecot's construction is deterministic
    QMC. Independent complete scrambles are the units of uncertainty.
  * a fresh scrambled block is drawn per time step from the replicate's own
    generator.
  * the transport stage is OUR production carried-velocity stepper. Lecot's
    Burgers transport uses shock/rarefaction rules with collision-time updates
    (his Step 1), which is NOT reproduced. This arm is therefore a diffusion
    adaptation on our solver, not a reproduction of the published algorithm.

ENDPOINT HANDLING. Scrambled Sobol' returns values in [0,1); Phi^{-1}(0) = -inf.
p_2 is clipped to [eps, 1-eps] with eps = 2^-53 and the number of clipped values
is COUNTED and reported, never silently discarded. No point is skipped, padded
or truncated: every block is a complete 2^m-point net.

WHAT THIS IS NOT. QMC across complete interacting runs is a different
construction: preserving the baseline run law would require the correct joint
input law for the whole ensemble over all steps, not a correct Brownian marginal
per particle. This arm structures inputs WITHIN a run and therefore may change
the finite-N interacting law and its bias; its bias is estimated separately.
"""
import numpy as np
from scipy.stats import qmc, norm

EPS = 2.0 ** -53


class RQMCDiffusion:
    """Per-step rank-assigned diffusion increments from a scrambled (0,m,2)-net."""

    def __init__(self, N, seed):
        m = int(np.log2(N))
        if 2 ** m != N:
            raise ValueError(f'N must be a power of two for a base-2 net; got {N}')
        self.N, self.m = N, m
        self.engine = qmc.Sobol(d=2, scramble=True, seed=seed)
        self.clipped = 0
        self.blocks = 0

    def increments(self, sigma):
        """Returns z[rank] for one time step; z[r] is the increment for rank r."""
        p = self.engine.random(self.N)
        rank = np.floor(self.N * p[:, 0]).astype(np.int64)
        np.clip(rank, 0, self.N - 1, out=rank)
        # net property: each elementary interval [r/N,(r+1)/N) x [0,1) holds
        # exactly one point, so `rank` must be a permutation. Checked, not assumed.
        if not np.array_equal(np.sort(rank), np.arange(self.N)):
            raise RuntimeError('rank assignment is not a bijection: net property '
                               'violated (check N is a power of two and the '
                               'generator is not being advanced mid-block)')
        u = p[:, 1]
        nclip = int(np.sum((u < EPS) | (u > 1.0 - EPS)))
        self.clipped += nclip
        self.blocks += 1
        u = np.clip(u, EPS, 1.0 - EPS)
        z = np.empty(self.N)
        z[rank] = sigma * norm.ppf(u)
        return z


class IIDDiffusion:
    """Same interface, iid normals: the control used to show the isolated solver
    reduces exactly to production when the structured input is removed."""

    def __init__(self, N, seed):
        self.N = N
        self.rng = np.random.default_rng(seed)
        self.clipped = 0
        self.blocks = 0

    def increments(self, sigma):
        self.blocks += 1
        return sigma * self.rng.standard_normal(self.N)


def advance_rank_diffusion(initial, nu, dt, K, source, fprime):
    """One run: production carried-velocity transport, rank-assigned diffusion.

    Stage order is exactly production's: transport with the CARRIED velocity,
    sort (x, m) together, set the next velocity from the reconstruction, then
    diffuse. The sort that defines the ranks is the production sort; no extra
    permutation is introduced and the redraw-after-diffusion ablation is not used.
    """
    x0 = np.asarray(initial[0], float)
    m = np.asarray(initial[1], float)
    ul = float(initial[2])
    o = np.argsort(x0, kind='stable')
    x, m = x0[o].copy(), m[o].copy()
    sd = np.sqrt(2.0 * nu * dt)
    v = fprime(ul + np.cumsum(m))
    for _ in range(K):
        x = x + v * dt
        o = np.argsort(x, kind='stable')
        x, m = x[o], m[o]
        v = fprime(ul + np.cumsum(m))
        x = x + source.increments(sd)          # ranks are the sorted order
    return x, m, ul
