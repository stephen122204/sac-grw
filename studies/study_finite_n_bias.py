"""Q1 for Candidate 1: is there a resolvable finite-N bias to remove?

MLMC exists to buy an unbiased fine-level answer cheaply. If the finite-N bias
is below the ensemble noise floor, the coarse level alone suffices and a level
hierarchy is pure overhead, whatever the coupling quality.

Measures the ensemble-mean field at several N with a LARGE ensemble, using mean
transport at a small time step so the transport bias is not the thing being
seen. The successive differences ubar_N - ubar_{2N} estimate the N-bias; each
is reported against its own bootstrap interval so "unresolved" is distinguished
from "zero".
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import (advance_rbgbmc_particles,
                             reconstruct_cumulative_field)
from studies.study_smooth_transient import initialize_gaussian_gradient_particles

OUT = ROOT / 'output/finite_n_bias_2026_09_09'
S, NS = 400, (800, 1600, 3200, 6400)
NU, A, T, DT = .1, 4., 1., .00125


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / 'output/final_prepublication_tests/'
                       'gbmc_smooth_transient/reference.npz')
    x, ref = z['x'], z['u_ref']; dx = float(x[1] - x[0])
    steps = round(T / DT)
    fields = {}
    for N in NS:
        init = initialize_gaussian_gradient_particles(N)
        prof = []
        for s in range(S):
            k = [26090904, N, s]
            rl = np.random.default_rng(np.random.SeedSequence(k + [1]))
            rb = np.random.default_rng(np.random.SeedSequence(k + [2]))
            r = advance_rbgbmc_particles(*init, NU, A, DT, steps, rl,
                                         rng_brownian=rb,
                                         conditional_mean_transport=True)
            prof.append(reconstruct_cumulative_field(r['x'], r['m'], init[2], x))
        fields[N] = np.asarray(prof)
        v = fields[N]
        print(f"N={N:<6d} done, ||ubar-ref||={np.sqrt(dx*np.sum((v.mean(0)-ref)**2)):.4e}",
              flush=True)
    rng = np.random.default_rng(5)
    out = []
    print(f"\n{'pair':>14} {'||ubar_N - ubar_2N||':>21} {'95% CI':>26} {'resolved?':>10}")
    for i in range(len(NS) - 1):
        a, b = NS[i], NS[i + 1]
        d = np.sqrt(dx * np.sum((fields[a].mean(0) - fields[b].mean(0)) ** 2))
        boot = []
        for _ in range(3000):
            ia = rng.integers(0, S, S); ib = rng.integers(0, S, S)
            boot.append(np.sqrt(dx * np.sum(
                (fields[a][ia].mean(0) - fields[b][ib].mean(0)) ** 2)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # noise floor: difference between two disjoint halves of the SAME N
        h = S // 2
        floor = np.sqrt(dx * np.sum(
            (fields[b][:h].mean(0) - fields[b][h:].mean(0)) ** 2)) / np.sqrt(2)
        res = 'yes' if lo > floor else 'no'
        out.append(dict(N_coarse=a, N_fine=b, diff=float(d), ci_lo=float(lo),
                        ci_hi=float(hi), same_N_noise_floor=float(floor),
                        resolved=res))
        print(f"{a:>6}->{b:<7} {d:21.4e} [{lo:.3e},{hi:.3e}]  floor={floor:.2e} {res:>6}")
    (OUT / 'finite_n_bias.json').write_text(json.dumps(
        dict(S=S, Ns=list(NS), nu=NU, a=A, T=T, dt=DT, pairs=out), indent=1))


if __name__ == '__main__':
    main()
