"""Fixed-budget allocation study: does a resolution hierarchy exist to exploit?

Candidate 1 (multilevel coupling) presupposes the standard MLMC structure: a
cheap biased level and an expensive accurate one, with cost per SAMPLE rising
as accuracy improves.  Gradient particle methods may not have it, because the
particle count N is simultaneously the discretisation parameter AND the sample
size: one run of N particles already has field variance ~ C/N.

If var ~ C/N (verified flat in the decomposition data), then for S independent
runs the ensemble-mean variance is ~ C/(S N) while cost ~ S N (T/dt).  Variance
per unit work then depends only on the PRODUCT S*N, so at fixed budget the
split between run count and particle count can only matter through BIAS.

This driver measures the estimator error E|| ubar - uref ||^2 at MATCHED work
for several allocations of the same budget, with measured runtime recorded so
the particle-step proxy can be checked.  R independent ensembles per allocation
give the expectation and its uncertainty.
"""
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import (advance_rbgbmc_particles,
                             initialize_tanh_shock_particles,
                             reconstruct_cumulative_field)
from studies.study_smooth_transient import initialize_gaussian_gradient_particles

OUT = ROOT / 'output/fixed_budget_2026_09_09'
R = 15
PROBLEMS = {'shock':    dict(a=2., nu=.5, T=2.5, dt0=.005),
            'gaussian': dict(a=4., nu=.1, T=1.,  dt0=.005)}
# (label, N, S, dt factor, arm, antithetic)  -- budget S*N*(T/dt) matched to A
ALLOC = [
    ('A_ref            N=6400  S=50', 6400, 50, 1, 'mean', False),
    ('B_manyruns       N=1600  S=200', 1600, 200, 1, 'mean', False),
    ('C_fewruns        N=25600 S=12', 25600, 12, 1, 'mean', False),
    ('D_finer_dt       N=6400  S=25', 6400, 25, 2, 'mean', False),
    ('E_bigN           N=12800 S=25', 12800, 25, 1, 'mean', False),
    ('F_antithetic     N=6400  S=50', 6400, 50, 1, 'mean', True),
    ('G_comp_antith    N=6400  S=50', 6400, 50, 1, 'compensated', True),
    ('H_twospeed_diag  N=6400  S=50', 6400, 50, 1, 'two_speed', False),
]
KW = {'mean': dict(conditional_mean_transport=True),
      'compensated': dict(compensate_transport_variance=True),
      'two_speed': dict()}


def setup(problem, nu, N):
    if problem == 'shock':
        x = np.linspace(-10., 14., 400)
        return x, -np.tanh((x - 2.) / (2 * nu)), initialize_tanh_shock_particles(N, nu, 1., 2.)
    z = np.load(ROOT / 'output/final_prepublication_tests/'
                       'gbmc_smooth_transient/reference.npz')
    return z['x'], z['u_ref'], initialize_gaussian_gradient_particles(N)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for pname, cfg in PROBLEMS.items():
        for (label, N, S, dtf, arm, anti) in ALLOC:
            dt = cfg['dt0'] / dtf
            steps = round(cfg['T'] / dt)
            x, ref, initial = setup(pname, cfg['nu'], N)
            dx = float(x[1] - x[0])
            errs, secs = [], []
            for rep in range(R):
                t0 = time.perf_counter()
                acc = np.zeros_like(ref)
                for s in range(S):
                    # antithetic: runs pair up (s even = base, s odd = reflected partner
                    # sharing the same stream), so the ensemble still costs S runs.
                    stream = s // 2 if anti else s
                    use_anti = anti and (s % 2 == 1)
                    key = [26090903, abs(hash(pname)) % 9973, N, S, dtf,
                           abs(hash(arm)) % 9973, rep, stream]
                    rl = np.random.default_rng(np.random.SeedSequence(key + [1]))
                    rb = np.random.default_rng(np.random.SeedSequence(key + [2]))
                    r = advance_rbgbmc_particles(*initial, cfg['nu'], cfg['a'], dt,
                                                 steps, rl, rng_brownian=rb,
                                                 antithetic=use_anti, **KW[arm])
                    acc += reconstruct_cumulative_field(r['x'], r['m'], initial[2], x)
                secs.append(time.perf_counter() - t0)
                errs.append(float(dx * np.sum((acc / S - ref) ** 2)))
            errs = np.asarray(errs)
            rng = np.random.default_rng(11); idx = rng.integers(0, R, (4000, R))
            b = np.sqrt(errs[idx].mean(axis=1))
            rows.append(dict(problem=pname, alloc=label, N=N, S=S, dt=dt, arm=arm,
                             antithetic=anti, R=R,
                             particle_steps=S * N * steps,
                             rms=float(np.sqrt(errs.mean())),
                             ci_lo=float(np.percentile(b, 2.5)),
                             ci_hi=float(np.percentile(b, 97.5)),
                             median_seconds=float(np.median(secs))))
            q = rows[-1]
            print(f"{pname:>9} {label:<34} rms(ubar)={q['rms']:.5e} "
                  f"[{q['ci_lo']:.4e},{q['ci_hi']:.4e}] "
                  f"work={q['particle_steps']:.3g} t={q['median_seconds']:.2f}s",
                  flush=True)
    with (OUT / 'fixed_budget.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / 'fixed_budget.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
