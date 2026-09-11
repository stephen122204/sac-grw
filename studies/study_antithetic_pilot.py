"""Equal-work antithetic pilot: do useful correlations survive nonlinear
interaction between gradient particles?

The comparison is deliberately fair.  An antithetic ESTIMATOR averages a run
and its reflected partner: two solver runs.  It is therefore compared against
the average of TWO INDEPENDENT runs, not against one.  Both estimators cost two
runs, so any advantage is a genuine variance reduction rather than a doubled
budget.

Reported per configuration:
  mse_anti   E|| (u + u') / 2 - uref ||^2   over replicates (2 runs each)
  mse_indep  E|| (u1 + u2) / 2 - uref ||^2  over replicates (2 runs each)
  rho        correlation between partners' signed field errors, the mechanism
             that would have to be negative for antithetic pairing to help

An asymmetric transient is included so that a success cannot be attributed to
cancellation across a symmetric shock.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import (advance_rbgbmc_particles,
                             initialize_tanh_shock_particles,
                             reconstruct_cumulative_field)
from studies.study_smooth_transient import initialize_gaussian_gradient_particles

OUT = ROOT / 'output/antithetic_pilot_2026_09_09'
N, R = 6400, 60
ARMS = {'mean': dict(conditional_mean_transport=True),
        'compensated': dict(compensate_transport_variance=True),
        'two_speed': dict()}
CASES = [dict(name='shock', problem='shock', a=2., nu=.5, dt=.005, T=2.5),
         dict(name='gaussian', problem='gaussian', a=4., nu=.1, dt=.005, T=1.)]


def setup(problem, nu):
    if problem == 'shock':
        x = np.linspace(-10., 14., 400)
        return x, -np.tanh((x - 2.) / (2 * nu)), initialize_tanh_shock_particles(N, nu, 1., 2.)
    z = np.load(ROOT / 'output/final_prepublication_tests/'
                       'gbmc_smooth_transient/reference.npz')
    return z['x'], z['u_ref'], initialize_gaussian_gradient_particles(N)


def run(initial, x, cfg, kw, seedkey, anti):
    rl = np.random.default_rng(np.random.SeedSequence(seedkey + [1]))
    rb = np.random.default_rng(np.random.SeedSequence(seedkey + [2]))
    r = advance_rbgbmc_particles(*initial, cfg['nu'], cfg['a'], cfg['dt'],
                                 round(cfg['T'] / cfg['dt']), rl,
                                 rng_brownian=rb, antithetic=anti, **kw)
    return reconstruct_cumulative_field(r['x'], r['m'], initial[2], x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ci, cfg in enumerate(CASES):
        x, ref, initial = setup(cfg['problem'], cfg['nu'])
        dx = float(x[1] - x[0])
        for arm, kw in ARMS.items():
            e_anti, e_indep, ea, eb = [], [], [], []
            for r_ in range(R):
                base = [26090902, ci, r_]
                ua = run(initial, x, cfg, kw, base + [0], False)
                ub = run(initial, x, cfg, kw, base + [0], True)      # reflected partner
                u1 = run(initial, x, cfg, kw, base + [10], False)
                u2 = run(initial, x, cfg, kw, base + [20], False)    # independent
                e_anti.append(dx * np.sum(((ua + ub) / 2 - ref) ** 2))
                e_indep.append(dx * np.sum(((u1 + u2) / 2 - ref) ** 2))
                ea.append(ua - ref); eb.append(ub - ref)
            ea, eb = np.asarray(ea), np.asarray(eb)
            # correlation of partner errors, pooled over replicates and points
            ca = ea - ea.mean(axis=0); cb = eb - eb.mean(axis=0)
            rho = float((ca * cb).sum() / np.sqrt((ca ** 2).sum() * (cb ** 2).sum()))
            ma, mi = float(np.mean(e_anti)), float(np.mean(e_indep))
            rng = np.random.default_rng(9); idx = rng.integers(0, R, (5000, R))
            ratio_b = np.sqrt(np.asarray(e_anti)[idx].mean(axis=1)
                              / np.asarray(e_indep)[idx].mean(axis=1))
            lo, hi = np.percentile(ratio_b, [2.5, 97.5])
            rows.append(dict(case=cfg['name'], arm=arm, R=R, N=N,
                             mse_anti=ma, mse_indep=mi,
                             rms_ratio=float(np.sqrt(ma / mi)),
                             ci_lo=float(lo), ci_hi=float(hi), partner_rho=rho))
            print(f"{cfg['name']:>9} {arm:>12} rms(anti)/rms(indep) = "
                  f"{np.sqrt(ma/mi):.4f} [{lo:.4f}, {hi:.4f}]   "
                  f"partner error correlation = {rho:+.4f}", flush=True)
    with (OUT / 'antithetic.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / 'antithetic.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
