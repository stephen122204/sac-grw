"""Where does the remaining error live after the systematic diffusion error is corrected?

Varies particle count and time step INDEPENDENTLY for three transport arms and
splits the mean squared L2 error into a squared-bias and a solution-variance
part.  The finite-ensemble correction matters here: with S realizations,

    E[ ||ubar - uref||^2 ] = ||mu_N - uref||^2 + E[spread^2]/(S-1),

so the raw ensemble-mean error OVERSTATES the true bias.  The corrected
estimate subtracts spread^2/(S-1) and is clipped at zero when the bias is
below the sampling floor, which is itself the finding in that cell.
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

OUT = ROOT / 'output/error_decomposition_2026_09_09'
S = 50
ARMS = {'mean': dict(conditional_mean_transport=True),
        'compensated': dict(compensate_transport_variance=True),
        'two_speed': dict()}
PROBLEMS = {
    'shock':    dict(a=2., nu=.5, T=2.5, dts=(.01, .005, .0025)),
    'gaussian': dict(a=4., nu=.1, T=1.,  dts=(.005, .0025, .00125)),
}
COUNTS = (1600, 6400, 25600)


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
        for N in COUNTS:
            x, ref, initial = setup(pname, cfg['nu'], N)
            dx = float(x[1] - x[0])
            for dt in cfg['dts']:
                steps = round(cfg['T'] / dt)
                for arm, kw in ARMS.items():
                    prof, secs = [], []
                    for s in range(S):
                        rl = np.random.default_rng(np.random.SeedSequence(
                            [26090901, hash(pname) % 9973, N, int(dt * 1e6), s, 1]))
                        rb = np.random.default_rng(np.random.SeedSequence(
                            [26090901, hash(pname) % 9973, N, int(dt * 1e6), s, 2]))
                        t0 = time.perf_counter()
                        r = advance_rbgbmc_particles(*initial, cfg['nu'], cfg['a'],
                                                     dt, steps, rl, rng_brownian=rb, **kw)
                        secs.append(time.perf_counter() - t0)
                        prof.append(reconstruct_cumulative_field(r['x'], r['m'],
                                                                 initial[2], x))
                    v = np.asarray(prof); avg = v.mean(axis=0)
                    raw_bias2 = float(dx * np.sum((avg - ref) ** 2))
                    spread2 = float(dx * np.mean(np.sum((v - avg) ** 2, axis=1)))
                    mse = float(dx * np.mean(np.sum((v - ref) ** 2, axis=1)))
                    corrected = raw_bias2 - spread2 / (S - 1)
                    rows.append(dict(problem=pname, N=N, dt=dt, arm=arm, S=S,
                                     mse=mse, raw_bias2=raw_bias2, spread2=spread2,
                                     bias2_corrected=corrected,
                                     bias_below_floor=bool(corrected <= 0),
                                     bias_frac=max(corrected, 0.0) / mse,
                                     seconds=float(np.median(secs))))
                    r_ = rows[-1]
                    print(f"{pname:>9} N={N:<6d} dt={dt:<8g} {arm:>12} "
                          f"mse={mse:.3e} bias2={max(corrected,0):.3e} "
                          f"({r_['bias_frac']:5.1%}) var={spread2:.3e} "
                          f"t={r_['seconds']:.3f}s", flush=True)
    with (OUT / 'decomposition.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / 'decomposition.json').write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {len(rows)} cells")


if __name__ == '__main__':
    main()
