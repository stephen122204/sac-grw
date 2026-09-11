"""Does the derived error mechanism yield a correction that removes the error?

The analysis says the two-speed step is the conditional-mean step plus one
mean-zero increment of conditional variance (a^2 - u_i^2) dt^2.  If that is the
whole story, subtracting exactly that variance from the Brownian step should
restore the accuracy of direct transport while keeping the relaxation
structure.  This is a falsifiable consequence, not a tuning knob: the
subtraction is fixed by the derivation, with no free parameter.

Three arms share the Brownian stream: conditional-mean control, two-speed, and
two-speed with variance compensation.  Prediction: the compensated arm's paired
fitted-viscosity difference against the control is zero.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relaxation_gbmc import (advance_rbgbmc_particles,
                             initialize_tanh_shock_particles,
                             reconstruct_cumulative_field)
import studies.study_response_time as st

OUT = 'output/reevaluation_2026_09/compensation_test.json'
CASES = [dict(at=4.0, dtt=0.005, tauts=(5.0, 20.0)),
         dict(at=2.0, dtt=0.005, tauts=(5.0, 20.0))]
ARMS = ['cond_mean', 'two_speed', 'two_speed_compensated']


def run_case(A, nu, at, dtt, tauts, seeds):
    a, dt = at * A, dtt * nu / A ** 2
    steps = {t: int(round(t / dtt)) for t in tauts}
    n_max = max(steps.values())
    x_out, dx = st._window(A, nu)
    fits = {}
    for arm in ARMS:
        for s in range(seeds):
            x0, m0, u_left = initialize_tanh_shock_particles(st.N_FIXED, nu, A, st.XC)
            rl, rb = st._rngs(st._key(A, nu, at, dtt) + [99], s)
            run = advance_rbgbmc_particles(
                x0, m0, u_left, nu, a, dt, n_max, rl, rng_brownian=rb,
                snapshot_steps=set(steps.values()),
                conditional_mean_transport=(arm == 'cond_mean'),
                compensate_transport_variance=(arm == 'two_speed_compensated'))
            for t, k in steps.items():
                xs, ms = run['snapshots'][k]
                u = reconstruct_cumulative_field(xs, ms, u_left, x_out)
                fits[(arm, t, s)] = st.fit_tanh3(u=u, x=x_out, A=A, nu=nu,
                                                 x_lo=x_out[0], x_hi=x_out[-1])[2]
    return fits, steps


def main(seeds=150):
    A, nu = 1.0, 0.5
    rng = np.random.default_rng(7)
    rows = []
    for c in CASES:
        t0 = time.perf_counter()
        fits, steps = run_case(A, nu, c['at'], c['dtt'], c['tauts'], seeds)
        D = st.d_vel(c['at'] * A, A, c['dtt'] * nu / A ** 2)
        for t in c['tauts']:
            for arm in ('two_speed', 'two_speed_compensated'):
                d = np.array([fits[(arm, t, s)] - fits[('cond_mean', t, s)]
                              for s in range(seeds)])
                b = np.array([d[rng.integers(0, seeds, seeds)].mean()
                              for _ in range(5000)])
                lo, hi = np.percentile(b, [2.5, 97.5])
                rows.append(dict(at=c['at'], dtt=c['dtt'], taut=t, arm=arm,
                                 S=seeds, D_vel=D, d_nu=float(d.mean()),
                                 lo=float(lo), hi=float(hi),
                                 g=float(d.mean() / D),
                                 g_lo=float(lo / D), g_hi=float(hi / D)))
                print(f"a/A={c['at']:g} T~={t:<5g} {arm:<22s} "
                      f"d_nu={d.mean():+.6f} [{lo:+.6f},{hi:+.6f}]  "
                      f"g={d.mean()/D:+.4f} [{lo/D:+.4f},{hi/D:+.4f}]")
        print(f"  ({time.perf_counter()-t0:.0f} s)")
    json.dump(rows, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
