"""Particle-count sensitivity of the paired ratio g.

The effective equation is an N -> infinity statement, so a residual dependence
of g on N would bound the validity of the comparison.  Recomputes g at three
particle counts for a/A = 2, dtt = 0.005, at two dimensionless times.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import studies.study_response_time as st
from analysis.predict import predict

OUT = 'output/reevaluation_2026_09/particle_count_check.json'


def main(seeds=120, counts=(1600, 6400, 25600), tauts=(2.0, 10.0), at=2.0):
    A, nu, dtt = 1.0, 0.5, 0.005
    rows = []
    for N in counts:
        st.N_FIXED = N                      # the driver's frozen production count
        tag = st._key(A, nu, at, dtt) + [N]
        r, prof, x_out, dx = st.run_cell(A, nu, at, dtt, tauts, seeds, tag)
        summ = st.paired_stats(r, prof, x_out, dx, A, nu, at * A, dtt * nu / A ** 2)
        for s in summ:
            p = predict(at, dtt, s['taut'], N=N)
            rows.append(dict(N=N, taut=s['taut'], S=seeds, g=s['g'],
                             g_lo=s['g_lo'], g_hi=s['g_hi'],
                             d_nu=s['d_nu'], D_vel=s['D_vel'],
                             M2=p['M2_g'], M1=p['M1_g']))
            print(f"N={N:6d} T~={s['taut']:5g}  g={s['g']:.4f} "
                  f"[{s['g_lo']:.4f},{s['g_hi']:.4f}]  M2={p['M2_g']:.4f}")
    json.dump(rows, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main(seeds=int(sys.argv[1]) if len(sys.argv) > 1 else 120)
