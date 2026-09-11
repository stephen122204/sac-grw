"""Round-6 part 1e: high-power addendum to the sensitivity experiment.

At S=3000 the Bonferroni detection floor is 19-55% of the peak response, so
"zero resolved wrong-sign points" was a weak statement and a 3.5% wrong-sign
FRACTION is exactly what counting unresolved sign flips in that noise produces.
This run raises S by 16x on the Burgers case (and one heat control) so the floor
falls by 4x, and reports the resulting BOUND on any wrong-sign response rather
than only a count. Same configuration, same displaced particle, same resolved
definition, independent seed keys.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.study_round06_sensitivity import (base_state, pick_particle, evolve,
                                               XG, ALPHA, NU, T, H, NPART)

OUT = ROOT / 'output/round06_sensitivity_2026_09_09'
S_HIGH = 48000
CASES = [('burgers', 0.05, 11), ('burgers', 0.1, 12), ('heat', 0.05, 13)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x0, m0, ul = base_state()
    i0 = pick_particle(x0, m0)
    sgn = np.sign(m0[i0])
    z = norm.ppf(1.0 - ALPHA / (2 * len(XG)))
    dxg = float(XG[1] - XG[0])
    print(f"high-power sensitivity: S={S_HIGH}, particle {i0} "
          f"(m sign {int(sgn):+d}), Bonferroni over {len(XG)} points\n")
    print(f"{'drift':>9} {'d':>6} {'key':>4} {'peak|R|':>10} {'floor':>10} "
          f"{'floor/peak':>11} {'wrong':>6} {'bound on wrong/peak':>20}")
    rows, arch = [], {}
    t0 = time.perf_counter()
    for drift, d, key in CASES:
        xp = x0.copy(); xp[i0] += d
        ids = np.arange(NPART)
        Fb = evolve(x0, m0, ul, drift, key, S_HIGH, ids)
        Fp = evolve(xp, m0, ul, drift, key, S_HIGH, ids)
        D = Fp - Fb
        R = D.mean(axis=0)
        se = D.std(axis=0, ddof=1) / np.sqrt(S_HIGH)
        wrong = sgn * R > z * se
        floor = float(z * se.max())
        peak = float(np.max(np.abs(R)))
        rows.append(dict(drift=drift, displacement=d, seed_key=key, seeds=S_HIGH,
                         peak_abs_response=peak, detection_floor=floor,
                         floor_over_peak=floor / peak,
                         n_resolved=int(np.sum(np.abs(R) > z * se)),
                         n_wrong_sign=int(wrong.sum()),
                         max_wrong_sign_response=float(np.max(sgn * R)),
                         bound_wrong_over_peak=floor / peak,
                         integrated_wrong_relative=float(
                             np.sum(np.maximum(sgn * R, 0.0)) /
                             np.sum(np.abs(R))),
                         bonferroni_z=float(z)))
        r = rows[-1]
        print(f"{drift:>9} {d:6g} {key:>4} {peak:10.3e} {floor:10.3e} "
              f"{floor/peak:11.4f} {r['n_wrong_sign']:6d} "
              f"{'<= ' + format(floor/peak, '.4f'):>20}")
        tag = f'highpower_{drift}_d{d:g}_k{key}'
        arch[f'{tag}_R'] = R
        arch[f'{tag}_se'] = se
    np.savez_compressed(OUT / 'sensitivity_highpower.npz', **arch)
    (OUT / 'sensitivity_highpower.json').write_text(json.dumps(dict(
        purpose='raise detection power so the wrong-sign statement is a bound, '
                'not a count at an unstated floor',
        seeds=S_HIGH, nu=NU, T=T, h=H, N=NPART, alpha=ALPHA,
        displaced_particle=int(i0), rows=rows,
        interpretation='no resolved wrong-sign point; any wrong-sign response '
                       'is bounded by the reported floor/peak ratio. This does '
                       'NOT prove monotonicity, it bounds any violation.',
        runtime_s=time.perf_counter() - t0), indent=1))
    print(f"\nelapsed {time.perf_counter()-t0:.1f}s   saved -> {OUT}")


if __name__ == '__main__':
    main()
