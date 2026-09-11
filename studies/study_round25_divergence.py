"""Round-25: archive the divergence fraction for spatially separated sign regions
on a larger, disjoint seed set, with an exact binomial interval.

WHY. The manuscript needs a statement of how often the switched and reflected
couplings part company when the sign regions start spatially separated. The
archived round-22 result is 11/40 (heat) and 13/40 (Burgers) on seeds 4000-4039.
A round-23 report quoted a pooled 200-seed figure that was never archived. This
driver produces one, on seeds 5000-5199, disjoint from every earlier seed set,
and reports Clopper-Pearson intervals rather than a bare fraction.

The one-sign configuration is rerun as the exact-identity control: the multiplier
is +1 for every history, so the two couplings must agree bit-for-bit on the full
state (positions, masses, carried velocities).

READING. A divergence fraction is a property of THIS configuration, N, h, nu and
horizon. It is evidence that spatial separation supplies no unconditional
identity; it is not an estimate of a universal probability.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.study_round22_corrections import paired_run

OUT = ROOT / 'output/round25_divergence_2026_09_10'
N, S, SEED0 = 2048, 200, 5000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"DIVERGENCE OF THE TWO COUPLINGS, N={N}, seeds {SEED0}-{SEED0+S-1}\n")
    print(f"   {'structure':>11} {'dynamics':>9} {'diverged':>10} {'fraction':>9} "
          f"{'95% Clopper-Pearson':>22} {'median first step':>18} {'max|state diff|':>16}")
    rows = []
    for cfg in ('one_sign', 'separated'):
        for drift in ('heat', 'burgers'):
            div, firsts, worst = 0, [], 0.0
            for s in range(S):
                st = paired_run(cfg, N, drift, SEED0 + s)
                d = max(float(np.max(np.abs(st['RAW'][k] - st['FULL'][k])))
                        for k in ('xA', 'mA', 'vA', 'xB', 'mB', 'vB'))
                worst = max(worst, d)
                if st['FULL']['first_disagree'] is not None:
                    div += 1; firsts.append(st['FULL']['first_disagree'])
            lo, hi = stats.binomtest(div, S).proportion_ci(method='exact')
            med = int(np.median(firsts)) if firsts else None
            rows.append(dict(structure=cfg, dynamics=drift, diverged=div, seeds=S,
                             fraction=div / S, ci=[float(lo), float(hi)],
                             median_first_step=med, max_state_diff=worst,
                             first_steps=[int(f) for f in firsts]))
            print(f"   {cfg:>11} {drift:>9} {str(div)+'/'+str(S):>10} {div/S:9.3f} "
                  f"[{lo:.3f}, {hi:.3f}]{'':>7} {str(med) if med is not None else '--':>18} "
                  f"{worst:16.2e}")
    pooled = sum(r['diverged'] for r in rows if r['structure'] == 'separated')
    lo, hi = stats.binomtest(pooled, 2 * S).proportion_ci(method='exact')
    print(f"\n   separated, both dynamics pooled: {pooled}/{2*S} = {pooled/(2*S):.3f} "
          f"[{lo:.3f}, {hi:.3f}]")
    print("   (pooling assumes the two dynamics share a rate; reported as a summary,")
    print("    not as an estimate of a universal probability)")
    json.dump(dict(N=N, seeds=S, seed0=SEED0, T=1.0, h=0.005, nu=0.1, rows=rows,
                   pooled_separated=dict(diverged=pooled, seeds=2 * S,
                                         fraction=pooled / (2 * S),
                                         ci=[float(lo), float(hi)]),
                   note='seeds disjoint from the round-22 set 4000-4039; exact '
                        'binomial intervals; configuration-specific'),
              open(OUT / 'divergence.json', 'w'), indent=1)
    print(f"\n   {time.perf_counter()-t0:.0f}s   saved -> {OUT}/divergence.json")


if __name__ == '__main__':
    main()
