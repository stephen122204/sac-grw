"""Fraction of seeds on which switched and reflected pairing diverge for separated sign regions (Section 3.5)."""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.principal_burgers_runs import paired_run

OUT = ROOT / 'output/coupling_divergence_seeds'
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
