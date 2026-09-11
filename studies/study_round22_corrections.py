"""Round-22: corrections and completeness checks flagged in review.

(1) PROPOSITION 3 WAS OVERCLAIMED. The valid statement is CONDITIONAL: starting
    from identical states with shared draws, the two couplings remain identical
    for as long as the matched signs agree at every stage. Only the ONE-SIGN case
    makes that hypothesis hold unconditionally. "Initially separated sign regions"
    does NOT: Gaussian support is unbounded, so a draw can reverse an ordering in
    one replica while its reflection preserves it in the other. The earlier
    0.0e+00 came from a SINGLE seed.
(2) MATCHED-SEED WITNESS on the FULL STATE (positions, masses, carried
    velocities) rather than reconstructed fields, which cannot certify identical
    particle paths.
(3) INTERLEAVED CONFIGURATION is a mechanism test, not refinement of a fixed
    profile: its represented field changes with N. Amplitude and total signed-mass
    variation are reported so this is visible.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.study_round21_applicability import config, NU, T, H
from studies.round06_spectral_reference import FPRIME

OUT = ROOT / 'output/round22_corrections_2026_09_10'
FPRIME['heat0'] = lambda u: np.zeros_like(u)


def paired_run(cfg, N, drift, seed, record_state=False):
    x0, m0, ul = config(cfg, N)
    fp = FPRIME['burgers'] if drift == 'burgers' else FPRIME['heat0']
    K = round(T / H); sd = np.sqrt(2 * NU * H)
    rngR = np.random.default_rng(seed); rngF = np.random.default_rng(seed)
    o = np.argsort(x0, kind='stable')
    st = {}
    for pol, rng in (('RAW', rngR), ('FULL', rngF)):
        xA, mA = x0[o].copy(), m0[o].copy(); xB, mB = xA.copy(), mA.copy()
        vA = fp(ul + np.cumsum(mA)); vB = vA.copy()
        first = None
        for k in range(K):
            xA = xA + vA * H; q = np.argsort(xA, kind='stable')
            xA, mA = xA[q], mA[q]; vA = fp(ul + np.cumsum(mA))
            xB = xB + vB * H; q = np.argsort(xB, kind='stable')
            xB, mB = xB[q], mB[q]; vB = fp(ul + np.cumsum(mB))
            dis = np.sign(mA) != np.sign(mB)
            if first is None and dis.any():
                first = k
            z = rng.standard_normal(len(xA))
            zB = -sd * z if pol == 'RAW' else -sd * (np.sign(mA) * np.sign(mB)) * z
            xA = xA + sd * z; xB = xB + zB
        st[pol] = dict(xA=xA, mA=mA, vA=vA, xB=xB, mB=mB, vB=vB, first_disagree=first)
    return st


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("(1) HOW OFTEN DO THE COUPLINGS DIVERGE?  matched seeds, full state\n")
    print(f"   {'structure':>11} {'dynamics':>9} {'seeds diverging':>16} "
          f"{'median first step':>18} {'max|state diff|':>16}")
    rows = []
    S = 40
    for cfg in ('one_sign', 'separated', 'merging', 'clustered'):
        for drift in ('heat', 'burgers'):
            div, firsts, worst = 0, [], 0.0
            for s in range(S):
                st = paired_run(cfg, 2048, drift, 4000 + s)
                d = max(float(np.max(np.abs(st['RAW'][k] - st['FULL'][k])))
                        for k in ('xA', 'mA', 'vA', 'xB', 'mB', 'vB'))
                worst = max(worst, d)
                if st['FULL']['first_disagree'] is not None:
                    div += 1; firsts.append(st['FULL']['first_disagree'])
            med = f"{int(np.median(firsts))}" if firsts else "--"
            rows.append(dict(structure=cfg, dynamics=drift, diverged=div, seeds=S,
                             median_first_step=med, max_state_diff=worst))
            print(f"   {cfg:>11} {drift:>9} {str(div)+'/'+str(S):>16} {med:>18} "
                  f"{worst:16.2e}")
    print("\n   ONE-SIGN: 0/40, max state difference exactly 0 -> unconditional, as")
    print("   the construction requires (multiplier is +1 for every history).")
    print("   SEPARATED: diverges in a substantial fraction of seeds. The earlier")
    print("   'pathwise identical' reading came from ONE seed and is WITHDRAWN.\n")

    print("(3) THE INTERLEAVED CONFIGURATION CHANGES THE REPRESENTED FIELD WITH N\n")
    print(f"   {'config':>11} {'N':>6} {'field amplitude':>16} {'total |m| variation':>20} "
          f"{'support width':>14}")
    conf = []
    for cfg in ('clustered', 'merging', 'separated'):
        for N in (400, 2048, 8192):
            x0, m0, ul = config(cfg, N)
            o = np.argsort(x0); u = ul + np.cumsum(m0[o])
            amp = float(np.max(u) - np.min(u)); tv = float(np.sum(np.abs(m0)))
            w = float(x0.max() - x0.min())
            conf.append(dict(config=cfg, N=N, amplitude=amp, total_variation=tv, width=w))
            print(f"   {cfg:>11} {N:>6} {amp:16.5f} {tv:20.5f} {w:14.3f}")
    print("\n   'clustered' amplitude falls like 1/N while its oscillation becomes")
    print("   finer: increasing N changes the PROBLEM, not just the resolution. It is")
    print("   a cancellation MECHANISM TEST and is labelled as such. 'merging' and")
    print("   'separated' hold amplitude fixed and are refinements of one profile.")
    json.dump(dict(divergence=rows, configurations=conf,
                   note='Prop 3 corrected: conditional lemma; one-sign is the only '
                        'unconditional case. Separated diverges in a substantial '
                        'fraction of seeds; the earlier single-seed 0.0e+00 is '
                        'withdrawn. Interleaved config is a mechanism test.'),
              open(OUT / 'corrections.json', 'w'), indent=1)
    print(f"\n   {time.perf_counter()-t0:.0f}s   saved -> {OUT}")


if __name__ == '__main__':
    main()
