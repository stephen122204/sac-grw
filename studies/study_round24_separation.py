"""Round-24: repair the matched-separation diagnostic.

DEFECT FOUND IN REVIEW (confirmed in our own source, round-23 line 24):
`matched_separation` evolved the WITHIN arm with RAW *reflected* coupling and
applied within-sign matching only when computing the final separation. The
separation ratios therefore did not describe WITHIN histories. The variance
comparisons used the correct coupling and are unaffected.

TWO SEPARATE QUESTIONS, kept apart because they are not the same comparison:

 (A) MATCHED-STATE. At ONE paired state, what separation does each MATCHING RULE
     produce? Rank matching pairs (i,i); within-sign matching pairs the k-th
     particle of sign s in A with the k-th of sign s in B. This isolates the
     matching rule from the history and is the comparison the mechanism is about.

 (B) OWN-HISTORY. Evolve each policy with ITS OWN correct coupling and measure the
     separation its own matching produces. This is what actually happens, but the
     two arms then have different histories, so it is an association, not an
     isolation.

Both are computed over REPLICATES, not one seed.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.study_round23_mechanism import (u0_bump, u0_two, u0_osc, PROFILES,
                                             sign_regions, NU, H, N)
from studies.study_round19_transfer import init_from

OUT = ROOT / 'output/round24_separation_2026_09_10'
REPS, TIMES = 24, (0.25, 1.0, 2.0)


def sep_rank(xA, xB):
    return float(np.mean(np.abs(xA - xB)))


def sep_within(xA, mA, xB, mB):
    s = []
    for sgn in (1.0, -1.0):
        ia = np.flatnonzero(np.sign(mA) == sgn); ib = np.flatnonzero(np.sign(mB) == sgn)
        q = min(len(ia), len(ib))
        if q: s.append(np.abs(xA[ia[:q]] - xB[ib[:q]]))
    return float(np.mean(np.concatenate(s))) if s else np.nan


def evolve(init, T, policy, seed, collect_final=True):
    """Correct coupling for every policy, including WITHIN."""
    K = round(T / H); x0, m0, ul = init
    o = np.argsort(x0, kind='stable')
    xA, mA = x0[o].copy(), m0[o].copy(); xB, mB = xA.copy(), mA.copy()
    sd = np.sqrt(2 * NU * H); rng = np.random.default_rng(seed)
    fp = lambda u: u
    vA = fp(ul + np.cumsum(mA)); vB = vA.copy()
    for k in range(K):
        xA = xA + vA * H; q = np.argsort(xA, kind='stable'); xA, mA = xA[q], mA[q]
        vA = fp(ul + np.cumsum(mA))
        xB = xB + vB * H; q = np.argsort(xB, kind='stable'); xB, mB = xB[q], mB[q]
        vB = fp(ul + np.cumsum(mB))
        if k == K - 1 and collect_final:
            return xA.copy(), mA.copy(), xB.copy(), mB.copy()
        z = rng.standard_normal(len(xA))
        if policy == 'FULL':
            zB = -sd * (np.sign(mA) * np.sign(mB)) * z
        elif policy == 'WITHIN':
            zB = np.empty(len(xA))
            for sgn in (1.0, -1.0):
                ia = np.flatnonzero(np.sign(mA) == sgn)
                ib = np.flatnonzero(np.sign(mB) == sgn)
                qn = min(len(ia), len(ib))
                zB[ib[:qn]] = -sd * z[ia[:qn]]
                if len(ib) > qn:
                    zB[ib[qn:]] = sd * rng.standard_normal(len(ib) - qn)
        else:
            zB = -sd * z
        xA = xA + sd * z; xB = xB + zB
    return xA, mA, xB, mB


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter(); rows = []
    print("(A) MATCHED-STATE: both rules applied to the SAME paired state.")
    print("    Isolates the matching rule. State reached with FULL coupling.\n")
    print(f"   {'profile':>16} {'regions':>8} {'T':>5} {'rank sep':>10} "
          f"{'within sep':>11} {'ratio':>19}")
    for pname, u0fn in PROFILES:
        nreg = sign_regions(u0fn); init = init_from(u0fn, N)
        for T in TIMES:
            rr = []
            for r in range(REPS):
                xA, mA, xB, mB = evolve(init, T, 'FULL', 2400 + r)
                rr.append(sep_rank(xA, xB) / sep_within(xA, mA, xB, mB))
            rr = np.array(rr); m = rr.mean()
            se = rr.std(ddof=1) / np.sqrt(REPS)
            xA, mA, xB, mB = evolve(init, T, 'FULL', 2400)
            print(f"   {pname:>16} {nreg:>8} {T:>5.2f} {sep_rank(xA,xB):10.4f} "
                  f"{sep_within(xA,mA,xB,mB):11.4f} "
                  f"{m:7.4f} [{m-1.96*se:.3f},{m+1.96*se:.3f}]")
            rows.append(dict(kind='matched_state', profile=pname, regions=nreg, T=T,
                             ratio=float(m), se=float(se)))
    print("\n(B) OWN-HISTORY: each policy evolved with ITS OWN correct coupling.")
    print("    The two arms have different histories, so this is an association.\n")
    print(f"   {'profile':>16} {'T':>5} {'FULL own sep':>13} {'WITHIN own sep':>15} "
          f"{'ratio':>19}")
    for pname, u0fn in PROFILES:
        init = init_from(u0fn, N)
        for T in TIMES:
            rr = []
            for r in range(REPS):
                a1, m1, b1, n1 = evolve(init, T, 'FULL', 2500 + r)
                a2, m2, b2, n2 = evolve(init, T, 'WITHIN', 2600 + r)
                rr.append(sep_rank(a1, b1) / sep_within(a2, m2, b2, n2))
            rr = np.array(rr); m = rr.mean(); se = rr.std(ddof=1) / np.sqrt(REPS)
            a1, m1, b1, n1 = evolve(init, T, 'FULL', 2500)
            a2, m2, b2, n2 = evolve(init, T, 'WITHIN', 2600)
            print(f"   {pname:>16} {T:>5.2f} {sep_rank(a1,b1):13.4f} "
                  f"{sep_within(a2,m2,b2,n2):15.4f} "
                  f"{m:7.4f} [{m-1.96*se:.3f},{m+1.96*se:.3f}]")
            rows.append(dict(kind='own_history', profile=pname, T=T,
                             ratio=float(m), se=float(se)))
    json.dump(dict(reps=REPS, times=list(TIMES), rows=rows,
                   defect='round-23 evolved WITHIN with reflected coupling and '
                          'applied within-sign matching only at the final step; '
                          'variance comparisons were unaffected'),
              open(OUT / 'separation.json', 'w'), indent=1)
    print(f"\n   {time.perf_counter()-t0:.0f}s   saved -> {OUT}")


if __name__ == '__main__':
    main()
