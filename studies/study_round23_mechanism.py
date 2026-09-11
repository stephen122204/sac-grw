"""Round-23: does spatial-rank matching add anything over within-sign matching,
and where does that addition weaken?

PREDECLARED MECHANISM AND PREDICTION (written before running).

  WITHIN pairs the k-th positive particle of A with the k-th positive of B, and
  likewise for negatives. It never produces an unlike-sign matched pair. But when
  the sign classes are spatially interleaved, the k-th positive of A can sit far
  from the k-th positive of B in space.
  FULL pairs by GLOBAL rank, so matched particles are spatially adjacent, at the
  price of unlike-sign pairs -- which it then synchronises rather than reflects.

  Proposition 2 says the reflected-pair gain depends on the matched separation
  relative to sigma. So:

  PREDICTION: FULL beats WITHIN most where the sign classes INTERLEAVE (WITHIN's
  matched pairs are then spatially distant), and least where the sign classes are
  SPATIALLY SEPARATED (there rank order groups by sign, so the two matchings
  nearly coincide).

  FALSIFIABLE: if FULL/WITHIN is flat across the ladder below, or ordered the
  other way, the mechanism is wrong.

LADDER OF PROFILES (fixed sources; sign-region count is the controlled variable)
  P1 single bump      2 sign regions, separated       -> weak benefit predicted
  P2 two-pulse        4 sign regions                  -> intermediate (the standard)
  P3 oscillatory      ~8 sign regions, interleaved    -> strong benefit predicted
Each observed at T = 0.25 (early), 1.0 (interaction), 2.0 (late).

Also measured: the mean matched-pair separation under each policy, which is the
quantity the mechanism claims is doing the work.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import reconstruct_cumulative_field
from studies.study_round06_cubic import advance_pair_flux, XG
from studies.round06_spectral_reference import solve, evaluate
from studies.study_round19_transfer import init_from

OUT = ROOT / 'output/round23_mechanism_2026_09_10'
NU, H, N, REPS = 0.1, 0.005, 2048, 48
ARM = {'RAW': 2, 'WITHIN': 4, 'FULL': 3}
TIMES = (0.25, 1.0, 2.0)


def u0_bump(x):    return 0.8 * np.exp(-(x + 0.2) ** 2 / (2 * 0.45 ** 2))
def u0_two(x):     return (0.8 * np.exp(-(x + 0.7) ** 2 / (2 * 0.5 ** 2))
                           - 0.5 * np.exp(-(x - 0.4) ** 2 / (2 * 0.3 ** 2)))
def u0_osc(x):     return 0.45 * np.sin(3.0 * x) * np.exp(-x ** 2 / (2 * 1.1 ** 2))

PROFILES = (('P1 single bump', u0_bump), ('P2 two-pulse', u0_two),
            ('P3 oscillatory', u0_osc))


def sign_regions(u0fn):
    xs = np.linspace(-6, 6, 40001); g = np.gradient(u0fn(xs), xs[1] - xs[0])
    m = np.abs(g) > 1e-6 * np.max(np.abs(g))
    s = np.sign(g[m])
    return int(np.sum(np.diff(s) != 0)) + 1


def batch(arm, init, T, key):
    K = round(T / H); x0, m0, ul = init; P = []
    for r in range(REPS):
        s = np.random.default_rng(np.random.SeedSequence([key, ARM[arm], r]))
        A, B, u = advance_pair_flux((x0, m0, ul), NU, H, K, s, arm, 'burgers')
        P.append(0.5 * (reconstruct_cumulative_field(A[0], A[1], u, XG)
                        + reconstruct_cumulative_field(B[0], B[1], u, XG)))
    return np.array(P)


def matched_separation(init, T, policy, seed=11):
    """Mean |a_i - b_i| over matched pairs at the final stage, by policy."""
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
        if k == K - 1:
            if policy == 'FULL':
                return float(np.mean(np.abs(xA - xB)))
            sep = []
            for sg in (1.0, -1.0):
                ia = np.flatnonzero(np.sign(mA) == sg); ib = np.flatnonzero(np.sign(mB) == sg)
                n = min(len(ia), len(ib))
                if n: sep.append(np.abs(xA[ia[:n]] - xB[ib[:n]]))
            return float(np.mean(np.concatenate(sep))) if sep else np.nan
        z = rng.standard_normal(len(xA))
        zB = -sd * z if policy != 'FULL' else -sd * (np.sign(mA) * np.sign(mB)) * z
        xA = xA + sd * z; xB = xB + zB
    return np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dx = float(XG[1] - XG[0]); rng = np.random.default_rng(2301); t0 = time.perf_counter()
    print("PREDECLARED PREDICTION: FULL/WITHIN improves as sign regions interleave;")
    print("it should be weakest for the single bump (2 separated sign regions).\n")
    print(f"   {'profile':>16} {'sign regions':>13} {'T':>5} "
          f"{'WITHIN/RAW':>11} {'FULL/RAW':>10} {'FULL/WITHIN':>22} "
          f"{'sep FULL/WITHIN':>16}")
    rows = []
    for pname, u0fn in PROFILES:
        nreg = sign_regions(u0fn)
        init = init_from(u0fn, N)
        for T in TIMES:
            _, _, uh, kk = solve('burgers', T, NU, M=2048, dt=1e-4, u0=u0fn)
            ref = evaluate(uh, kk, XG, 2048)
            V = {a: batch(a, init, T, 2300) for a in ARM}
            iv = {a: float(dx * np.sum(V[a].var(0, ddof=1))) for a in V}
            idx = rng.integers(0, REPS, (3000, REPS))
            bb = np.array([dx * np.sum(V['FULL'][i].var(0, ddof=1))
                           / (dx * np.sum(V['WITHIN'][j].var(0, ddof=1)))
                           for i, j in zip(idx, rng.integers(0, REPS, (3000, REPS)))])
            lo, hi = np.percentile(bb, [2.5, 97.5])
            sF = matched_separation(init, T, 'FULL'); sW = matched_separation(init, T, 'WITHIN')
            print(f"   {pname:>16} {nreg:>13} {T:>5.2f} {iv['WITHIN']/iv['RAW']:11.4f} "
                  f"{iv['FULL']/iv['RAW']:10.4f} "
                  f"{iv['FULL']/iv['WITHIN']:8.4f} [{lo:.3f},{hi:.3f}] "
                  f"{sF/sW:16.3f}")
            rows.append(dict(profile=pname, sign_regions=nreg, T=T,
                             within_over_raw=iv['WITHIN'] / iv['RAW'],
                             full_over_raw=iv['FULL'] / iv['RAW'],
                             full_over_within=iv['FULL'] / iv['WITHIN'],
                             ci=[float(lo), float(hi)],
                             sep_full=sF, sep_within=sW, sep_ratio=sF / sW))
    json.dump(dict(N=N, h=H, nu=NU, reps=REPS, times=list(TIMES), rows=rows,
                   prediction='FULL/WITHIN improves with sign-region interleaving; '
                              'weakest for 2 separated sign regions'),
              open(OUT / 'mechanism.json', 'w'), indent=1)
    print(f"\n   {time.perf_counter()-t0:.0f}s   saved -> {OUT}")


if __name__ == '__main__':
    main()
