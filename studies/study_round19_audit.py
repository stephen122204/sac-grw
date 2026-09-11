"""Round-19 step 1: audit the Round-18 prediction claim.

FOUR THREATS, each tested rather than argued:

 (T1) SHARED DENOMINATOR. Round 18 regressed log(V_FULL/V_RAW) on log(V_RAW),
      reusing the same noisy V_RAW on both sides. With Vhat = V(1+eps),
          Cov(log rhat, log Vhat_RAW) = Cov_true - Var(log(1+eps)),
      so the slope is biased NEGATIVE even if the truth is a constant ratio.
      For S replicates Var(log Vhat) ~ 2/(S-1). Fix: estimate the denominator
      from an INDEPENDENT batch of RAW replicates.

 (T2) SPATIAL DEPENDENCE. ~200 grid probes are not 200 independent observations.
      Fix: whole-replicate resampling (resample replicates, not probes) and a
      contiguous spatial block bootstrap.

 (T3) TAIL EXCLUSION. The >1e-3*max cut on V_RAW was chosen once, informally.
      Fix: report the fit across a range of cuts.

 (T4) IS A POWER LAW EVEN NEEDED? Compare against the simplest alternative,
      a constant reduction V_FULL = c V_RAW, on held-out data.

Cheap cell (N=2048, h=0.005) so the audit can afford three independent batches.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import reconstruct_cumulative_field
from studies.study_round06_cubic import advance_pair_flux, XG
from studies.round06_spectral_reference import solve, evaluate
from studies import twopulse_reference as TP

OUT = ROOT / 'output/round19_audit_2026_09_10'
NU, T, H, N_PART, REPS = 0.1, 1.0, 0.005, 2048, 64


def batch(arm, key, reps, init, K):
    x0, m0, ul = init
    P = []
    for r in range(reps):
        s = np.random.default_rng(np.random.SeedSequence([key, r]))
        A, B, u = advance_pair_flux((x0, m0, ul), NU, H, K, s, arm, 'burgers')
        P.append(0.5 * (reconstruct_cumulative_field(A[0], A[1], u, XG)
                        + reconstruct_cumulative_field(B[0], B[1], u, XG)))
    return np.array(P)


def fit(lv, lr):
    p, c = np.polyfit(lv, lr, 1)
    return p, c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    init = TP.initialize(N_PART)[:3]
    K = round(T / H)
    t0 = time.perf_counter()
    # three INDEPENDENT batches: two RAW (1901, 1902) and one FULL (1903)
    R1 = batch('RAW', 1901, REPS, init, K)
    R2 = batch('RAW', 1902, REPS, init, K)
    F1 = batch('FULL', 1903, REPS, init, K)
    print(f"three independent batches of {REPS} replicates each "
          f"({time.perf_counter()-t0:.0f}s)\n")
    V1, V2, VF = (A.var(0, ddof=1) for A in (R1, R2, F1))

    print("T1. SHARED DENOMINATOR")
    var_log = 2.0 / (REPS - 1)
    print(f"    Var(log Vhat) for S={REPS} is about 2/(S-1) = {var_log:.4f}")
    m = V1 > 1e-3 * V1.max()
    vl = np.var(np.log(V1[m]))
    print(f"    spatial Var(log V_RAW) over {int(m.sum())} probes = {vl:.3f}")
    print(f"    => slope bias if the truth were a CONSTANT ratio: "
          f"{-var_log/(vl+var_log):+.4f}")
    ps = fit(np.log(V1[m]), np.log((VF / V1)[m]))[0]      # shared (round-18 style)
    pi = fit(np.log(V2[m]), np.log((VF / V1)[m]))[0]      # independent denominator
    pi2 = fit(np.log(V1[m]), np.log((VF / V2)[m]))[0]     # the other pairing
    print(f"    slope, SHARED denominator (round-18 style) : {ps:+.3f}")
    print(f"    slope, INDEPENDENT denominator batch       : {pi:+.3f}")
    print(f"    slope, independent, denominators swapped   : {pi2:+.3f}")
    # placebo: RAW vs RAW should give slope ~ the artefact value, not -0.47
    pp = fit(np.log(V1[m]), np.log((V2 / V1)[m]))[0]
    pp2 = fit(np.log(V2[m]), np.log((V2 / V1)[m]))[0]
    print(f"    PLACEBO log(V_RAW2/V_RAW1) on log(V_RAW1)  : {pp:+.3f}  <- pure artefact")
    print(f"    PLACEBO, independent x-axis                : {pp2:+.3f}")

    print("\nT2. SPATIAL DEPENDENCE (independent-denominator slope)")
    rng = np.random.default_rng(1904)
    # whole-replicate resampling: resample replicates, recompute variances
    bs = []
    for _ in range(1000):
        i1 = rng.integers(0, REPS, REPS); i2 = rng.integers(0, REPS, REPS)
        iF = rng.integers(0, REPS, REPS)
        a = R1[i1].var(0, ddof=1); b = R2[i2].var(0, ddof=1); f = F1[iF].var(0, ddof=1)
        mm = a > 1e-3 * a.max()
        bs.append(fit(np.log(b[mm]), np.log((f / a)[mm]))[0])
    bs = np.array(bs)
    print(f"    whole-replicate bootstrap slope: {pi:+.3f} "
          f"[{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
    # contiguous spatial block bootstrap on probes
    idx = np.flatnonzero(m); nb = 10
    blocks = np.array_split(idx, nb)
    bb = []
    for _ in range(1000):
        pick = np.concatenate([blocks[j] for j in rng.integers(0, nb, nb)])
        bb.append(fit(np.log(V2[pick]), np.log((VF / V1)[pick]))[0])
    bb = np.array(bb)
    print(f"    spatial block bootstrap ({nb} blocks): "
          f"[{np.percentile(bb,2.5):+.3f}, {np.percentile(bb,97.5):+.3f}]")

    print("\nT3. TAIL-EXCLUSION SENSITIVITY (independent denominator)")
    print(f"    {'cut (frac of max V_RAW)':>26} {'n probes':>9} {'slope':>8}")
    for cut in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        mm = (V1 > cut * V1.max()) & (V2 > 0) & (VF > 0)
        print(f"    {cut:>26.0e} {int(mm.sum()):>9} "
              f"{fit(np.log(V2[mm]), np.log((VF/V1)[mm]))[0]:+8.3f}")

    print("\nT4. IS A POWER LAW NEEDED? predict V_FULL from V_RAW, held out")
    # fit on batch pair (V2 -> VF) ... predict on a fresh held-out pair
    F2 = batch('FULL', 1905, REPS, init, K)
    R3 = batch('RAW', 1906, REPS, init, K)
    VF2, V3 = F2.var(0, ddof=1), R3.var(0, ddof=1)
    mtr = V1 > 1e-3 * V1.max(); mte = V3 > 1e-3 * V3.max()
    # model 1: constant reduction  V_FULL = c V_RAW
    c1 = np.exp(np.mean(np.log(VF[mtr]) - np.log(V2[mtr])))
    # model 2: power law           V_FULL = C V_RAW^q
    q, lc = np.polyfit(np.log(V2[mtr]), np.log(VF[mtr]), 1)
    print(f"    fitted on batches (V2 -> VF): constant c = {c1:.4f}; "
          f"power law q = {q:.3f}, C = {np.exp(lc):.4f}")
    for name, pred in (('constant reduction', c1 * V3[mte]),
                       ('power law', np.exp(lc) * V3[mte] ** q)):
        e = np.log(VF2[mte]) - np.log(pred)
        print(f"    held-out (V3 -> VF2) {name:<20} median |log err| = "
              f"{np.median(np.abs(e)):.4f}  RMS = {np.sqrt(np.mean(e**2)):.4f}")
    json.dump(dict(N=N_PART, reps=REPS, slope_shared=float(ps),
                   slope_independent=float(pi), slope_independent_swapped=float(pi2),
                   placebo_shared=float(pp), placebo_independent=float(pp2),
                   artefact_if_constant=float(-var_log / (vl + var_log)),
                   whole_replicate_ci=[float(np.percentile(bs, 2.5)),
                                       float(np.percentile(bs, 97.5))],
                   block_ci=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
                   const_c=float(c1), power_q=float(q)),
              open(OUT / 'audit.json', 'w'), indent=1)
    print(f"\n    total {time.perf_counter()-t0:.0f}s   saved -> {OUT}")


if __name__ == '__main__':
    main()
