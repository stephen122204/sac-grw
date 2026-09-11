"""Round-18: does the exact final-stage identity predict WHERE the benefit accrues?

The measured benefit is strongly concentrated in space. We already have an exact
statement about ONE step (Corollary 3): conditional on the pre-final paired state,
the pair-mean variance difference is

    V_RAW(x) - V_FULL(x) = (1/2) sum_i m_Ai m_Bi [ Cov_i^RAW(x) - Cov_i^FULL(x) ],

where for matched pair i at pre-diffusion positions (a_i, b_i), with sigma the
one-step Brownian scale,

    Cov_i(x) = P(A_i <= x, B_i <= x) - Phi((x-a_i)/sigma) Phi((x-b_i)/sigma),
    RAW   : B_i = b_i - sigma Z_i  ->  P_joint = [Phi((x-a_i)/s) - Phi((b_i-x)/s)]^+
    FULL, unlike sign: B_i = b_i + sigma Z_i -> P_joint = Phi(min((x-a_i)/s,(x-b_i)/s))
    FULL, like sign  : identical to RAW.

Each pair's contribution is LOCALISED within a few sigma of its own position --
the joint probability factorises away from there -- so this predicts a spatial
PROFILE, not just a total.

HYPOTHESIS: the shape of that one-step profile predicts the shape of the measured
FULL-HISTORY benefit, even though its magnitude is only a few percent of the
total. FAVOURABLE = high rank correlation between profiles. UNFAVOURABLE = no
correlation, which would mean the spatial structure comes from history effects the
final-stage identity cannot see, and the mechanism claim must be dropped.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.round08_estimators import advance_pair_instrumented
from studies.study_round06_cubic import XG
from studies import twopulse_reference as TP

OUT = ROOT / 'output/round18_observable_2026_09_10'
NU, T, H, N_PART = 0.1, 1.0, 0.005, 8192


def profile(a, b, mA, mB, sigma, xg, policy):
    """Exact per-pair covariance profile, summed. Vectorised over pairs in chunks."""
    out = np.zeros(len(xg))
    for lo in range(0, len(a), 512):
        hi = min(lo + 512, len(a))
        tA = (xg[None, :] - a[lo:hi, None]) / sigma
        tB = (xg[None, :] - b[lo:hi, None]) / sigma
        PA, PB = ndtr(tA), ndtr(tB)
        if policy == 'FULL':
            unlike = (np.sign(mA[lo:hi]) * np.sign(mB[lo:hi]) < 0)[:, None]
            Pj = np.where(unlike, ndtr(np.minimum(tA, tB)),
                          np.clip(PA - ndtr(-tB), 0.0, None))
        else:
            Pj = np.clip(PA - ndtr(-tB), 0.0, None)
        out += ((mA[lo:hi] * mB[lo:hi])[:, None] * (Pj - PA * PB)).sum(0)
    return 0.5 * out


def main():
    x0, m0, ul, _ = TP.initialize(N_PART)
    K = round(T / H); sigma = float(np.sqrt(2 * NU * H))
    # one FULL run stopped one step short: its pre-final paired state
    o = advance_pair_instrumented((x0, m0, ul), NU, H, K, np.random.default_rng(1805),
                                  'FULL', 'burgers')
    a, mA, b, mB = o['YA'], o['pre_mA'], o['YB'], o['pre_mB']
    unlike = int(np.sum(np.sign(mA) * np.sign(mB) < 0))
    print(f"pre-final paired state: N={len(a)}, unlike-sign matched pairs = {unlike} "
          f"({100*unlike/len(a):.1f}%), sigma = {sigma:.4f}\n")

    dV_pred = profile(a, b, mA, mB, sigma, XG, 'RAW') - profile(a, b, mA, mB, sigma, XG, 'FULL')

    z = np.load(OUT / 'fields.npz')
    xg = z['x']; dx = float(xg[1] - xg[0])
    PMF = 0.5 * (z['FULL_A'].astype(np.float64) + z['FULL_B'].astype(np.float64))
    PMR = 0.5 * (z['RAW_A'].astype(np.float64) + z['RAW_B'].astype(np.float64))
    dV_meas = PMR.var(0, ddof=1) - PMF.var(0, ddof=1)

    m = PMR.var(0, ddof=1) > 1e-3 * PMR.var(0, ddof=1).max()
    sp = stats.spearmanr(dV_pred[m], dV_meas[m]).statistic
    pe = stats.pearsonr(dV_pred[m], dV_meas[m]).statistic
    print("DOES THE ONE-STEP PROFILE PREDICT THE FULL-HISTORY PROFILE?")
    print(f"   Spearman(predicted dV(x), measured dV(x)) = {sp:+.3f}")
    print(f"   Pearson  (predicted dV(x), measured dV(x)) = {pe:+.3f}   ({int(m.sum())} probes)")
    print(f"   integrated predicted (one step)  = {dx*np.sum(dV_pred):.4e}")
    print(f"   integrated measured (full history) = {dx*np.sum(dV_meas):.4e}")
    print(f"   one-step share of the total        = {np.sum(dV_pred)/np.sum(dV_meas):.4f}")
    # where do the two profiles put their mass?
    def centroid_iqr(w, xs):
        w = np.maximum(w, 0); c = np.cumsum(w) / w.sum()
        return (np.sum(w * xs) / w.sum(), np.interp(0.25, c, xs), np.interp(0.75, c, xs))
    cp = centroid_iqr(dV_pred[m], xg[m]); cm = centroid_iqr(dV_meas[m], xg[m])
    print(f"\n   predicted benefit centroid {cp[0]:+.3f}, central 50% in [{cp[1]:+.3f},{cp[2]:+.3f}]")
    print(f"   measured  benefit centroid {cm[0]:+.3f}, central 50% in [{cm[1]:+.3f},{cm[2]:+.3f}]")
    json.dump(dict(spearman=float(sp), pearson=float(pe),
                   one_step_share=float(np.sum(dV_pred) / np.sum(dV_meas)),
                   unlike_pairs=unlike, N=N_PART, sigma=sigma,
                   predicted_centroid=cp[0], measured_centroid=cm[0],
                   predicted_iqr=[cp[1], cp[2]], measured_iqr=[cm[1], cm[2]],
                   x=xg.tolist(), dV_pred=dV_pred.tolist(), dV_meas=dV_meas.tolist()),
              open(OUT / 'mechanism.json', 'w'), indent=1)
    print(f"\n   saved -> {OUT}/mechanism.json")


if __name__ == '__main__':
    main()
