"""Numerical checks of the final-stage identity (Section 3.4, Appendix A)."""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies import twopulse_reference as TP

OUT = ROOT / 'output/final_stage_identity_checks'
NU, H, T = 0.1, 0.005, 1.0
DRAWS, BATCHES = 12000, 40
GRID_PTS = 8001


def g(d, s):
    """E|d + s Z| for Z ~ N(0,1)."""
    d = np.asarray(d, float)
    if s == 0.0:
        return np.abs(d)
    r = d / s
    return s * (r * (2 * stats.norm.cdf(r) - 1) + 2 * stats.norm.pdf(r))


def prefinal(N, K, seed):
    """Evolve the pair with the switched coupling to just before the last stage."""
    x0, m0, ul, _ = TP.initialize(N)
    o = np.argsort(x0, kind='stable')
    xA, mA = x0[o].copy(), m0[o].copy(); xB, mB = xA.copy(), mA.copy()
    sd = np.sqrt(2 * NU * H); rng = np.random.default_rng(seed)
    vA = ul + np.cumsum(mA); vB = vA.copy()
    for k in range(K):
        xA = xA + vA * H; q = np.argsort(xA, kind='stable'); xA, mA = xA[q], mA[q]
        vA = ul + np.cumsum(mA)
        xB = xB + vB * H; q = np.argsort(xB, kind='stable'); xB, mB = xB[q], mB[q]
        vB = ul + np.cumsum(mB)
        if k == K - 1:
            return xA, mA, xB, mB, ul, rng
        z = rng.standard_normal(len(xA))
        xA = xA + sd * z
        xB = xB - sd * (np.sign(mA) * np.sign(mB)) * z
    raise RuntimeError


def closed_form(mA, mB, a, b, sd):
    un = np.sign(mA) * np.sign(mB) < 0
    d = a[un] - b[un]
    return 0.25 * float(np.sum(np.abs(mA[un] * mB[un]) * (g(d, 2 * sd) - np.abs(d)))), int(un.sum())


def measured(a, mA, b, mB, ul, rng, sd):
    """Paired Monte Carlo over the final stage; integrated variance of the pair mean."""
    lo = min(a.min(), b.min()) - 8 * sd - 0.5
    hi = max(a.max(), b.max()) + 8 * sd + 0.5
    xg = np.linspace(lo, hi, GRID_PTS); dx = float(xg[1] - xg[0])
    mult = -np.sign(mA) * np.sign(mB)
    per_batch = []
    for _ in range(BATCHES):
        UR = np.empty((DRAWS // BATCHES, GRID_PTS))
        US = np.empty_like(UR)
        for r in range(DRAWS // BATCHES):
            z = rng.standard_normal(len(a))
            xa = a + sd * z
            for name, U, xb in (('R', UR, b - sd * z), ('S', US, b + sd * mult * z)):
                xx = np.concatenate([xa, xb]); mm = np.concatenate([mA, mB]) * 0.5
                o = np.argsort(xx, kind='stable')
                cm = np.concatenate([[0.0], np.cumsum(mm[o])])
                U[r] = ul + cm[np.searchsorted(xx[o], xg, side='right')]
        per_batch.append(dx * (np.sum(UR.var(0, ddof=1)) - np.sum(US.var(0, ddof=1))))
    v = np.array(per_batch)
    m = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(len(v)))
    t = stats.t.ppf(0.975, len(v) - 1)
    return m, se, [m - t * se, m + t * se]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd = np.sqrt(2 * NU * H); K = round(T / H); t0 = time.perf_counter()
    print("PROPOSITION 2, CONDITIONAL ON THE PRE-FINAL STATE (whole-line variance)\n")
    print(f"   two-pulse Burgers, nu={NU}, h={H}, T={T}; {DRAWS} final-stage draws "
          f"in {BATCHES} batches, paired\n")
    print(f"   {'N':>6} {'seed':>5} {'unlike pairs':>13} {'closed form':>14} "
          f"{'measured':>14} {'95% CI of measured':>30} {'agrees':>8}")
    rows = []
    for N in (200, 400):
        for seed in (11, 12, 13):
            a, mA, b, mB, ul, rng = prefinal(N, K, seed)
            pred, nun = closed_form(mA, mB, a, b, sd)
            meas, se, ci = measured(a, mA, b, mB, ul, rng, sd)
            ok = ci[0] <= pred <= ci[1]
            print(f"   {N:6d} {seed:5d} {str(nun)+'/'+str(N):>13} {pred:14.6e} "
                  f"{meas:14.6e} [{ci[0]:.4e}, {ci[1]:.4e}] {str(ok):>8}")
            rows.append(dict(N=N, seed=seed, unlike_pairs=nun, closed_form=pred,
                             measured=meas, measured_se=se, ci=ci, agrees=bool(ok)))
    ok = sum(r['agrees'] for r in rows)
    print(f"\n   {ok}/{len(rows)} states agree within the Monte Carlo interval.")

    fs = json.load(open(ROOT / 'output/final_stage_switch/finalstep.json'))
    print("\n   SHARE OF THE TOTAL GAIN GUARANTEED BY THE FINAL STAGE")
    print("   (archived round-6 aggregates, whole-line variance, 64 seeds)\n")
    print(f"   {'problem':>10} {'N':>6} {'h':>8} {'predicted E[Delta]':>19} "
          f"{'V_RAW-V_FULL':>14} {'guaranteed share':>17} {'measured share':>15}")
    share = []
    for r in fs['rows']:
        if r['RAW_minus_FULL'] == 0:
            continue
        gs = r['E_Delta'] / r['RAW_minus_FULL']
        ms = r['final_step_share']
        share.append(dict(problem=r['problem'], N=r['N'], h=r['h'],
                          E_Delta=r['E_Delta'], E_Delta_se=r['E_Delta_se'],
                          RAW_minus_FULL=r['RAW_minus_FULL'],
                          guaranteed_share=gs, measured_share=ms))
        print(f"   {r['problem']:>10} {r['N']:6d} {r['h']:8.4f} {r['E_Delta']:19.4e} "
              f"{r['RAW_minus_FULL']:14.4e} {gs:17.4f} {ms:15.4f}")

    json.dump(dict(nu=NU, h=H, T=T, draws=DRAWS, batches=BATCHES, grid_points=GRID_PTS,
                   conditional_checks=rows, share=share,
                   statement='V_reflect - V_switch = (1/4) sum_{unlike} |m_A m_B| '
                             '[g_{2 sigma}(d) - |d|], conditional on the pre-final '
                             'paired state, whole-line integrated variance of the pair mean',
                   note='the Monte Carlo estimate uses a bounded window containing every '
                        'realised particle plus 8 sigma; the identity is whole-line. '
                        'Refining the window grid from 6001 to 16001 points moved the '
                        'estimate by 0.02% at the checked state, so the residual gap at '
                        '4000 draws was Monte Carlo error, not quadrature error.'),
              open(OUT / 'finalstage.json', 'w'), indent=1)
    print(f"\n   {time.perf_counter()-t0:.0f}s   saved -> {OUT}/finalstage.json")


if __name__ == '__main__':
    main()
