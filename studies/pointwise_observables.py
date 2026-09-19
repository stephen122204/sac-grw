"""Pointwise variance ratios and nonlinear observables (Section 4.6)."""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradient_particles import reconstruct_cumulative_field
from studies.cubic_flux_transfer import advance_pair_flux, XG
from studies.spectral_reference import solve, evaluate
from studies import twopulse_reference as TP

OUT = ROOT / 'output/pointwise_observables'
NU, T, H, N_PART, REPS, KEY = 0.1, 1.0, 0.005, 8192, 48, 1803
ARM_ID = {'RAW': 2, 'FULL': 3}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, uh, kk = solve('burgers', T, NU, M=2048, dt=1e-4)
    ref = evaluate(uh, kk, XG, 2048); dx = float(XG[1] - XG[0]); K = round(T / H)
    x0, m0, ul, _ = TP.initialize(N_PART)
    t0 = time.perf_counter(); A = {}
    for arm in ('RAW', 'FULL'):
        fa, fb = [], []
        for r in range(REPS):
            s = np.random.default_rng(np.random.SeedSequence([KEY, ARM_ID[arm], r]))
            P, Qq, u = advance_pair_flux((x0, m0, ul), NU, H, K, s, arm, 'burgers')
            fa.append(reconstruct_cumulative_field(P[0], P[1], u, XG))
            fb.append(reconstruct_cumulative_field(Qq[0], Qq[1], u, XG))
        A[arm] = (np.array(fa), np.array(fb))
    print(f"generated {REPS} fresh replicates per arm in {time.perf_counter()-t0:.0f}s "
          f"(key {KEY}, unused)\n")

    rng = np.random.default_rng(1804)
    def rci(vf, vr):
        iF = rng.integers(0, REPS, (4000, REPS)); iR = rng.integers(0, REPS, (4000, REPS))
        rb = np.array([np.var(vf[a], ddof=1) / np.var(vr[b], ddof=1) for a, b in zip(iF, iR)])
        lo, hi = np.percentile(rb, [2.5, 97.5])
        return np.var(vf, ddof=1) / np.var(vr, ddof=1), lo, hi

    PM = {a: 0.5 * (A[a][0] + A[a][1]) for a in A}
    st = lambda M, xs: M[:, np.argmin(np.abs(XG - xs))]
    pg = lambda M: np.max(np.abs(np.gradient(M, dx, axis=1)), axis=1)
    def lev(M, L=0.25):
        o = []
        for a in M:
            i = np.argmax(a); j = i + (np.argmax(a[i:] < L) if np.any(a[i:] < L) else 0)
            o.append(np.interp(-L, -a[j - 1:j + 1], XG[j - 1:j + 1]) if j > 0 else XG[j])
        return np.array(o)

    rows = []
    print(f"   {'observable':<44} {'ratio':>7} {'95% CI':>16}  verdict")
    print(f"   {'-'*44} {'-'*7} {'-'*16}  -------")
    mF = float(np.mean(dx * np.sum((PM['FULL'] - ref) ** 2, axis=1)))
    mR = float(np.mean(dx * np.sum((PM['RAW'] - ref) ** 2, axis=1)))
    print(f"   {'integrated squared field error (MEAN)':<44} {mF/mR:7.4f} "
          f"{'--':>16}  target metric")
    rows.append(dict(observable='integrated squared field error (mean)', ratio=mF / mR))
    print("   [linear in the field: Var ratio of the pair-mean estimator]")
    for xs, tag in ((-2.0, 'flat left flank'), (-0.7, 'left pulse'),
                    (0.1, 'steepest core'), (1.5, 'right flank')):
        r, lo, hi = rci(st(PM['FULL'], xs), st(PM['RAW'], xs))
        v = 'help' if hi < 1 else ('HARM' if lo > 1 else 'unresolved')
        print(f"   {f'u at x={xs:+.1f}  ({tag})':<44} {r:7.4f} [{lo:.3f},{hi:.3f}]  {v}")
        rows.append(dict(observable=f'u at x={xs}', tag=tag, ratio=r, ci=[lo, hi], verdict=v))
    print("   [nonlinear: Q applied to the pair-mean field]")
    for fn, tag in ((pg, 'max|grad u| (peak gradient)'), (lev, 'level crossing u=0.25')):
        r, lo, hi = rci(fn(PM['FULL']), fn(PM['RAW']))
        v = 'help' if hi < 1 else ('HARM' if lo > 1 else 'unresolved')
        print(f"   {'Q_of_mean: ' + tag:<44} {r:7.4f} [{lo:.3f},{hi:.3f}]  {v}")
        rows.append(dict(observable='Q_of_mean ' + tag, ratio=r, ci=[lo, hi], verdict=v))
    print("   [nonlinear: Q averaged over the two replicas instead]")
    for fn, tag in ((pg, 'max|grad u| (peak gradient)'), (lev, 'level crossing u=0.25')):
        qf = 0.5 * (fn(A['FULL'][0]) + fn(A['FULL'][1]))
        qr = 0.5 * (fn(A['RAW'][0]) + fn(A['RAW'][1]))
        r, lo, hi = rci(qf, qr)
        v = 'help' if hi < 1 else ('HARM' if lo > 1 else 'unresolved')
        print(f"   {'mean_of_Q: ' + tag:<44} {r:7.4f} [{lo:.3f},{hi:.3f}]  {v}")
        rows.append(dict(observable='mean_of_Q ' + tag, ratio=r, ci=[lo, hi], verdict=v))

    VF = PM['FULL'].var(0, ddof=1); VR = PM['RAW'].var(0, ddof=1)
    m = VR > 1e-3 * VR.max(); rp = (VF / np.maximum(VR, 1e-300))[m]
    g = np.abs(np.gradient(ref, dx))[m]; w = VR[m] / VR[m].sum()
    from scipy import stats as sst
    print(f"\n   SPATIAL STRUCTURE (confirmation on the fresh key):")
    print(f"     integrated (variance-weighted) pointwise ratio = {np.sum(w*rp):.4f}")
    print(f"     median pointwise ratio                        = {np.median(rp):.4f}")
    print(f"     Spearman(log V_RAW(x), log ratio(x))          = "
          f"{sst.spearmanr(np.log(VR[m]),np.log(rp)).statistic:+.3f}")
    print(f"     Spearman(log|u_x(x)|, log ratio(x))           = "
          f"{sst.spearmanr(np.log(g),np.log(rp)).statistic:+.3f}")
    qs = np.percentile(g, [0, 20, 40, 60, 80, 100])
    print(f"     {'|u_x| quintile':>16} {'median ratio':>13} {'share of int. var':>18}")
    quint = []
    for i in range(5):
        b = (g >= qs[i]) & (g <= qs[i + 1])
        print(f"     {'Q'+str(i+1):>16} {np.median(rp[b]):13.4f} "
              f"{100*VR[m][b].sum()/VR[m].sum():17.1f}%")
        quint.append(dict(q=i + 1, median_ratio=float(np.median(rp[b])),
                          var_share=float(VR[m][b].sum() / VR[m].sum())))
    np.savez_compressed(OUT / 'fields.npz', x=XG, reference=ref,
                        **{f'{a}_{p}': A[a][i].astype(np.float32)
                           for a in A for i, p in ((0, 'A'), (1, 'B'))})
    json.dump(dict(key=KEY, reps=REPS, N=N_PART, h=H, rows=rows, quintiles=quint,
                   integrated_pointwise=float(np.sum(w * rp)),
                   median_pointwise=float(np.median(rp)),
                   note='confirmation on a fresh unused key of an effect first seen '
                        'in the round-17 pilot fields; exploratory and confirmatory '
                        'evidence kept separate'),
              open(OUT / 'observable.json', 'w'), indent=1)
    print(f"\n   saved -> {OUT}")


if __name__ == '__main__':
    main()
