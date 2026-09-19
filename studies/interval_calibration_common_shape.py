"""Earlier common-shape calibration of the interval procedures (Appendix E, Table 9)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'output/principal_burgers_runs'
M, N_REP = 1500, 48


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / 'output/pointwise_observables/fields.npz')
    x = z['x']; dx = float(x[1] - x[0])
    PM = 0.5 * (z['RAW_A'].astype(np.float64) + z['RAW_B'].astype(np.float64))
    obs = {'station x=+0.1': PM[:, np.argmin(np.abs(x - 0.1))],
           'station x=-2.0': PM[:, np.argmin(np.abs(x + 2.0))],
           'max|grad u|': np.max(np.abs(np.gradient(PM, dx, axis=1)), axis=1)}
    lv = []
    for a in PM:
        i = np.argmax(a)
        j = i + (np.argmax(a[i:] < 0.25) if np.any(a[i:] < 0.25) else 0)
        lv.append(np.interp(-0.25, -a[j - 1:j + 1], x[j - 1:j + 1]) if j > 0 else x[j])
    obs['level x(u=0.25)'] = np.array(lv)

    rng = np.random.default_rng(2240)
    print("ARCHIVED CALIBRATION OF VARIANCE-RATIO INTERVALS\n")
    print(f"   resampling the empirical per-replicate distribution, n={N_REP} per arm,")
    print(f"   {M} synthetic datasets per cell; nominal coverage 0.95\n")
    print(f"   {'observable':>18} {'skew':>7} {'kurt':>7} {'Shapiro p':>10} "
          f"{'true r':>7} {'bootstrap':>16} {'F(n-1,n-1)':>16}")
    rows = []
    for name, v in obs.items():
        sk = float(stats.skew(v)); ku = float(stats.kurtosis(v))
        sh = float(stats.shapiro(v).pvalue)
        s = (v - v.mean()) / v.std(ddof=1)
        for true_r in (0.15, 0.75):
            cb = cf = 0
            for _ in range(M):
                a = rng.choice(s, N_REP, replace=True) * np.sqrt(true_r)
                b = rng.choice(s, N_REP, replace=True)
                ib = rng.integers(0, N_REP, (400, N_REP))
                jb = rng.integers(0, N_REP, (400, N_REP))
                rb = np.array([np.var(a[i], ddof=1) / np.var(b[j], ddof=1)
                               for i, j in zip(ib, jb)])
                lo, hi = np.percentile(rb, [2.5, 97.5]); cb += (lo <= true_r <= hi)
                r = np.var(a, ddof=1) / np.var(b, ddof=1)
                fq = stats.f.ppf(0.975, N_REP - 1, N_REP - 1)
                cf += (r / fq <= true_r <= r * fq)
            pb, pf = cb / M, cf / M
            se = lambda p: np.sqrt(p * (1 - p) / M)
            print(f"   {name:>18} {sk:7.3f} {ku:7.3f} {sh:10.4f} {true_r:7.2f} "
                  f"{pb:.3f}+-{se(pb):.3f}   {pf:.3f}+-{se(pf):.3f}")
            rows.append(dict(observable=name, skew=sk, kurtosis=ku, shapiro_p=sh,
                             true_ratio=true_r, boot_coverage=pb, boot_se=float(se(pb)),
                             F_coverage=pf, F_se=float(se(pf))))
    print("\n   LIMITATIONS, stated: coverage is assessed under the EMPIRICAL")
    print("   distribution of these observables at this configuration only. A")
    print("   Shapiro p-value records failure to reject, not normality. F assumes")
    print("   independent normal replicates; where its coverage is poor for an")
    print("   observable, the bootstrap interval is the one reported for it.")
    json.dump(dict(M=M, n=N_REP, rows=rows,
                   construction='centre/scale the empirical per-replicate values, '
                                'resample to impose a known ratio, count coverage',
                   limitation='performance under this empirical distribution and '
                              'configuration; not a guarantee for other problems'),
              open(OUT / 'calibration.json', 'w'), indent=1)
    print(f"\n   saved -> {OUT}/calibration.json")


if __name__ == '__main__':
    main()
