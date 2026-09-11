"""Round-25: repair the interval calibration so the two synthetic arms have
DIFFERENT standardised shapes.

DEFECT FOUND IN REVIEW (round-22 independent review, item 3): the archived
calibration built BOTH synthetic arms by resampling the SAME standardised
empirical RAW distribution and rescaling one of them. That legitimately tests a
common-shape design; it says nothing about coverage when the numerator and
denominator samples have different shapes -- which is exactly the situation when
FULL is compared with RAW. A round-23 report quoted a "redone" calibration, but
no such archive exists. This driver supplies it.

CONSTRUCTION, stated:
  * per-replicate values of each observable are taken from the archived N=8192
    round-18 fields, separately for the RAW pair mean and the FULL pair mean;
  * each sample is centred and scaled by its OWN standard deviation, giving two
    empirical shapes s_FULL and s_RAW;
  * a synthetic numerator arm resamples s_FULL and is multiplied by sqrt(r) for a
    known true variance ratio r; the denominator arm resamples s_RAW;
  * nominal-95% intervals for r are formed by (a) percentile bootstrap over whole
    replicates and (b) F(n-1, n-1), and coverage is counted over M datasets.

The common-shape design of the archived round-22 calibration is rerun here as a
control so the two can be compared in one place.

LIMITATION, stated: this assesses coverage UNDER THESE EMPIRICAL DISTRIBUTIONS at
THIS configuration. It is not a guarantee for other problems. A non-significant
normality test records failure to reject, not normality.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'output/round25_calibration_2026_09_10'
M, N_REP, NBOOT = 1500, 48, 400
SEED = 2500


def observables(PM, x, dx):
    o = {'station x=+0.1': PM[:, np.argmin(np.abs(x - 0.1))],
         'station x=-2.0': PM[:, np.argmin(np.abs(x + 2.0))],
         'max|grad u|': np.max(np.abs(np.gradient(PM, dx, axis=1)), axis=1)}
    lv = []
    for a in PM:
        i = np.argmax(a)
        j = i + (np.argmax(a[i:] < 0.25) if np.any(a[i:] < 0.25) else 0)
        lv.append(np.interp(-0.25, -a[j - 1:j + 1], x[j - 1:j + 1]) if j > 0 else x[j])
    o['level x(u=0.25)'] = np.array(lv)
    return o


def coverage(sh_num, sh_den, true_r, rng):
    """Coverage of nominal-95% variance-ratio intervals under two given shapes."""
    fq = stats.f.ppf(0.975, N_REP - 1, N_REP - 1)
    cb = cf = 0
    for _ in range(M):
        a = rng.choice(sh_num, N_REP, replace=True) * np.sqrt(true_r)
        b = rng.choice(sh_den, N_REP, replace=True)
        ia = rng.integers(0, N_REP, (NBOOT, N_REP))
        ib = rng.integers(0, N_REP, (NBOOT, N_REP))
        rb = a[ia].var(axis=1, ddof=1) / b[ib].var(axis=1, ddof=1)
        lo, hi = np.percentile(rb, [2.5, 97.5])
        cb += (lo <= true_r <= hi)
        r = a.var(ddof=1) / b.var(ddof=1)
        cf += (r / fq <= true_r <= r * fq)
    return cb / M, cf / M


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / 'output/round18_observable_2026_09_10/fields.npz')
    x = z['x']; dx = float(x[1] - x[0])
    PM = {a: 0.5 * (z[f'{a}_A'].astype(np.float64) + z[f'{a}_B'].astype(np.float64))
          for a in ('RAW', 'FULL')}
    obs = {a: observables(PM[a], x, dx) for a in PM}

    rng = np.random.default_rng(SEED); t0 = time.perf_counter()
    print("INTERVAL CALIBRATION WITH TWO DIFFERENT EMPIRICAL SHAPES\n")
    print(f"   numerator shape = standardised FULL sample, denominator = standardised RAW")
    print(f"   n={N_REP} per arm, M={M} synthetic datasets per cell, nominal 0.95\n")
    hdr = (f"   {'observable':>18} {'design':>12} {'true r':>7} "
           f"{'bootstrap':>16} {'F(n-1,n-1)':>16}")
    print(hdr)
    rows, shapes = [], []
    se = lambda p: float(np.sqrt(p * (1 - p) / M))
    for name in obs['RAW']:
        vR, vF = obs['RAW'][name], obs['FULL'][name]
        sR = (vR - vR.mean()) / vR.std(ddof=1)
        sF = (vF - vF.mean()) / vF.std(ddof=1)
        shapes.append(dict(observable=name,
                           skew_raw=float(stats.skew(vR)), kurt_raw=float(stats.kurtosis(vR)),
                           shapiro_p_raw=float(stats.shapiro(vR).pvalue),
                           skew_full=float(stats.skew(vF)), kurt_full=float(stats.kurtosis(vF)),
                           shapiro_p_full=float(stats.shapiro(vF).pvalue),
                           ks_shape_p=float(stats.ks_2samp(sR, sF).pvalue)))
        for true_r in (0.15, 0.75):
            for design, (sn, sd_) in (('two-shape', (sF, sR)), ('common-shape', (sR, sR))):
                pb, pf = coverage(sn, sd_, true_r, rng)
                print(f"   {name:>18} {design:>12} {true_r:7.2f} "
                      f"{pb:.3f}+-{se(pb):.3f}   {pf:.3f}+-{se(pf):.3f}")
                rows.append(dict(observable=name, design=design, true_ratio=true_r,
                                 boot_coverage=pb, boot_se=se(pb),
                                 F_coverage=pf, F_se=se(pf)))
        print()
    print("   LIMITATION: coverage under THESE empirical distributions at THIS")
    print("   configuration only; a Shapiro p-value records failure to reject.")
    json.dump(dict(M=M, n=N_REP, n_boot=NBOOT, seed=SEED, rows=rows, shapes=shapes,
                   source='output/round18_observable_2026_09_10/fields.npz',
                   construction='standardise the RAW and FULL per-replicate samples '
                                'separately; resample the FULL shape (scaled) against '
                                'the RAW shape to impose a known variance ratio; count '
                                'coverage. The common-shape design of round 22 is rerun '
                                'as a control.',
                   limitation='performance under these empirical distributions and this '
                              'configuration; not a guarantee for other problems'),
              open(OUT / 'calibration.json', 'w'), indent=1)
    print(f"\n   {time.perf_counter()-t0:.0f}s   saved -> {OUT}/calibration.json")


if __name__ == '__main__':
    main()
