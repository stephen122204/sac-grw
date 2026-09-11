"""Round-6 part 5: preliminary accuracy versus work, from archived results only.

For B INDEPENDENT pair estimates averaged together,

    E[error] = b(N,h)^2 + V_pair(N,h) / B,      work = B * C_pair(N,h),

with b the common discretisation bias (identical across couplings: every
coupling preserves each arm's marginal law) and V_pair the terminal pair-mean
variance the coupling actually changes.

Two things are reported separately and never merged:

  MODEL     b2_hat + V_hat/B, a curve extrapolated from the S-seed moments;
  MEASURED  the mean squared error of B-averages formed from DISJOINT blocks of
            the archived seeds, so no seed enters two blocks. Overlapping
            regroupings would reuse the same realisations and understate the
            spread, so they are not used; S=64 supports B in {1,2,4,8} with
            64, 32, 16 and 8 independent blocks respectively.

The bias estimator

    b2_hat = int (mean_S PM - ref)^2 dx  -  V_hat / S

is unbiased for int b^2 and CAN COME OUT NEGATIVE when the true bias is small
relative to Monte Carlo scatter. It is reported signed, with a bootstrap
interval. A negative value is evidence that the bias is not resolved at this S;
it is not a proof that the bias is zero, and it is never clamped.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARCH = ROOT / 'output/round06_replay_2026_09_09/archive_v2.npz'
OLD = ROOT / 'output/round05_validation_2026_09_09/archive.npz'
OUT = ROOT / 'output/round06_accuracy_work_2026_09_09'
COUPLINGS = ('raw_rank', 'sign_adjusted', 'within_sign', 'independent')
BS = (1, 2, 4, 8)
BOOT = 4000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ARCH)
    old = np.load(OLD)
    rng = np.random.default_rng(6066)
    rows = []
    for problem, N in (('gaussian', 1600), ('twopulse', 1600)):
        x = z[f'{problem}_{N}_x']
        ref = z[f'{problem}_{N}_ref']
        dx = float(x[1] - x[0])
        print(f"\n{problem} N={N}   (S=64 archived seeds; work uses the round-5 "
              f"per-seed clock, the only timing measured under one protocol)")
        print("-" * 104)
        print(f"  {'coupling':>14} {'b2_hat':>12} {'b2 95% CI':>24} {'V_pair':>11} "
              f"{'C_pair':>9} {'B*=V/b2':>9}")
        for coup in COUPLINGS:
            PM = z[f'{problem}_{N}_{coup}_pairmean']
            S = PM.shape[0]
            C = float(np.mean(old[f'{problem}_{N}_{coup}_secs']))
            V = float(dx * np.sum(PM.var(axis=0, ddof=1)))
            b2 = float(dx * np.sum((PM.mean(0) - ref) ** 2) - V / S)
            idx = rng.integers(0, S, (BOOT, S))
            bb = np.array([
                dx * np.sum((PM[i].mean(0) - ref) ** 2)
                - dx * np.sum(PM[i].var(axis=0, ddof=1)) / S for i in idx])
            lo, hi = np.percentile(bb, [2.5, 97.5])
            bstar = (V / b2) if b2 > 0 else float('inf')
            print(f"  {coup:>14} {b2:12.4e} [{lo:11.4e},{hi:11.4e}] {V:11.4e} "
                  f"{C:9.4f} {bstar:9.1f}")
            for B in BS:
                nb = S // B
                blocks = PM[:nb * B].reshape(nb, B, -1).mean(axis=1)
                meas = float(np.mean(dx * np.sum((blocks - ref) ** 2, axis=1)))
                se = float(np.std(dx * np.sum((blocks - ref) ** 2, axis=1),
                                  ddof=1) / np.sqrt(nb))
                rows.append(dict(problem=problem, N=N, coupling=coup, B=B,
                                 blocks=nb, work_s=B * C,
                                 model_error=b2 + V / B,
                                 measured_error=meas, measured_se=se,
                                 b2_hat=b2, b2_ci=[float(lo), float(hi)],
                                 V_pair=V, C_pair=C))
        print(f"\n  {'coupling':>14} " + " ".join(
            f"{'B=' + str(B):>26}" for B in BS))
        print(f"  {'':>14} " + " ".join(
            f"{'work  model / measured':>26}" for _ in BS))
        for coup in COUPLINGS:
            cells = []
            for B in BS:
                r = [q for q in rows if q['problem'] == problem
                     and q['coupling'] == coup and q['B'] == B][0]
                cells.append(f"{r['work_s']:6.2f}s {r['model_error']:.2e}/"
                             f"{r['measured_error']:.2e}")
            print(f"  {coup:>14} " + " ".join(f"{c:>26}" for c in cells))

        # matched-work comparison: raw at B vs sign-adjusted at the same work
        print(f"\n  matched-work reading (equal wall clock, model curve):")
        raw = [q for q in rows if q['problem'] == problem
               and q['coupling'] == 'raw_rank']
        sgn = [q for q in rows if q['problem'] == problem
               and q['coupling'] == 'sign_adjusted']
        for r in raw:
            w = r['work_s']
            Bs_equiv = w / sgn[0]['C_pair']
            err = sgn[0]['b2_hat'] + sgn[0]['V_pair'] / Bs_equiv
            print(f"    work {w:6.2f}s: raw B={r['B']} -> {r['model_error']:.3e} ; "
                  f"sign-adjusted B={Bs_equiv:4.1f} -> {err:.3e}   "
                  f"(ratio {err / r['model_error']:.3f})")
    with (OUT / 'accuracy_work.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / 'accuracy_work.json').write_text(json.dumps(dict(
        purpose='preliminary accuracy vs work from archived results only',
        model='error = b2 + V_pair/B, work = B*C_pair',
        measured='disjoint blocks only; no seed enters two blocks',
        bias_estimator='int(mean-ref)^2 dx - V/S, unbiased, reported signed, '
                       'never clamped; a negative value means the bias is '
                       'unresolved at S=64, not that it is zero',
        timing_source='round-5 per-seed clock (includes the estimator and the '
                      'audit scan); it is the only timing measured under a '
                      'single protocol for all four couplings',
        B_values=list(BS), bootstrap=BOOT, rows=rows), indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
