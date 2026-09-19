"""Paired bootstrap intervals for the flagged screen cells. Provenance only."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradient_particles import reconstruct_cumulative_field
from studies.conjecture_screen import config, variance, NU, T, N, SEEDS

OUT = ROOT / 'output/screen_intervals'
FLAGGED = [('separated', 'heat', 0.0025),
           ('separated', 'burgers', 0.005),
           ('separated', 'burgers', 0.0025)]
BOOT = 4000
N_SCREEN_CELLS = 12


def gram(Xs, Ms, u_left):
    """q_i = int U_i^2 dx and G_ik = int U_i U_k dx, exactly, on the union grid."""
    R = len(Xs)
    bp = np.unique(np.concatenate(Xs))
    mid = np.concatenate([[bp[0] - 1.0], 0.5 * (bp[1:] + bp[:-1]), [bp[-1] + 1.0]])
    lens = np.zeros(len(mid))
    lens[1:-1] = np.diff(bp)                      # end cells carry zero variance
    U = np.empty((R, len(mid)))
    for r in range(R):
        U[r] = reconstruct_cumulative_field(Xs[r], Ms[r], u_left, mid)
    W = U * lens
    G = U @ W.T
    return np.diag(G).copy(), G


def sub_var(q, G, idx):
    """Exact int Var dx over the resampled index multiset `idx`."""
    R = len(idx)
    c = np.bincount(idx, minlength=len(q)).astype(float)
    return (q[idx].sum() - (c @ G @ c) / R) / (R - 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(6061)
    rows = []
    print(f"paired exact bootstrap, {BOOT} replicates, {SEEDS} seeds, "
          f"screen key 1 (the flagged run)\n")
    print(f"{'cell':>28} {'ratio':>9} {'95% CI':>24} {'99.6% CI (Bonf)':>26} "
          f"{'verdict':>22}")
    for cfg, drift, h in FLAGGED:
        _, XR, MR = variance(cfg, h, drift, 'raw_rank', SEEDS, 1)
        _, XS, MS = variance(cfg, h, drift, 'sign_adjusted', SEEDS, 1)
        ul = 0.0
        qR, GR = gram(XR, MR, ul)
        qS, GS = gram(XS, MS, ul)
        vR = sub_var(qR, GR, np.arange(SEEDS))
        vS = sub_var(qS, GS, np.arange(SEEDS))
        idx = rng.integers(0, SEEDS, (BOOT, SEEDS))
        rr = np.array([sub_var(qS, GS, i) / sub_var(qR, GR, i) for i in idx])
        lo, hi = np.percentile(rr, [2.5, 97.5])
        blo, bhi = np.percentile(rr, [100 * 0.025 / N_SCREEN_CELLS,
                                      100 * (1 - 0.025 / N_SCREEN_CELLS)])
        excl = lo > 1.0
        bexcl = blo > 1.0
        verdict = 'CONFIRMED REVERSAL' if excl else 'not resolved'
        rows.append(dict(config=cfg, drift=drift, h=h, seeds=SEEDS,
                         var_raw=float(vR), var_sign=float(vS),
                         ratio=float(vS / vR), ci95=[float(lo), float(hi)],
                         ci_bonferroni=[float(blo), float(bhi)],
                         excludes_one_95=bool(excl),
                         excludes_one_bonferroni=bool(bexcl),
                         verdict=verdict, bootstrap=BOOT))
        print(f"{cfg + '/' + drift + '/h=' + str(h):>28} {vS / vR:9.5f} "
              f"[{lo:.5f},{hi:.5f}] [{blo:.5f},{bhi:.5f}] {verdict:>22}")

    conf = [r for r in rows if r['excludes_one_95']]
    print(f"\nconfirmed reversals among the flagged cells at 95%: {len(conf)}")
    print("the other nine screen cells retain DESCRIPTIVE point ratios; no "
          "interval was computed for them and none is claimed.")
    (OUT / 'screen_ci.json').write_text(json.dumps(dict(
        purpose='reconstruct the documented interval rule for the flagged cells only',
        rule='ratio > 1 AND paired bootstrap interval excluding 1',
        flagged=[list(f) for f in FLAGGED], rows=rows, bootstrap=BOOT,
        seeds=SEEDS, N=N, T=T, nu=NU, screen_cells=N_SCREEN_CELLS,
        exactness='bootstrap replicates use the exact whole-line integral via '
                  'the (q, G) identity, not a grid approximation',
        unflagged_cells='descriptive point ratios only'), indent=1))
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
