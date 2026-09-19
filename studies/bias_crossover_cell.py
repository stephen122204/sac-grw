"""Joint refinement cell for the bias-dominated regime (Section 4.6, Figure 5)."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradient_particles import reconstruct_cumulative_field
from studies.cubic_flux_transfer import advance_pair_flux, XG
from studies.spectral_reference import solve, evaluate
from studies import twopulse_reference as TP

OUT = ROOT / 'output/bias_crossover_cell'
NU, T, SEEDS, BOOT = 0.1, 1.0, 64, 4000


def run(N, h, ref, key=1):
    x0, m0, ul, _ = TP.initialize(N)
    K = round(T / h)
    dx = float(XG[1] - XG[0])
    res = {}
    for p in ('RAW', 'FULL'):
        PM, secs = [], []
        for s in range(SEEDS):
            r = np.random.default_rng(np.random.SeedSequence([6064, key, s]))
            t0 = time.perf_counter()
            A, B, u_l = advance_pair_flux((x0, m0, ul), NU, h, K, r, p, 'burgers')
            secs.append(time.perf_counter() - t0)
            fa = reconstruct_cumulative_field(A[0], A[1], u_l, XG)
            fb = reconstruct_cumulative_field(B[0], B[1], u_l, XG)
            PM.append((fa + fb) / 2)
        PM = np.asarray(PM)
        V = float(dx * np.sum(PM.var(axis=0, ddof=1)))
        th = float(dx * np.sum((PM.mean(0) - ref) ** 2))
        res[p] = dict(PM=PM, V=V, theta=th, T=th - V / SEEDS,
                      cost=float(np.mean(secs)))
    return res, dx


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, uh, k = solve('burgers', T, NU, M=2048, dt=1e-4)
    ref = evaluate(uh, k, XG, 2048)
    rng = np.random.default_rng(7073)
    print(f"two-pulse, nu={NU}, T={T}, spectral reference, {SEEDS} paired seeds\n")
    print(f"{'cell':>22} {'b2 (FULL) [pivotal CI]':>36} {'V_FULL':>11} "
          f"{'V_RAW':>11} {'B*_FULL':>9}")
    rows = []
    for N, h in ((6400, 0.0025), (6400, 0.005), (1600, 0.0025)):
        t0 = time.perf_counter()
        res, dx = run(N, h, ref)
        idx = rng.integers(0, SEEDS, (BOOT, SEEDS))
        P = res['FULL']['PM']
        Ts = np.array([dx * np.sum((P[i].mean(0) - ref) ** 2)
                       - dx * np.sum(P[i].var(axis=0, ddof=1)) / SEEDS for i in idx])
        Tf, thf = res['FULL']['T'], res['FULL']['theta']
        lo = Tf + thf - np.percentile(Ts, 97.5)
        hi = Tf + thf - np.percentile(Ts, 2.5)
        bstar = res['FULL']['V'] / Tf
        print(f"{'N=' + str(N) + ' h=' + format(h, 'g'):>22} "
              f"{Tf:.3e} [{lo:.3e},{hi:.3e}] {res['FULL']['V']:11.4e} "
              f"{res['RAW']['V']:11.4e} {bstar:9.1f}   ({time.perf_counter()-t0:.0f}s)")
        rows.append(dict(N=N, h=h, b2_full=Tf, b2_ci=[float(lo), float(hi)],
                         V_full=res['FULL']['V'], V_raw=res['RAW']['V'],
                         Bstar_full=float(bstar),
                         var_ratio=res['FULL']['V'] / res['RAW']['V'],
                         cost_full=res['FULL']['cost'], cost_raw=res['RAW']['cost']))
    print("\nvariance ratio FULL/RAW at each cell:")
    for r in rows:
        print(f"  N={r['N']} h={r['h']:g}: {r['var_ratio']:.4f}")
    (OUT / 'joint_cell.json').write_text(json.dumps(dict(
        purpose='decide whether the usable ensemble window survives joint N,h refinement',
        baseline='two-pulse, nu=0.1, T=1, spectral reference, 64 paired seeds',
        rows=rows,
        caveat='three cells indicate a direction; they do not establish an '
               'asymptotic rate or separate all N/h interactions'), indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
