"""Round-12: work required to reach a common total-error target, when each
method may choose particle count, time step and replication count.

QUESTION (Codex's, and the right one): does the coupling reduce the WORK needed
for a given total reconstruction error, rather than the variance at fixed
settings?

MODEL. Every arm preserves each replica's discrete law, so the squared bias
b^2(N,h) is COMMON to all arms at a cell. For B independent replicates of an
estimator with per-replicate variance V,

    total error  E = b^2(N,h) + V(N,h,arm)/B,      work = B * C(N,h,arm).

For a target E <= tol the cell is feasible iff b^2 < tol, and then
B >= V/(tol - b^2), so   work(tol) = C * V / (tol - b^2),  minimised over cells.
Fractional B is model interpolation; the integer-ceiling version is also given.

ARMS.
  SINGLE  one production run per replicate (conditional-mean transport), the
          honest baseline: a pair estimator is not free.
  RAW     two replicas, rank-matched, mirror reflection everywhere.
  FULL    two replicas, rank-matched, mirror/synchronous by matched mass sign.

CELLS. two-pulse Burgers, nu=0.1, T=1, spectral reference, seed key 1 (the key
used by the round-6/7 archives), N in {1600, 6400} x h in {0.005, 0.0025}.
64 paired seeds per cell. This is a 2x2 grid, so the minimisation is over a
coarse set of choices, not an optimisation over a continuum.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import advance_rbgbmc_particles, reconstruct_cumulative_field
from studies.study_round06_cubic import advance_pair_flux, XG
from studies.round06_spectral_reference import solve, evaluate
from studies import twopulse_reference as TP

OUT = ROOT / 'output/round12_work_2026_09_10'
NU, T, A_REL, SEEDS, BOOT = 0.1, 1.0, 2.0, 64, 4000
CELLS = [(1600, 0.005), (1600, 0.0025), (6400, 0.005), (6400, 0.0025)]
TOLS = (3e-5, 1e-5, 5e-6, 3e-6, 2e-6)


def single_run(x0, m0, ul, h, K, rng):
    o = advance_rbgbmc_particles(x0, m0, ul, NU, A_REL, h, K, np.random.default_rng(0),
                                 rng_brownian=rng, conditional_mean_transport=True)
    return reconstruct_cumulative_field(o['x'], o['m'], ul, XG)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, uh, kk = solve('burgers', T, NU, M=2048, dt=1e-4)
    ref = evaluate(uh, kk, XG, 2048)
    dx = float(XG[1] - XG[0])
    rng = np.random.default_rng(1212)
    cells = {}
    print(f"two-pulse Burgers, nu={NU}, T={T}, spectral reference, {SEEDS} seeds\n")
    for N, h in CELLS:
        x0, m0, ul, _ = TP.initialize(N)
        K = round(T / h)
        F = {}
        for pol in ('RAW', 'FULL'):
            PM, SA, secs = [], [], []
            for s in range(SEEDS):
                r = np.random.default_rng(np.random.SeedSequence([6064, 1, s]))
                t0 = time.perf_counter()
                A, B, u_l = advance_pair_flux((x0, m0, ul), NU, h, K, r, pol, 'burgers')
                secs.append(time.perf_counter() - t0)
                fa = reconstruct_cumulative_field(A[0], A[1], u_l, XG)
                fb = reconstruct_cumulative_field(B[0], B[1], u_l, XG)
                PM.append((fa + fb) / 2); SA.append(fa)
            F[pol] = dict(PM=np.array(PM), SA=np.array(SA),
                          cost=float(np.median(secs)))
        # genuine single-run cost, production stepper, interleaved reps
        ts = []
        for r in range(12):
            t0 = time.perf_counter()
            single_run(x0, m0, ul, h, K, np.random.default_rng(700 + r))
            ts.append(time.perf_counter() - t0)
        c_single = float(np.median(ts))
        V = {'SINGLE': float(dx * np.sum(F['RAW']['SA'].var(axis=0, ddof=1))),
             'RAW': float(dx * np.sum(F['RAW']['PM'].var(axis=0, ddof=1))),
             'FULL': float(dx * np.sum(F['FULL']['PM'].var(axis=0, ddof=1)))}
        C = {'SINGLE': c_single, 'RAW': F['RAW']['cost'], 'FULL': F['FULL']['cost']}
        P = F['FULL']['PM']
        th = float(dx * np.sum((P.mean(0) - ref) ** 2))
        b2 = th - V['FULL'] / SEEDS
        idx = rng.integers(0, SEEDS, (BOOT, SEEDS))
        Ts = np.array([dx * np.sum((P[i].mean(0) - ref) ** 2)
                       - dx * np.sum(P[i].var(axis=0, ddof=1)) / SEEDS for i in idx])
        lo = b2 + th - np.percentile(Ts, 97.5)
        hi = b2 + th - np.percentile(Ts, 2.5)
        cells[(N, h)] = dict(b2=b2, b2_ci=[float(lo), float(hi)], V=V, C=C)
        print(f"  N={N} h={h:g}: b2={b2:.3e} [{lo:.2e},{hi:.2e}]  "
              f"V single/raw/full = {V['SINGLE']:.3e}/{V['RAW']:.3e}/{V['FULL']:.3e}")
        print(f"      cost per replicate (s) single/raw/full = "
              f"{C['SINGLE']:.4f}/{C['RAW']:.4f}/{C['FULL']:.4f}   "
              f"(pair/single = {C['RAW']/C['SINGLE']:.2f}x, "
              f"FULL/RAW = {C['FULL']/C['RAW']:.3f}x)")

    print(f"\nWORK TO REACH A COMMON TOTAL-ERROR TARGET (each arm free to pick N, h, B)")
    print(f"  {'tol':>9} " + " ".join(f"{a:>26}" for a in ('SINGLE', 'RAW', 'FULL'))
          + f" {'FULL/RAW':>9} {'FULL/SINGLE':>12}")
    rows = []
    for tol in TOLS:
        best = {}
        for arm in ('SINGLE', 'RAW', 'FULL'):
            opts = []
            for (N, h), c in cells.items():
                if c['b2'] < tol:
                    B = c['V'][arm] / (tol - c['b2'])
                    opts.append((B * c['C'][arm], N, h, B))
            best[arm] = min(opts) if opts else None
        line = f"  {tol:9.1e} "
        for arm in ('SINGLE', 'RAW', 'FULL'):
            b = best[arm]
            line += (f"{b[0]:8.2f}s N={b[1]} h={b[2]:g} B={b[3]:4.1f}".rjust(27)
                     if b else "infeasible".rjust(27))
        if best['RAW'] and best['FULL']:
            line += f" {best['FULL'][0]/best['RAW'][0]:9.3f}"
            line += f" {best['FULL'][0]/best['SINGLE'][0]:12.3f}"
        print(line)
        rows.append(dict(tol=tol, **{a: (dict(work=best[a][0], N=best[a][1],
                                              h=best[a][2], B=best[a][3])
                                         if best[a] else None)
                                     for a in ('SINGLE', 'RAW', 'FULL')}))
    print(f"\n  integer-B versions (ceil), FULL vs RAW vs SINGLE:")
    for tol in TOLS:
        out = {}
        for arm in ('SINGLE', 'RAW', 'FULL'):
            opts = [(int(np.ceil(c['V'][arm] / (tol - c['b2']))) * c['C'][arm], N, h)
                    for (N, h), c in cells.items() if c['b2'] < tol]
            out[arm] = min(opts) if opts else None
        if all(out.values()):
            print(f"   tol={tol:.1e}: single {out['SINGLE'][0]:7.2f}s  "
                  f"raw {out['RAW'][0]:7.2f}s  full {out['FULL'][0]:7.2f}s   "
                  f"FULL/RAW {out['FULL'][0]/out['RAW'][0]:.3f}  "
                  f"FULL/SINGLE {out['FULL'][0]/out['SINGLE'][0]:.3f}")
    json.dump(dict(cells={f'{k[0]}_{k[1]}': v for k, v in cells.items()},
                   work=rows, tols=list(TOLS), seeds=SEEDS,
                   note='b^2 common by marginal-law preservation; estimated from '
                        'the FULL arm (smallest V) with a pivotal bootstrap CI. '
                        'Fractional B is model interpolation. Four (N,h) cells '
                        'only, so this is a coarse choice set, not an optimum.'),
              open(OUT / 'work.json', 'w'), indent=1)
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
