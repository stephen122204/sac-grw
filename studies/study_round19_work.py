"""Round-19 step 3: (a) do the frozen decisions survive the arbitrary probe cut,
and (b) does a predicted variance ratio become a WORK-TO-ACCURACY advantage?

(b) is the point Codex raised: a variance ratio is not automatically a saving.
For the station observable u(x*,T) the total error of a B-replicate estimator is
    E[(uhat - u_ref)^2](x*) = b(x*)^2 + V(x*)/B,
and both arms share b(x*) because every replica keeps its exact marginal law.
So work to a target eps^2 at that station is
    W = C_arm * V_arm(x*) / (eps^2 - b(x*)^2),  feasible iff b^2 < eps^2.
We measure b(x*) against an independently computed spectral reference and report
attained error and measured work, not a model-only saving.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import reconstruct_cumulative_field
from studies.study_round06_cubic import advance_pair_flux, XG
from studies.round06_spectral_reference import solve, evaluate
from studies.study_round19_transfer import (u0_three, u0_widenarrow, init_from,
                                            batch, N_PART, REPS, T, H, CUT, THRESH)
from studies import twopulse_reference as TP

OUT = ROOT / 'output/round19_transfer_2026_09_10'


def main():
    K = round(T / H); t0 = time.perf_counter()
    tr = TP.initialize(N_PART)[:3]
    Rx = batch('RAW', 1901, REPS, tr, K, 0.1); Ry = batch('RAW', 1902, REPS, tr, K, 0.1)
    Ft = batch('FULL', 1903, REPS, tr, K, 0.1)
    Vx, Vy, Vf = (A.var(0, ddof=1) for A in (Rx, Ry, Ft))

    print("(a) DO THE FROZEN DECISIONS SURVIVE THE ARBITRARY PROBE CUT?\n")
    init1 = init_from(u0_three, N_PART); init2 = init_from(u0_widenarrow, N_PART)
    V = {}
    for tag, init, nu, key in (('P1', init1, 0.10, 1910), ('P2', init2, 0.05, 1920)):
        V[tag] = tuple(A.var(0, ddof=1) for A in
                       (batch('RAW', key, REPS, init, K, nu),
                        batch('RAW', key + 1, REPS, init, K, nu),
                        batch('FULL', key + 2, REPS, init, K, nu)))
    print(f"   {'cut':>8} {'q fitted':>9} {'accuracy':>9} {'median |log(rhat/r)|':>21}")
    cutrows = []
    for cut in (1e-2, 1e-3, 1e-4):
        m = Vx > cut * Vx.max()
        q, lc = np.polyfit(np.log(Vy[m]), np.log(Vf[m]), 1); C = np.exp(lc)
        ok, cal = [], []
        for tag in ('P1', 'P2'):
            Vp, Ve, Vfe = V[tag]
            mm = Vp > cut * Vp.max(); idx = np.flatnonzero(mm)
            st = [idx[np.argmin(np.abs(Vp[idx] - np.percentile(Vp[idx], p)))]
                  for p in (20, 40, 60, 80, 95)]
            for j in st:
                rh = C * Vp[j] ** (q - 1.0); rm = Vfe[j] / Ve[j]
                ok.append((rh < THRESH) == (rm < THRESH))
                cal.append(abs(np.log(rh / rm)))
        cutrows.append(dict(cut=cut, q=float(q), acc=float(np.mean(ok)),
                            cal=float(np.median(cal))))
        print(f"   {cut:>8.0e} {q:9.4f} {np.mean(ok):9.2f} {np.median(cal):21.3f}")
    print("   -> the fitted exponent moves with the cut; the DECISIONS are what "
          "must be\n      stable, and that is what this table reports.\n")

    print("(b) DOES A PREDICTED RATIO BECOME A WORK ADVANTAGE AT THAT STATION?\n")
    # measured per-replicate cost of each arm at this cell
    cost = {}
    for arm in ('RAW', 'FULL'):
        ts = []
        for r in range(12):
            s = np.random.default_rng(np.random.SeedSequence([1999, r]))
            t1 = time.perf_counter()
            A, B, u = advance_pair_flux(init1, 0.1, H, K, s, arm, 'burgers')
            reconstruct_cumulative_field(A[0], A[1], u, XG)
            reconstruct_cumulative_field(B[0], B[1], u, XG)
            ts.append(time.perf_counter() - t1)
        cost[arm] = float(np.median(ts))
    print(f"   measured cost per replicate: RAW {cost['RAW']:.4f}s  "
          f"FULL {cost['FULL']:.4f}s  (ratio {cost['FULL']/cost['RAW']:.3f})\n")

    rows = []
    for tag, nm, init, u0fn, nu, key in (('P1', 'three-pulse', init1, u0_three, 0.10, 1910),
                                         ('P2', 'wide-narrow', init2, u0_widenarrow, 0.05, 1920)):
        _, _, uh, kk = solve('burgers', T, nu, M=2048, dt=1e-4, u0=u0fn)
        ref = evaluate(uh, kk, XG, 2048)
        Vp, Ve, Vfe = V[tag]
        ER = batch('RAW', key + 1, REPS, init, K, nu)
        EF = batch('FULL', key + 2, REPS, init, K, nu)
        mm = Vp > CUT * Vp.max(); idx = np.flatnonzero(mm)
        # one station where the frozen rule predicted HELP, one where it predicted NOT
        stH = idx[np.argmin(np.abs(Vp[idx] - np.percentile(Vp[idx], 95)))]
        stN = idx[np.argmin(np.abs(Vp[idx] - np.percentile(Vp[idx], 20)))]
        print(f"   {tag} {nm}")
        for lbl, j in (('predicted HELP  (95th pct)', stH), ('predicted NONE  (20th pct)', stN)):
            bR = float(ER[:, j].mean() - ref[j]); bF = float(EF[:, j].mean() - ref[j])
            vR = float(Ve[j]); vF = float(Vfe[j]); b2 = 0.5 * (bR ** 2 + bF ** 2)
            eps2 = max(4.0 * b2, 1e-12)          # frozen: target = 4x the shared bias^2
            wR = cost['RAW'] * vR / (eps2 - b2); wF = cost['FULL'] * vF / (eps2 - b2)
            print(f"     {lbl}  x*={XG[j]:+.3f}")
            print(f"       bias RAW {bR:+.3e}  bias FULL {bF:+.3e}  (shared by marginal law)")
            print(f"       V_RAW {vR:.3e}  V_FULL {vF:.3e}  variance ratio {vF/vR:.4f}")
            print(f"       work to eps^2={eps2:.2e}: RAW {wR:.3f}s  FULL {wF:.3f}s  "
                  f"WORK RATIO {wF/wR:.4f}")
            rows.append(dict(profile=tag+' '+nm, label=lbl, x=float(XG[j]), bias_raw=bR,
                             bias_full=bF, V_raw=vR, V_full=vF, var_ratio=vF / vR,
                             work_ratio=float(wF / wR)))
        print()
    json.dump(dict(cut_robustness=cutrows, cost=cost, work=rows,
                   note='station work uses the shared-bias model with measured '
                        'b, V and cost; target frozen at 4x the shared bias^2'),
              open(OUT / 'work.json', 'w'), indent=1)
    print(f"   {time.perf_counter()-t0:.0f}s   saved -> {OUT}/work.json")


if __name__ == '__main__':
    main()
