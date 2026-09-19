"""Tests whether a baseline-only pilot predicts the coupling benefit on other problems (Appendix E)."""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradient_particles import reconstruct_cumulative_field
from studies.cubic_flux_transfer import advance_pair_flux, XG
from studies.spectral_reference import solve
from studies import twopulse_reference as TP

OUT = ROOT / 'output/pilot_prediction_transfer'
T, H, N_PART, REPS = 1.0, 0.005, 2048, 64
CUT, THRESH = 1e-3, 0.5


def u0_three(x):
    return (0.7 * np.exp(-(x + 1.2) ** 2 / (2 * 0.45 ** 2))
            - 0.55 * np.exp(-(x + 0.1) ** 2 / (2 * 0.28 ** 2))
            + 0.4 * np.exp(-(x - 1.0) ** 2 / (2 * 0.35 ** 2)))


def u0_widenarrow(x):
    return (0.6 * np.exp(-(x + 0.9) ** 2 / (2 * 0.8 ** 2))
            - 0.65 * np.exp(-(x - 0.5) ** 2 / (2 * 0.18 ** 2)))


def init_from(u0fn, N, L=6.0):
    """Deterministic signed-quantile init: N/2 particles per sign, exact sum."""
    xs = np.linspace(-L, L, 200001)
    w = np.gradient(u0fn(xs), xs[1] - xs[0])
    out_x, out_m = [], []
    for sgn in (+1.0, -1.0):
        wp = np.where(np.sign(w) == sgn, np.abs(w), 0.0)
        c = np.cumsum(wp) * (xs[1] - xs[0])
        tot = c[-1]
        if tot <= 0:
            continue
        n = N // 2
        q = (np.arange(n) + 0.5) / n * tot
        out_x.append(np.interp(q, c, xs))
        out_m.append(np.full(n, sgn * tot / n))
    x0 = np.concatenate(out_x); m0 = np.concatenate(out_m)
    o = np.argsort(x0)
    return x0[o].copy(), m0[o].copy(), float(u0fn(-L))


def batch(arm, key, reps, init, K, nu):
    x0, m0, ul = init
    P = []
    for r in range(reps):
        s = np.random.default_rng(np.random.SeedSequence([key, r]))
        A, B, u = advance_pair_flux((x0, m0, ul), nu, H, K, s, arm, 'burgers')
        P.append(0.5 * (reconstruct_cumulative_field(A[0], A[1], u, XG)
                        + reconstruct_cumulative_field(B[0], B[1], u, XG)))
    return np.array(P)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    K = round(T / H); t0 = time.perf_counter()
    # ---- TRAINING on two-pulse, independent RAW batches ----
    tr = TP.initialize(N_PART)[:3]
    Rx = batch('RAW', 1901, REPS, tr, K, 0.1)
    Ry = batch('RAW', 1902, REPS, tr, K, 0.1)
    Ft = batch('FULL', 1903, REPS, tr, K, 0.1)
    Vx, Vy, Vf = (A.var(0, ddof=1) for A in (Rx, Ry, Ft))
    m = Vx > CUT * Vx.max()
    q, lc = np.polyfit(np.log(Vy[m]), np.log(Vf[m]), 1)
    C = float(np.exp(lc))
    cconst = float(np.exp(np.mean(np.log(Vf[m]) - np.log(Vy[m]))))
    print(f"TRAINING (two-pulse, {REPS} reps, independent denominator batch)")
    print(f"   FROZEN model : V_FULL = {C:.5f} * V_RAW^{q:.4f}")
    print(f"   FROZEN baseline (constant reduction): V_FULL = {cconst:.4f} * V_RAW")
    print(f"   frozen probe cut {CUT:g}, decision threshold rhat < {THRESH}\n")

    rows = []
    for name, u0fn, nu, key in (('P1 three-pulse', u0_three, 0.10, 1910),
                                ('P2 wide-narrow (nu=0.05)', u0_widenarrow, 0.05, 1920)):
        init = init_from(u0fn, N_PART)
        Rp = batch('RAW', key, REPS, init, K, nu)          # the PILOT: baseline only
        Re = batch('RAW', key + 1, REPS, init, K, nu)      # independent eval RAW
        Fe = batch('FULL', key + 2, REPS, init, K, nu)     # independent eval FULL
        Vp, Ve, Vfe = (A.var(0, ddof=1) for A in (Rp, Re, Fe))
        mm = Vp > CUT * Vp.max()
        idx = np.flatnonzero(mm)
        # frozen station rule: percentile locations of the PILOT variance
        st = [idx[np.argmin(np.abs(Vp[idx] - np.percentile(Vp[idx], p)))]
              for p in (20, 40, 60, 80, 95)]
        print(f"{name}: {int(mm.sum())} admissible probes, "
              f"integrated ratio (eval) = "
              f"{np.sum(Vfe[mm])/np.sum(Ve[mm]):.4f}")
        print(f"   {'x*':>7} {'V_RAW pilot':>12} {'r predicted':>12} {'r measured':>11} "
              f"{'decision':>10} {'truth':>7} {'ok':>4}")
        for j in st:
            rhat = C * Vp[j] ** (q - 1.0)
            rmea = Vfe[j] / Ve[j]
            dec = rhat < THRESH; tru = rmea < THRESH
            rows.append(dict(profile=name, x=float(XG[j]), Vpilot=float(Vp[j]),
                             rhat=float(rhat), rmeas=float(rmea),
                             decision=bool(dec), truth=bool(tru), ok=bool(dec == tru)))
            print(f"   {XG[j]:7.3f} {Vp[j]:12.3e} {rhat:12.4f} {rmea:11.4f} "
                  f"{str(dec):>10} {str(tru):>7} {'YES' if dec==tru else 'no':>4}")
        print()
    acc = np.mean([r['ok'] for r in rows])
    cal = np.median([abs(np.log(r['rhat'] / r['rmeas'])) for r in rows])
    # constant-reduction comparator on the same stations
    calc = np.median([abs(np.log(cconst / r['rmeas'])) for r in rows])
    accc = np.mean([(cconst < THRESH) == r['truth'] for r in rows])
    print(f"FROZEN SUCCESS CRITERIA")
    print(f"   (S1) decision accuracy   = {acc:.2f}   (need >= 0.70)  "
          f"{'PASS' if acc>=0.70 else 'FAIL'}")
    print(f"   (S2) median |log(rhat/r)| = {cal:.3f}  (need < 0.50)   "
          f"{'PASS' if cal<0.50 else 'FAIL'}")
    print(f"   overall: {'TRANSFERABLE' if (acc>=0.70 and cal<0.50) else 'NOT ESTABLISHED'}")
    print(f"\n   constant-reduction baseline on the same stations: "
          f"accuracy {accc:.2f}, median |log err| {calc:.3f}")
    json.dump(dict(frozen=dict(C=C, q=float(q), const=cconst, cut=CUT, thresh=THRESH,
                               criteria='S1 acc>=0.70 and S2 median|log|<0.50'),
                   rows=rows, accuracy=float(acc), calibration=float(cal),
                   const_accuracy=float(accc), const_calibration=float(calc),
                   verdict='TRANSFERABLE' if (acc >= 0.70 and cal < 0.50)
                           else 'NOT ESTABLISHED',
                   runtime_s=time.perf_counter() - t0),
              open(OUT / 'transfer.json', 'w'), indent=1)
    print(f"\n   {time.perf_counter()-t0:.0f}s   saved -> {OUT}")


if __name__ == '__main__':
    main()
