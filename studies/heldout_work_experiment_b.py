"""Held-out work comparison including within-sign pairing (Section 4.5, Experiment B)."""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradient_particles import reconstruct_cumulative_field
from studies.cubic_flux_transfer import advance_pair_flux, XG
from studies.spectral_reference import solve, evaluate
from studies import twopulse_reference as TP

OUT = ROOT / 'output/profile_time_flux_sweep'
NU, T, H, N = 0.1, 1.0, 0.005, 2048
ARMS = ('RAW', 'WITHIN', 'FULL'); AID = {'RAW': 2, 'FULL': 3, 'WITHIN': 4}
TOL, KAPPA, PILOT, BLOCKS, BMAX, CAP = 2.5e-5, 1.3, 32, 32, 300, 900.0


def one(arm, init, K, ss):
    A, B, u = advance_pair_flux(init, NU, H, K, np.random.default_rng(ss), arm, 'burgers')
    return 0.5 * (reconstruct_cumulative_field(A[0], A[1], u, XG)
                  + reconstruct_cumulative_field(B[0], B[1], u, XG))


def main():
    t0 = time.perf_counter()
    _, _, uh, kk = solve('burgers', T, NU, M=2048, dt=1e-4)
    ref = evaluate(uh, kk, XG, 2048); dx = float(XG[1] - XG[0]); K = round(T / H)
    init = TP.initialize(N)[:3]
    rng = np.random.default_rng(2311)

    P, C = {}, {}
    tune0 = time.perf_counter()
    for i in range(PILOT):
        for a in rng.permutation(list(ARMS)):
            s = time.perf_counter()
            f = one(a, init, K, np.random.SeedSequence([2310, AID[a], i]))
            C.setdefault(a, []).append(time.perf_counter() - s)
            P.setdefault(a, []).append(f)
    P = {a: np.array(v) for a, v in P.items()}
    tune_cost = time.perf_counter() - tune0
    idx = rng.integers(0, PILOT, (3000, PILOT))
    st = {}
    for a in ARMS:
        V = float(dx * np.sum(P[a].var(0, ddof=1)))
        Vb = np.array([dx * np.sum(P[a][i].var(0, ddof=1)) for i in idx])
        th = float(dx * np.sum((P[a].mean(0) - ref) ** 2)); b2 = th - V / PILOT
        Tb = np.array([dx * np.sum((P[a][i].mean(0) - ref) ** 2)
                       - dx * np.sum(P[a][i].var(0, ddof=1)) / PILOT for i in idx])
        st[a] = dict(V=V, V_hi=float(np.percentile(Vb, 97.5)), b2=b2,
                     b2_hi=float(b2 + th - np.percentile(Tb, 2.5)),
                     C=float(np.median(C[a])))
    b2h = st['FULL']['b2_hi']; tgt = TOL / KAPPA
    print(f"PILOT (key 2310, {PILOT} reps/arm; tuning cost {tune_cost:.1f}s, "
          f"reported separately)")
    for a in ARMS:
        print(f"   {a:>7}: V={st[a]['V']:.4e}  V_hi={st[a]['V_hi']:.4e}  "
              f"C={st[a]['C']:.4f}s")
    print(f"   shared b2 (FULL arm) = {st['FULL']['b2']:.3e} [hi {b2h:.3e}]; "
          f"design target {tgt:.3e}\n")
    plans = {}
    for a in ARMS:
        B = int(np.ceil(st[a]['V_hi'] / (tgt - b2h)))
        plans[a] = None if B > BMAX else dict(B=B, work=B * st[a]['C'])
        print(f"   {a:>7}: B={B}  projected work/block "
              f"{B*st[a]['C']:.3f}s" + ("  (over B_max)" if B > BMAX else ""))
    proj = sum(p['work'] * BLOCKS for p in plans.values() if p)
    print(f"\n   projected evaluation {proj:.0f}s; cap {CAP:.0f}s")
    if proj > CAP:
        print("   -> OVER BUDGET, not run, not trimmed."); return

    print(f"\nEVALUATION (key 2320, {BLOCKS} blocks, frozen)\n")
    res, meas = {}, {}
    for a in ARMS:
        if not plans[a]: continue
        B = plans[a]['B']; errs, secs = [], []
        for b in range(BLOCKS):
            acc = np.zeros(len(XG)); s0 = time.perf_counter()
            for r in range(B):
                acc += one(a, init, K, np.random.SeedSequence([2320, AID[a], b, r]))
            secs.append(time.perf_counter() - s0)
            errs.append(float(dx * np.sum((acc / B - ref) ** 2)))
        e = np.array(errs); res[a] = e; meas[a] = float(np.median(secs))
        m = e.mean(); se = e.std(ddof=1) / np.sqrt(BLOCKS)
        tc = stats.t.ppf(0.975, BLOCKS - 1); tb = stats.t.ppf(1 - 0.025 / 3, BLOCKS - 1)
        v = 'ATTAINED' if m + tc * se < TOL else ('MISSED' if m - tc * se > TOL else 'unresolved')
        vs = 'ATTAINED' if m + tb * se < TOL else ('MISSED' if m - tb * se > TOL else 'unresolved')
        print(f"   {a:>7} B={B:3d} ({2*B:3d} trajectories): err {m:.4e} "
              f"[{m-tc*se:.3e},{m+tc*se:.3e}] {v:>10} | simult {vs:>10} | "
              f"work {meas[a]:.3f}s")
    if 'FULL' in meas and 'WITHIN' in meas:
        print(f"\n   FULL / WITHIN measured work = {meas['FULL']/meas['WITHIN']:.4f}")
        print(f"   FULL / RAW    measured work = {meas['FULL']/meas['RAW']:.4f}")
    json.dump(dict(tol=TOL, kappa=KAPPA, N=N, pilot_key=2310, eval_key=2320,
                   blocks=BLOCKS, tuning_cost_s=tune_cost, pilot=st,
                   plans={a: plans[a] for a in ARMS},
                   realised={a: dict(mean=float(v.mean()), n=len(v),
                                     se=float(v.std(ddof=1) / np.sqrt(len(v))),
                                     per_block=v.tolist(), work_s=meas[a])
                             for a, v in res.items()},
                   note='selected-plan attainment at a common target; WITHIN '
                        'included; standard two-pulse configuration, not the '
                        'oscillatory profile favourable to FULL'),
              open(OUT / 'worktarget.json', 'w'), indent=1)
    print(f"\n   {time.perf_counter()-t0:.0f}s   saved -> {OUT}/worktarget.json")


if __name__ == '__main__':
    main()
