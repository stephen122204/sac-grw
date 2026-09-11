"""Round-16: frozen four-arm work-to-target comparison, with a budget gate.

PREDECLARED, fixed before any pilot result is inspected.
  cells        N in {2048, 8192} (powers of two, admissible base-2 net sizes),
               h = 0.005, two-pulse Burgers, nu=0.1, T=1
  arms         SINGLE, RAW, FULL, RQMC (randomized rank-based diffusion
               adaptation inspired by Lecot, on our transport solver)
  target       tol = 5e-6 EXPECTED integrated squared error on [-5,5]
  margin       plan against tol/kappa with kappa = 1.3, using the UPPER 95%
               endpoints of the pilot V and b^2:  B = ceil(V_hi/(tol/kappa - b2_hi))
  bias         SINGLE/RAW/FULL share the discrete law: b^2 taken from the FULL
               arm (smallest V, sharpest), signed and never clamped. RQMC may
               change the finite-N law, so it gets its OWN b^2.
  pilot        key 1601, 32 replicates per arm per cell (selection only)
  evaluation   key 1602, 32 disjoint blocks per arm, plans frozen
  inference    per-arm attainment is POINTWISE (95% t on block errors:
               attained if the upper end < tol, failed if the lower end > tol,
               otherwise unresolved). Any joint claim uses Bonferroni at
               alpha/4 and is labelled simultaneous.
  caps         B_max = 200 per arm; projected evaluation budget must be <= 20
               minutes or the comparison is REPORTED AS NOT RUN, not trimmed.
  stopping     one pilot, one evaluation, no topping up after seeing results.
  costs        production and tuning reported separately; no amortisation is
               used to establish usefulness.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import reconstruct_cumulative_field
from studies.round16_rqmc import RQMCDiffusion, advance_rank_diffusion
from studies.study_round06_cubic import advance_pair_flux, XG
from studies.round06_spectral_reference import solve, evaluate, FPRIME
from studies.study_round13_validation import advance_single_flux
from studies import twopulse_reference as TP

OUT = ROOT / 'output/round16_compare_2026_09_10'
NU, T, H_STEP, TOL, KAPPA = 0.1, 1.0, 0.005, 5e-6, 1.3
CELLS = [2048, 8192]
ARMS = ('SINGLE', 'RAW', 'FULL', 'RQMC')
ARM_ID = {'SINGLE': 1, 'RAW': 2, 'FULL': 3, 'RQMC': 4}
PILOT_KEY, EVAL_KEY, PILOT, BLOCKS = 1601, 1602, 32, 32
B_MAX, BUDGET_S = 200, 1200.0


def seed_of(key, arm, i, j=0):
    return np.random.SeedSequence([key, ARM_ID[arm], i, j])


def one_field(arm, N, init, K, ss):
    """One replicate of `arm`; returns the estimator field for that replicate."""
    x0, m0, ul = init
    if arm == 'SINGLE':
        x, m, u_l = advance_single_flux((x0, m0, ul), NU, H_STEP, K,
                                        np.random.default_rng(ss), 'burgers')
        return reconstruct_cumulative_field(x, m, u_l, XG)
    if arm == 'RQMC':
        q = RQMCDiffusion(N, seed=int(ss.generate_state(1)[0]))
        x, m, u_l = advance_rank_diffusion((x0, m0, ul), NU, H_STEP, K, q,
                                           FPRIME['burgers'])
        return reconstruct_cumulative_field(x, m, u_l, XG)
    A, B, u_l = advance_pair_flux((x0, m0, ul), NU, H_STEP, K,
                                  np.random.default_rng(ss), arm, 'burgers')
    fa = reconstruct_cumulative_field(A[0], A[1], u_l, XG)
    fb = reconstruct_cumulative_field(B[0], B[1], u_l, XG)
    return 0.5 * (fa + fb)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, uh, kk = solve('burgers', T, NU, M=2048, dt=1e-4)
    ref = evaluate(uh, kk, XG, 2048)
    dx = float(XG[1] - XG[0])
    K = round(T / H_STEP)
    t0 = time.perf_counter()

    print("PILOT (selection only, key 1601, 32 replicates per arm per cell)\n")
    pilot, cost = {}, {}
    for N in CELLS:
        init = TP.initialize(N)[:3]
        for arm in ARMS:                       # interleaved by replicate below
            pilot[(N, arm)] = []
            cost[(N, arm)] = []
        for i in range(PILOT):
            for arm in ARMS:
                s = time.perf_counter()
                f = one_field(arm, N, init, K, seed_of(PILOT_KEY, arm, i))
                cost[(N, arm)].append(time.perf_counter() - s)
                pilot[(N, arm)].append(f)
        for arm in ARMS:
            pilot[(N, arm)] = np.array(pilot[(N, arm)])
    t_pilot = time.perf_counter() - t0

    stats_ = {}
    for N in CELLS:
        for arm in ARMS:
            F = pilot[(N, arm)]
            V = float(dx * np.sum(F.var(axis=0, ddof=1)))
            th = float(dx * np.sum((F.mean(0) - ref) ** 2))
            b2 = th - V / PILOT
            # upper endpoints: chi-square for V, pivotal bootstrap for b2
            V_hi = V * (PILOT - 1) / stats.chi2.ppf(0.025, PILOT - 1)
            rng = np.random.default_rng(1603)
            idx = rng.integers(0, PILOT, (2000, PILOT))
            Ts = np.array([dx * np.sum((F[i].mean(0) - ref) ** 2)
                           - dx * np.sum(F[i].var(axis=0, ddof=1)) / PILOT for i in idx])
            b2_hi = b2 + th - np.percentile(Ts, 2.5)
            stats_[(N, arm)] = dict(V=V, V_hi=float(V_hi), b2=b2, b2_hi=float(b2_hi),
                                    C=float(np.median(cost[(N, arm)])))
        print(f"  N={N}: " + "  ".join(
            f"{a} V={stats_[(N,a)]['V']:.2e} C={stats_[(N,a)]['C']:.4f}s" for a in ARMS))
        print(f"        b2: FULL(shared for SINGLE/RAW/FULL)={stats_[(N,'FULL')]['b2']:.3e} "
              f"[hi {stats_[(N,'FULL')]['b2_hi']:.3e}]   "
              f"RQMC(own)={stats_[(N,'RQMC')]['b2']:.3e} [hi {stats_[(N,'RQMC')]['b2_hi']:.3e}]")

    print(f"\nPLANS (tol={TOL:.1e}, kappa={KAPPA}, upper pilot endpoints)")
    tgt = TOL / KAPPA
    plans, proj = {}, 0.0
    for arm in ARMS:
        opts = []
        for N in CELLS:
            b2h = stats_[(N, 'RQMC' if arm == 'RQMC' else 'FULL')]['b2_hi']
            if b2h >= tgt:
                continue
            B = int(np.ceil(stats_[(N, arm)]['V_hi'] / (tgt - b2h)))
            if B > B_MAX:
                continue
            opts.append((B * stats_[(N, arm)]['C'], N, B))
        if not opts:
            plans[arm] = None
            print(f"  {arm:>7}: NO FEASIBLE CELL within B_max={B_MAX} -> reported infeasible")
            continue
        w, N, B = min(opts)
        plans[arm] = dict(N=N, B=B, projected_work=w)
        proj += w * BLOCKS
        print(f"  {arm:>7}: N={N} B={B}  projected work/block {w:.3f}s")
    print(f"\n  projected evaluation cost: {proj:.0f}s ({proj/60:.1f} min); "
          f"pilot took {t_pilot:.0f}s; cap {BUDGET_S:.0f}s")
    if proj > BUDGET_S:
        print("  -> OVER BUDGET. Evaluation NOT RUN and NOT trimmed, as predeclared.")
        json.dump(dict(status='not_run_over_budget', projected_s=proj,
                       pilot_s=t_pilot, plans={a: plans[a] for a in ARMS},
                       stats={f'{n}_{a}': stats_[(n, a)] for n in CELLS for a in ARMS}),
                  open(OUT / 'compare.json', 'w'), indent=1)
        return

    print(f"\nEVALUATION (key 1602, {BLOCKS} disjoint blocks, plans frozen)\n")
    res, meas = {}, {}
    for arm in ARMS:
        if plans[arm] is None:
            continue
        N, B = plans[arm]['N'], plans[arm]['B']
        init = TP.initialize(N)[:3]
        errs, secs = [], []
        for b in range(BLOCKS):
            acc = np.zeros(len(XG)); s = time.perf_counter()
            for r in range(B):
                acc += one_field(arm, N, init, K, seed_of(EVAL_KEY, arm, b, r))
            secs.append(time.perf_counter() - s)
            errs.append(float(dx * np.sum((acc / B - ref) ** 2)))
        errs = np.array(errs); res[arm] = errs; meas[arm] = float(np.median(secs))
        m = errs.mean(); se = errs.std(ddof=1) / np.sqrt(BLOCKS)
        t = stats.t.ppf(0.975, BLOCKS - 1)
        lo, hi = m - t * se, m + t * se
        verdict = 'ATTAINED' if hi < TOL else ('FAILED' if lo > TOL else 'unresolved')
        print(f"  {arm:>7} N={N} B={B:3d}: error {m:.4e} [{lo:.3e},{hi:.3e}] "
              f"{verdict:>11}  work {meas[arm]:.3f}s  blocks<tol {int((errs<TOL).sum())}/{BLOCKS}")
    json.dump(dict(status='complete', tol=TOL, kappa=KAPPA, cells=CELLS,
                   plans={a: plans[a] for a in ARMS},
                   pilot=dict(key=PILOT_KEY, n=PILOT, seconds=t_pilot),
                   evaluation=dict(key=EVAL_KEY, blocks=BLOCKS),
                   stats={f'{n}_{a}': stats_[(n, a)] for n in CELLS for a in ARMS},
                   realised={a: dict(mean=float(v.mean()),
                                     se=float(v.std(ddof=1) / np.sqrt(BLOCKS)),
                                     per_block=v.tolist(),
                                     work_s=meas[a]) for a, v in res.items()},
                   inference='pointwise 95% t per arm; joint claims need Bonferroni alpha/4',
                   note='RQMC arm = randomized rank-based diffusion adaptation '
                        'inspired by Lecot Eqs.(44)-(45) on our carried-velocity '
                        'transport solver; NOT a reproduction of his complete method'),
              open(OUT / 'compare.json', 'w'), indent=1)
    print(f"\ntotal {time.perf_counter()-t0:.0f}s   saved -> {OUT}")


if __name__ == '__main__':
    main()
