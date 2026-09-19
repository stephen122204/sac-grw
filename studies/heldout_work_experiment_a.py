"""Held-out work comparison against unpaired, reflected, and RQMC estimators (Section 4.5, Experiment A)."""
import json, os, platform, sys, time
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradient_particles import reconstruct_cumulative_field
from studies.rqmc_diffusion import RQMCDiffusion, advance_rank_diffusion
from studies.cubic_flux_transfer import advance_pair_flux, XG
from studies.spectral_reference import solve, evaluate, FPRIME
from studies.work_target_validation import advance_single_flux
from studies import twopulse_reference as TP

OUT = ROOT / 'output/heldout_work_experiment_a'
NU, T, H, TOL, KAPPA = 0.1, 1.0, 0.005, 7e-6, 1.3
CELLS, ARMS = [2048, 8192], ('SINGLE', 'RAW', 'FULL', 'RQMC')
ARM_ID = {'SINGLE': 1, 'RAW': 2, 'FULL': 3, 'RQMC': 4}
PILOT_KEY, EVAL_KEY, PILOT, BLOCKS = 1601, 1702, 32, 32
B_MAX, GATE_S, HARD_S, BOOT = 200, 1200.0, 1500.0, 4000


def ss(key, arm, i, j=0):
    return np.random.SeedSequence([key, ARM_ID[arm], i, j])


def field(arm, N, init, K, s):
    x0, m0, ul = init
    if arm == 'SINGLE':
        x, m, u = advance_single_flux((x0, m0, ul), NU, H, K,
                                      np.random.default_rng(s), 'burgers')
        return reconstruct_cumulative_field(x, m, u, XG)
    if arm == 'RQMC':
        q = RQMCDiffusion(N, seed=int(s.generate_state(1)[0]), debug=False)
        x, m, u, _ = advance_rank_diffusion((x0, m0, ul), NU, H, K, q, FPRIME['burgers'])
        return reconstruct_cumulative_field(x, m, u, XG)
    A, B, u = advance_pair_flux((x0, m0, ul), NU, H, K, np.random.default_rng(s),
                                arm, 'burgers')
    return 0.5 * (reconstruct_cumulative_field(A[0], A[1], u, XG)
                  + reconstruct_cumulative_field(B[0], B[1], u, XG))


def heat_fixture():
    """Archived analytic one-step heat check with probe-wise data retained."""
    N, S, sig = 64, 4000, 0.05
    xd = np.linspace(-1, 1, N); md = np.where(np.arange(N) % 2 == 0, 1., -1.) / N
    md -= md.mean(); g = np.linspace(-3, 3, 241)
    exact = (md[:, None] * norm.cdf((g[None, :] - xd[:, None]) / sig)).sum(0)
    A = np.empty((S, len(g)))
    for s in range(S):
        q = RQMCDiffusion(N, seed=10_000 + s, debug=True)
        A[s] = reconstruct_cumulative_field(xd + q.increments(sig), md, 0.0, g)
    mean = A.mean(0); se = A.std(0, ddof=1) / np.sqrt(S)
    nondeg = se > 0
    t = np.zeros_like(mean); t[nondeg] = (mean[nondeg] - exact[nondeg]) / se[nondeg]
    k = int(nondeg.sum())
    crit = stats.norm.ppf(1 - 0.025 / max(k, 1))          # Bonferroni over nondegenerate probes
    rng = np.random.default_rng(17)
    C = A - A.mean(0)
    boot = np.array([np.max(np.abs(C[rng.integers(0, S, S)][:, nondeg].mean(0)
                                   / se[nondeg])) for _ in range(1000)])
    return dict(N=N, S=S, sigma=sig, n_probes=len(g), n_nondegenerate=k,
                n_zero_variance=int((~nondeg).sum()),
                max_abs_t=float(np.max(np.abs(t))), bonferroni_crit=float(crit),
                simultaneous_boot_crit_95=float(np.percentile(boot, 95)),
                exceedances=int(np.sum(np.abs(t) > crit)),
                probes=g.tolist(), mean=mean.tolist(), analytic=exact.tolist(),
                se=se.tolist(), nondegenerate=nondeg.tolist(),
                note='finite-sample diagnostic of the one-step estimator mean; '
                     'multiplicity handled by Bonferroni over nondegenerate probes '
                     'and by a bootstrap simultaneous max-|t| critical value. '
                     'Degenerate (zero empirical variance) probes are retained '
                     'separately and excluded from the t statistics, not dropped '
                     'to obtain a desired result. This does NOT establish '
                     'multistep conditional Brownian laws.')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    _, _, uh, kk = solve('burgers', T, NU, M=2048, dt=1e-4)
    ref = evaluate(uh, kk, XG, 2048); dx = float(XG[1] - XG[0]); K = round(T / H)

    print("FIXTURE: analytic one-step heat check (archived probe-wise)")
    hf = heat_fixture()
    print(f"   {hf['n_nondegenerate']} nondegenerate of {hf['n_probes']} probes "
          f"({hf['n_zero_variance']} zero-variance, retained separately)")
    print(f"   max|t| = {hf['max_abs_t']:.2f}; Bonferroni crit {hf['bonferroni_crit']:.2f}; "
          f"bootstrap simultaneous crit {hf['simultaneous_boot_crit_95']:.2f}; "
          f"exceedances {hf['exceedances']}")

    print("\nPILOT replay (key 1601) + re-measured timings, debug OFF, balanced order")
    old = json.load(open(ROOT / 'output/heldout_work_pilot/compare.json'))['stats']
    pilot, cost, prov = {}, {}, None
    for N in CELLS:
        init = TP.initialize(N)[:3]
        for a in ARMS:
            pilot[(N, a)] = []; cost[(N, a)] = []
        order = list(ARMS)
        for i in range(PILOT):
            for a in (order if i % 2 == 0 else order[::-1]):   # balanced order
                t0 = time.perf_counter()
                f = field(a, N, init, K, ss(PILOT_KEY, a, i))
                cost[(N, a)].append(time.perf_counter() - t0)
                pilot[(N, a)].append(f)
        for a in ARMS:
            pilot[(N, a)] = np.array(pilot[(N, a)])
    prov = RQMCDiffusion.provenance()

    st, rng = {}, np.random.default_rng(1703)
    idx = rng.integers(0, PILOT, (BOOT, PILOT))
    print(f"   {'cell':>6} {'arm':>7} {'V':>11} {'V_hi(boot)':>11} {'C new':>8} "
          f"{'C r16':>8} {'V replay chk':>13}")
    for N in CELLS:
        for a in ARMS:
            F = pilot[(N, a)]
            V = float(dx * np.sum(F.var(axis=0, ddof=1)))
            Vb = np.array([dx * np.sum(F[i].var(axis=0, ddof=1)) for i in idx])
            V_hi = float(np.percentile(Vb, 97.5))
            th = float(dx * np.sum((F.mean(0) - ref) ** 2)); b2 = th - V / PILOT
            Tb = np.array([dx * np.sum((F[i].mean(0) - ref) ** 2)
                           - dx * np.sum(F[i].var(axis=0, ddof=1)) / PILOT for i in idx])
            b2_hi = float(b2 + th - np.percentile(Tb, 2.5))
            C = float(np.median(cost[(N, a)]))
            st[(N, a)] = dict(V=V, V_hi=V_hi, b2=b2, b2_hi=b2_hi, C=C,
                              C_pilot=old[f'{N}_{a}']['C'])
            rel = abs(V - old[f'{N}_{a}']['V']) / old[f'{N}_{a}']['V']
            print(f"   {N:>6} {a:>7} {V:11.3e} {V_hi:11.3e} {C:8.4f} "
                  f"{old[f'{N}_{a}']['C']:8.4f} {rel:13.2e}")
    print("   (V replay chk = |V_new - V_pilot|/V_pilot; timings differ by design)")

    tgt = TOL / KAPPA; plans, proj = {}, 0.0
    print(f"\nPLANS  tol={TOL:.1e} (PILOT-INFORMED), kappa={KAPPA}, bootstrap endpoints")
    for a in ARMS:
        opts = []
        for N in CELLS:
            b2h = st[(N, 'RQMC' if a == 'RQMC' else 'FULL')]['b2_hi']
            if b2h >= tgt: continue
            B = int(np.ceil(st[(N, a)]['V_hi'] / (tgt - b2h)))
            if B > B_MAX: continue
            opts.append((B * st[(N, a)]['C'], N, B))
        if not opts:
            plans[a] = None; print(f"   {a:>7}: NO FEASIBLE PLAN (reported, not skipped)")
            continue
        w, N, B = min(opts); plans[a] = dict(N=N, B=B, work=w); proj += w * BLOCKS
        print(f"   {a:>7}: N={N} B={B:3d}  work/block {w:.3f}s")
    print(f"\n   projected evaluation {proj:.0f}s ({proj/60:.1f} min); gate {GATE_S:.0f}s")

    meta = dict(target=TOL, target_provenance='PILOT-INFORMED after round-16; the '
                'original 5e-6 experiment was never run (budget gate)',
                kappa=KAPPA, cells=CELLS, arms=list(ARMS), pilot_key=PILOT_KEY,
                eval_key=EVAL_KEY, blocks=BLOCKS, B_max=B_MAX, boot=BOOT,
                endpoints='whole-replicate bootstrap, APPROXIMATE at 32 replicates; '
                          'b2 pivotal upper = b2 + th - q2.5(T*), centred on the '
                          'bootstrap-world plug-in th, valid asymptotically and '
                          'poorly determined near zero bias',
                inference='pointwise 95% t per arm; Bonferroni alpha/4 for joint claims',
                rqmc=prov, heat_fixture=hf,
                env=dict(python=platform.python_version(), numpy=np.__version__,
                         platform=platform.platform()),
                stats={f'{n}_{a}': st[(n, a)] for n in CELLS for a in ARMS},
                plans={a: plans[a] for a in ARMS}, projected_s=proj)
    np.savez_compressed(OUT / 'pilot_fields.npz',
                        **{f'{n}_{a}': pilot[(n, a)].astype(np.float32)
                           for n in CELLS for a in ARMS},
                        **{f'{n}_{a}_secs': np.array(cost[(n, a)])
                           for n in CELLS for a in ARMS},
                        x_grid=XG, reference=ref)
    if proj > GATE_S:
        meta['status'] = 'not_run_over_budget'
        json.dump(meta, open(OUT / 'compare.json', 'w'), indent=1)
        print("   -> OVER GATE. Not run, not trimmed."); return

    print(f"\nEVALUATION key {EVAL_KEY} ({BLOCKS} blocks), checkpointed per block\n")
    res, meas = {}, {}
    for a in ARMS:
        if plans[a] is None: continue
        N, B = plans[a]['N'], plans[a]['B']; init = TP.initialize(N)[:3]
        errs, secs = [], []
        for b in range(BLOCKS):
            if time.perf_counter() - t_start > HARD_S:
                print(f"   ELAPSED STOP during {a} at block {b}; partial reported")
                break
            acc = np.zeros(len(XG)); s0 = time.perf_counter()
            for r in range(B):
                acc += field(a, N, init, K, ss(EVAL_KEY, a, b, r))
            secs.append(time.perf_counter() - s0)
            errs.append(float(dx * np.sum((acc / B - ref) ** 2)))
            json.dump(dict(arm=a, block=b, err=errs[-1], sec=secs[-1]),
                      open(OUT / f'ckpt_{a}.json', 'w'))
        errs = np.array(errs); res[a] = errs; meas[a] = float(np.median(secs))
        n = len(errs); m = errs.mean(); se = errs.std(ddof=1) / np.sqrt(n)
        tc = stats.t.ppf(0.975, n - 1); lo, hi = m - tc * se, m + tc * se
        tb = stats.t.ppf(1 - 0.025 / len(ARMS), n - 1)
        slo, shi = m - tb * se, m + tb * se
        v = 'ATTAINED' if hi < TOL else ('FAILED' if lo > TOL else 'unresolved')
        vs = 'ATTAINED' if shi < TOL else ('FAILED' if slo > TOL else 'unresolved')
        print(f"   {a:>7} N={N} B={B:3d} blocks={n}: err {m:.4e} "
              f"[{lo:.3e},{hi:.3e}] {v:>11} | simultaneous {vs:>11} | "
              f"work {meas[a]:.3f}s | <tol {int((errs<TOL).sum())}/{n}")
    meta['status'] = 'complete'
    meta['realised'] = {a: dict(mean=float(v.mean()), n=len(v),
                                se=float(v.std(ddof=1) / np.sqrt(len(v))),
                                per_block=v.tolist(), work_s=meas[a])
                        for a, v in res.items()}
    meta['total_s'] = time.perf_counter() - t_start
    json.dump(meta, open(OUT / 'compare.json', 'w'), indent=1)
    print(f"\ntotal {meta['total_s']:.0f}s  saved -> {OUT}")


if __name__ == '__main__':
    main()
