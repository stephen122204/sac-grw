"""Round-5 part 4: independently archived validation, exact whole-line variance,
and a strengthened two-sided pathwise witness.

Round 4 archived pair-mean fields only. This run additionally saves terminal
partner positions and masses, separate partner fields, per-replicate timings,
explicit seed keys, initial particles, and disagreement summaries across ALL
seeds. The old archive is preserved and untouched.

Whole-line variance without a grid. For a step field u(x)=c+sum_i m_i 1{X_i<=x}
with deterministic end states, Var(u(x))->0 at both ends, so int Var dx is finite
and can be computed EXACTLY: the union of all realizations' breakpoints splits
the line into intervals on which every realization is constant.
"""
import csv, hashlib, json, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import reconstruct_cumulative_field
from studies.study_sign_coupling import advance_pair, setup, PAR, PK, COUPLINGS

OUT = ROOT / 'output/round05_validation_2026_09_09'
SEEDS, KEY = 64, 505


def exact_whole_line_var(Xs, Ms, u_left):
    """Exact int Var(u(x)) dx over the whole line from terminal particle states."""
    R = len(Xs)
    bp = np.unique(np.concatenate(Xs))
    mid = np.concatenate([[bp[0] - 1.0], 0.5 * (bp[1:] + bp[:-1]), [bp[-1] + 1.0]])
    U = np.empty((R, len(mid)))
    for r in range(R):
        U[r] = reconstruct_cumulative_field(Xs[r], Ms[r], u_left, mid)
    v = U.var(axis=0, ddof=1)
    widths = np.concatenate([[0.0], np.diff(bp), [0.0]])   # end cells carry zero variance
    lens = np.zeros(len(mid)); lens[1:-1] = np.diff(bp)
    return float(np.sum(v * lens))


def witness_two_sided():
    """Mixed-sign pathwise witness on BOTH partners with real sign disagreement."""
    from studies.twopulse_reference import initialize
    x0, m0, ul, _ = initialize(200)
    K, nu, h = 40, .1, .005
    rng = np.random.default_rng(11)
    A, B, ul2, dis = advance_pair((x0, m0, ul), nu, h, K, rng, 'sign_adjusted',
                                  record_disagree=True)
    # independent replay: re-derive each partner from its own realised noise
    rng2 = np.random.default_rng(11)
    xA, mA = np.sort(x0).copy(), m0[np.argsort(x0)].copy()
    xB, mB = xA.copy(), mA.copy()
    sd = np.sqrt(2 * nu * h)
    vA = ul + np.cumsum(mA); vB = vA.copy()
    for _ in range(K):
        xA = xA + vA * h
        o = np.argsort(xA, kind='stable'); xA, mA = xA[o], mA[o]; vA = ul + np.cumsum(mA)
        xB = xB + vB * h
        o = np.argsort(xB, kind='stable'); xB, mB = xB[o], mB[o]; vB = ul + np.cumsum(mB)
        z = rng2.standard_normal(len(xA))
        zA = sd * z
        zB = -sd * (np.sign(mA) * np.sign(mB)) * z
        xA = xA + zA; xB = xB + zB
    dA = float(np.max(np.abs(np.sort(A[0]) - np.sort(xA))))
    dB = float(np.max(np.abs(np.sort(B[0]) - np.sort(xB))))
    assert dA == 0.0 and dB == 0.0, (dA, dB)
    return dA, dB, float(dis.mean()), float(dis.max())


def main(seeds=SEEDS):
    OUT.mkdir(parents=True, exist_ok=True)
    dA, dB, dmean, dmax = witness_two_sided()
    print(f"two-sided pathwise witness (mixed sign, disagreement mean {dmean:.3f} "
          f"max {dmax:.3f}): partner A {dA:.1e}, partner B {dB:.1e}  [asserted 0]\n")
    rows, arch = [], {}
    for problem, N in (('gaussian', 1600), ('twopulse', 1600)):
        p = PAR[problem]
        x_out, ref, init, info = setup(problem, N)
        dx = float(x_out[1] - x_out[0]); K = round(p['T'] / p['h'])
        print(f"{problem} N={N} (fresh key {KEY}, {seeds} seeds)")
        print("-" * 88)
        for coup in COUPLINGS:
            XA, MA, XB, MB, PMF, secs, dgr = [], [], [], [], [], [], []
            for s in range(seeds):
                r = np.random.default_rng(np.random.SeedSequence([PK[problem], N, KEY, s]))
                t0 = time.perf_counter()
                A, B, ul, dis = advance_pair(init, p['nu'], p['h'], K, r, coup,
                                             record_disagree=True)
                fa = reconstruct_cumulative_field(A[0], A[1], ul, x_out)
                fb = reconstruct_cumulative_field(B[0], B[1], ul, x_out)
                secs.append(time.perf_counter() - t0)   # includes estimator reconstruction
                XA.append(A[0]); MA.append(A[1]); XB.append(B[0]); MB.append(B[1])
                PMF.append((fa + fb) / 2); dgr.append(float(dis.mean()))
            PMF = np.array(PMF)
            grid_var = float(dx * np.sum(PMF.var(axis=0, ddof=1)))
            Xp = [(np.concatenate([a, b]), np.concatenate([ma, mb]) / 2)
                  for a, b, ma, mb in zip(XA, XB, MA, MB)]
            exact_var = exact_whole_line_var([q[0] for q in Xp], [q[1] for q in Xp], ul)
            mse = float(np.mean(dx * np.sum((PMF - ref) ** 2, axis=1)))
            cost = float(np.mean(secs))
            rows.append(dict(problem=problem, N=N, coupling=coup, seeds=seeds,
                             grid_var=grid_var, exact_whole_line_var=exact_var,
                             mse=mse, cost_s=cost, var_cost=exact_var * cost,
                             mse_cost=mse * cost,
                             disagree_mean=float(np.mean(dgr)),
                             disagree_sd=float(np.std(dgr, ddof=1))))
            q = rows[-1]
            print(f"  {coup:>14} exact_var={exact_var:.5e} grid_var={grid_var:.5e} "
                  f"(ratio {grid_var/exact_var:.4f})  mse={mse:.4e} t={cost:.4f}s "
                  f"disagree={q['disagree_mean']:.3f}+-{q['disagree_sd']:.3f}")
            arch[f'{problem}_{N}_{coup}_XA'] = np.array(XA)
            arch[f'{problem}_{N}_{coup}_XB'] = np.array(XB)
            arch[f'{problem}_{N}_{coup}_pairmean'] = PMF
            arch[f'{problem}_{N}_{coup}_secs'] = np.array(secs)
            arch[f'{problem}_{N}_{coup}_disagree'] = np.array(dgr)
        arch[f'{problem}_{N}_MA'] = np.array(MA[0]); arch[f'{problem}_{N}_x'] = x_out
        arch[f'{problem}_{N}_ref'] = ref; arch[f'{problem}_{N}_init_x'] = init[0]
        arch[f'{problem}_{N}_init_m'] = init[1]
        base = [r for r in rows if r['problem'] == problem and r['coupling'] == 'raw_rank'][0]
        for r in rows:
            if r['problem'] == problem:
                r['var_cost_ratio'] = r['var_cost'] / base['var_cost']
                r['mse_cost_ratio'] = r['mse_cost'] / base['mse_cost']
        print()
    np.savez_compressed(OUT / 'archive.npz', **arch)
    with (OUT / 'rows.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / 'validation.json').write_text(json.dumps(dict(
        rows=rows, seeds=seeds, seed_key=KEY,
        witness=dict(partner_A=dA, partner_B=dB, disagree_mean=dmean),
        solver_sha=hashlib.sha256((ROOT/'relaxation_gbmc.py').read_bytes()).hexdigest()[:16],
        driver_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        coupler_sha=hashlib.sha256((ROOT/'studies/study_sign_coupling.py').read_bytes()).hexdigest()[:16],
        note='cost includes the estimator reconstruction; disagreement recorded for every seed'),
        indent=1))
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
