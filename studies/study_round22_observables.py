"""Round-22 part 2: the within-sign comparator, and whether nonlinear observables
become more ACCURATE or merely less variable.

(A) WITHIN-SIGN PAIRING is the natural competitor a referee will propose: match
    positive particles with positive and negative with negative, then reflect.
    It answers "what does spatial-rank switching buy beyond simply separating the
    signs before pairing?" -- a different question from the gain over reflection.

(B) For a NONLINEAR observable Q, Q((U_A+U_B)/2) and (Q(U_A)+Q(U_B))/2 have
    DIFFERENT expectations. Marginal-law preservation makes the second unbiased
    for E[Q(single run)]; it gives the first no coupling-independent bias. So a
    variance ratio is not an accuracy ratio. We measure bias and total error
    against the independently computed spectral reference.

Observable definitions, stated because they are grid- and rule-dependent:
  max|grad u| : central differences of the reconstructed field on the output grid
                (a grid-dependent functional; reported as such)
  level x(u=L): the FIRST downward crossing of level L to the right of the global
                maximum, by linear interpolation. If no such crossing exists the
                replicate is recorded as censored (count reported).
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import reconstruct_cumulative_field
from studies.study_round06_cubic import advance_pair_flux, XG
from studies.round06_spectral_reference import solve, evaluate
from studies import twopulse_reference as TP

OUT = ROOT / 'output/round22_corrections_2026_09_10'
NU, T, H, REPS = 0.1, 1.0, 0.005, 48
ARM_ID = {'RAW': 2, 'FULL': 3, 'WITHIN': 4}


def run(arm, N, key, init, K):
    x0, m0, ul = init; fa, fb = [], []
    for r in range(REPS):
        s = np.random.default_rng(np.random.SeedSequence([key, ARM_ID[arm], r]))
        A, B, u = advance_pair_flux((x0, m0, ul), NU, H, K, s, arm, 'burgers')
        fa.append(reconstruct_cumulative_field(A[0], A[1], u, XG))
        fb.append(reconstruct_cumulative_field(B[0], B[1], u, XG))
    return np.array(fa), np.array(fb)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    N = 8192; K = round(T / H); dx = float(XG[1] - XG[0]); t0 = time.perf_counter()
    _, _, uh, kk = solve('burgers', T, NU, M=2048, dt=1e-4)
    ref = evaluate(uh, kk, XG, 2048)
    init = TP.initialize(N)[:3]
    A = {a: run(a, N, 2200, init, K) for a in ('RAW', 'FULL', 'WITHIN')}
    PM = {a: 0.5 * (A[a][0] + A[a][1]) for a in A}
    print(f"generated {REPS} replicates x 3 arms at N={N} "
          f"({time.perf_counter()-t0:.0f}s)\n")

    def pg(M): return np.max(np.abs(np.gradient(M, dx, axis=1)), axis=1)
    def lev(M, L=0.25):
        o, cens = [], 0
        M = np.atleast_2d(M)
        for a in M:
            i = np.argmax(a)
            if not np.any(a[i:] < L):
                cens += 1; o.append(np.nan); continue
            j = i + np.argmax(a[i:] < L)
            o.append(np.interp(-L, -a[j - 1:j + 1], XG[j - 1:j + 1]))
        return np.array(o), cens

    print("(A) WITHIN-SIGN PAIRING: what does rank SWITCHING buy over separating")
    print("    the signs first?  Integrated squared field error, and its variance.\n")
    f = stats.f.ppf(0.975, REPS - 1, REPS - 1)
    mse = {a: float(np.mean(dx * np.sum((PM[a] - ref) ** 2, axis=1))) for a in PM}
    vint = {a: float(dx * np.sum(PM[a].var(0, ddof=1))) for a in PM}
    print(f"   {'arm':>8} {'mean int sq err':>16} {'vs RAW':>8} {'int variance':>14} {'vs RAW':>8}")
    for a in ('RAW', 'WITHIN', 'FULL'):
        print(f"   {a:>8} {mse[a]:16.4e} {mse[a]/mse['RAW']:8.4f} "
              f"{vint[a]:14.4e} {vint[a]/vint['RAW']:8.4f}")
    print(f"\n   FULL vs WITHIN: integrated variance ratio "
          f"{vint['FULL']/vint['WITHIN']:.4f}; mean squared error ratio "
          f"{mse['FULL']/mse['WITHIN']:.4f}")

    print("\n(B) NONLINEAR OBSERVABLES: variance, bias and TOTAL error\n")
    rows = []
    for name, fn, L in (('max|grad u|', pg, None), ('level x(u=0.25)', lev, 0.25)):
        qref = fn(ref[None, :])[0][0] if L else fn(ref[None, :])[0]
        qref = float(np.atleast_1d(qref)[0])
        print(f"   {name}   reference value = {qref:.5f}")
        print(f"     {'arm':>7} {'estimator':>11} {'mean':>10} {'bias':>11} "
              f"{'variance':>11} {'total MSE':>11} {'MSE vs RAW':>11}")
        base = {}
        for a in ('RAW', 'WITHIN', 'FULL'):
            for est, vals in (('Q_of_mean', fn(PM[a])),
                              ('mean_of_Q', None)):
                if est == 'mean_of_Q':
                    qa = fn(A[a][0]); qb = fn(A[a][1])
                    v = 0.5 * (np.atleast_1d(qa[0] if L else qa)
                               + np.atleast_1d(qb[0] if L else qb))
                else:
                    v = np.atleast_1d(vals[0] if L else vals)
                v = v[~np.isnan(v)]
                b = float(v.mean() - qref); var = float(v.var(ddof=1))
                m = b * b + var
                base.setdefault(est, m)
                print(f"     {a:>7} {est:>11} {v.mean():10.5f} {b:+11.3e} "
                      f"{var:11.3e} {m:11.3e} {m/base[est]:11.4f}")
                rows.append(dict(obs=name, arm=a, estimator=est, mean=float(v.mean()),
                                 bias=b, var=var, mse=m, n=len(v)))
        print()
    print("   Reading: the two estimators have different means, so a variance ratio")
    print("   is not an accuracy ratio. Total MSE is what a user cares about and is")
    print("   reported alongside.")
    json.dump(dict(N=N, reps=REPS, mse=mse, int_var=vint, nonlinear=rows,
                   note='within-sign comparator restored; nonlinear observables '
                        'reported with bias and total MSE, not variance alone'),
              open(OUT / 'observables.json', 'w'), indent=1)
    print(f"   {time.perf_counter()-t0:.0f}s   saved -> {OUT}")


if __name__ == '__main__':
    main()
