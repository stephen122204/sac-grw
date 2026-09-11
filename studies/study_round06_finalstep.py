"""Round-6 part 3: the FINAL-step intervention and its exact nonlinear guarantee.

Policies, identical initial particles / schedule / parameters / matching:
  RAW    ordinary rank reflection at every diffusion stage
  FINAL  ordinary rank reflection through stage K-1, sign-adjusted at stage K only
  FULL   sign-adjusted at every diffusion stage
  WITHIN rank matching inside each sign class, reflected normals (comparator)

Corollary 3.  RAW and FINAL are pathwise identical up to the final noise draw, so
they share the pre-final state S. Both couplings preserve each replica's marginal
law, hence E[Uhat(T)|S] is the same for both, and the law of total variance gives

    V_RAW(T) - V_FINAL(T) = E_RAW[ Delta(S) ] >= 0,
    Delta(S) = (1/4) sum_{m_Ai m_Bi<0} |m_Ai m_Bi| [ g_{2 sigma}(a_i-b_i) - |a_i-b_i| ]

with V the integrated CENTERED variance of the terminal pair mean. The preceding
evolution may be the nonlinear Burgers transport; the argument needs only that the
final stage is a Gaussian diffusion followed by reconstruction with no further
interacting update -- which is exactly the production schedule. The guarantee is on
variance, not variance x cost, and says nothing about FULL.
"""
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from relaxation_gbmc import reconstruct_cumulative_field
from studies.study_sign_coupling import setup, PAR, PK
from studies.study_round05_validation import exact_whole_line_var

OUT = ROOT / 'output/round06_finalstep_2026_09_09'
SEEDS = 64


def g2(s, d):
    """E|d + s Z|, closed form: |d|(2Phi(|d|/s)-1) + 2 s phi(|d|/s)."""
    if s <= 0:
        return np.abs(d)
    a = np.abs(d) / s
    return np.abs(d) * (2 * norm.cdf(a) - 1) + 2 * s * norm.pdf(a)


def delta_gain(mA, mB, a, b, sigma):
    """Proposition 1 conditional gain at a matched pre-diffusion state."""
    unlike = (mA * mB) < 0
    if not np.any(unlike):
        return 0.0
    d = a[unlike] - b[unlike]
    return float(0.25 * np.sum(np.abs(mA[unlike] * mB[unlike]) *
                               (g2(2 * sigma, d) - np.abs(d))))


def run_policy(init, nu, h, K, rng, policy):
    """One paired trajectory. Returns terminal states and the pre-final gain."""
    x0, m0, ul = np.asarray(init[0], float), np.asarray(init[1], float), init[2]
    o = np.argsort(x0, kind='stable')
    xA, mA = x0[o].copy(), m0[o].copy(); xB, mB = xA.copy(), mA.copy()
    N = len(xA); sd = np.sqrt(2.0 * nu * h)
    vA = ul + np.cumsum(mA); vB = vA.copy()
    pre_gain = None
    for k in range(K):
        xA = xA + vA * h
        o = np.argsort(xA, kind='stable'); xA, mA = xA[o], mA[o]; vA = ul + np.cumsum(mA)
        xB = xB + vB * h
        o = np.argsort(xB, kind='stable'); xB, mB = xB[o], mB[o]; vB = ul + np.cumsum(mB)
        last = (k == K - 1)
        if last:                       # conditional gain available at this state
            pre_gain = delta_gain(mA, mB, xA, xB, sd)
        if policy == 'RAW':
            use_sign = False
        elif policy == 'FULL':
            use_sign = True
        elif policy == 'FINAL':
            use_sign = last
        elif policy == 'WITHIN':
            use_sign = None            # handled below
        z = rng.standard_normal(N)
        if policy == 'WITHIN':
            zb = np.empty(N)
            for sgn in (1.0, -1.0):
                iA = np.flatnonzero(np.sign(mA) == sgn); iB = np.flatnonzero(np.sign(mB) == sgn)
                n = min(len(iA), len(iB)); zb[iB[:n]] = -z[iA[:n]]
            zB = sd * zb
        else:
            zB = -sd * (np.sign(mA) * np.sign(mB)) * z if use_sign else -sd * z
        xA = xA + sd * z; xB = xB + zB
    return (xA, mA), (xB, mB), ul, pre_gain


def main(seeds=SEEDS):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, arch = [], {}
    cases = [('gaussian', 1600, .005), ('twopulse', 1600, .005),
             ('twopulse', 1600, .0025), ('shock', 1600, .005)]
    for problem, N, h in cases:
        p = PAR[problem]; x_out, ref, init, _ = setup(problem, N)
        K = round(p['T'] / h); nu = p['nu']
        print(f"\n{problem} N={N} h={h} K={K} ({seeds} seeds)")
        print("-" * 86)
        res = {}
        for policy in ('RAW', 'FINAL', 'FULL', 'WITHIN'):
            Xs, Ms, gains, secs, PM = [], [], [], [], []
            for s in range(seeds):
                r = np.random.default_rng(np.random.SeedSequence([PK[problem], N,
                                                                  int(h * 1e6), 606, s]))
                t0 = time.perf_counter()
                A, B, ul, gain = run_policy(init, nu, h, K, r, policy)
                fa = reconstruct_cumulative_field(A[0], A[1], ul, x_out)
                fb = reconstruct_cumulative_field(B[0], B[1], ul, x_out)
                secs.append(time.perf_counter() - t0)
                Xs.append(np.concatenate([A[0], B[0]]))
                Ms.append(np.concatenate([A[1], B[1]]) / 2)
                gains.append(gain); PM.append((fa + fb) / 2)
            V = exact_whole_line_var(Xs, Ms, ul)
            res[policy] = dict(V=V, gain=float(np.mean(gains)),
                               gain_sd=float(np.std(gains, ddof=1)),
                               cost=float(np.mean(secs)), PM=np.array(PM))
            arch[f'{problem}_{N}_{h:g}_{policy}_pairmean'] = np.array(PM)
            arch[f'{problem}_{N}_{h:g}_{policy}_gain'] = np.array(gains)
            print(f"  {policy:>7} V={V:.6e}  E[Delta]={res[policy]['gain']:.6e} "
                  f"cost={res[policy]['cost']:.4f}s")
        dRF = res['RAW']['V'] - res['FINAL']['V']
        dFF = res['FINAL']['V'] - res['FULL']['V']
        dRFu = res['RAW']['V'] - res['FULL']['V']
        EG = res['RAW']['gain']                       # E[Delta] under the RAW history
        se = res['RAW']['gain_sd'] / np.sqrt(seeds)
        print(f"  Corollary 3 check:  V_RAW - V_FINAL = {dRF:.6e}   "
              f"E[Delta] = {EG:.6e} +- {1.96*se:.2e}")
        print(f"  decomposition:  V_RAW - V_FULL = {dRFu:.6e} = "
              f"(RAW-FINAL) {dRF:.6e} + (FINAL-FULL) {dFF:.6e}")
        share = dRF / dRFu if dRFu else float('nan')
        print(f"  guaranteed final-step share of the total gain = {share:.3f}")
        rows.append(dict(problem=problem, N=N, h=h, K=K, seeds=seeds,
                         V_RAW=res['RAW']['V'], V_FINAL=res['FINAL']['V'],
                         V_FULL=res['FULL']['V'], V_WITHIN=res['WITHIN']['V'],
                         E_Delta=EG, E_Delta_se=float(se),
                         RAW_minus_FINAL=dRF, FINAL_minus_FULL=dFF,
                         RAW_minus_FULL=dRFu, final_step_share=share,
                         cost_RAW=res['RAW']['cost'], cost_FULL=res['FULL']['cost'],
                         cost_FINAL=res['FINAL']['cost']))
    np.savez_compressed(OUT / 'archive.npz', **arch)
    with (OUT / 'rows.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / 'finalstep.json').write_text(json.dumps(dict(rows=rows, seeds=seeds,
        solver_sha=hashlib.sha256((ROOT/'relaxation_gbmc.py').read_bytes()).hexdigest()[:16],
        driver_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]), indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
