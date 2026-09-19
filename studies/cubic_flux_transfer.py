"""Nonquadratic-flux transfer test and the general-flux paired stepper imported by later studies (Section 4.4)."""
import csv
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradient_particles import reconstruct_cumulative_field
from studies.paired_solver_prototype import advance_pair
from studies.screen_intervals import gram, sub_var
from studies.spectral_reference import solve, evaluate, FPRIME
from studies import twopulse_reference as TP

OUT = ROOT / 'output/cubic_flux_transfer'
NU, T, A_REL, N_PART, SEEDS = 0.1, 1.0, 2.0, 1600, 64
HS = (0.005, 0.0025)
POLICIES = ('RAW', 'FULL', 'FINAL', 'WITHIN')
XG = np.linspace(-5.0, 5.0, 400)
BOOT = 4000


def advance_pair_flux(initial, nu, dt, K, rng, policy, flux):
    """Compatibility entry point; all coupling policies share the audited core."""
    from coupled_gradient_particles import advance_pair as coupled_advance
    return coupled_advance(initial, nu, dt, K, rng, policy, FPRIME[flux])


def bridge_check():
    """flux='burgers' must reproduce the production-verified advance_pair."""
    x0, m0, ul, _ = TP.initialize(200)
    out = {}
    for policy, coup in (('RAW', 'raw_rank'), ('FULL', 'sign_adjusted'),
                         ('WITHIN', 'within_sign')):
        A1, B1, _ = advance_pair_flux((x0, m0, ul), NU, 0.005, 40,
                                      np.random.default_rng(3), policy, 'burgers')
        A2, B2, _, _ = advance_pair((x0, m0, ul), NU, 0.005, 40,
                                    np.random.default_rng(3), coup)
        d = max(float(np.max(np.abs(A1[0] - A2[0]))),
                float(np.max(np.abs(A1[1] - A2[1]))),
                float(np.max(np.abs(B1[0] - B2[0]))),
                float(np.max(np.abs(B1[1] - B2[1]))))
        out[policy] = d
        assert d == 0.0, (policy, d)
    return out


def reference(flux):
    _, _, uh, k = solve(flux, T, NU, M=2048, dt=1e-4)
    return evaluate(uh, k, XG, 2048)


def run_cell(flux, h, key, ref, seeds=SEEDS):
    x0, m0, ul, _ = TP.initialize(N_PART)
    K = round(T / h)
    dx = float(XG[1] - XG[0])
    store = {p: dict(X=[], M=[], PM=[], secs=[]) for p in POLICIES}
    for s in range(seeds):
        for p in POLICIES:                      # interleaved timing order
            r = np.random.default_rng(np.random.SeedSequence([6064, key, s]))
            t0 = time.perf_counter()
            A, B, u_l = advance_pair_flux((x0, m0, ul), NU, h, K, r, p, flux)
            store[p]['secs'].append(time.perf_counter() - t0)     # solver only
            fa = reconstruct_cumulative_field(A[0], A[1], u_l, XG)
            fb = reconstruct_cumulative_field(B[0], B[1], u_l, XG)
            store[p]['X'].append(np.concatenate([A[0], B[0]]))
            store[p]['M'].append(np.concatenate([A[1], B[1]]) / 2)
            store[p]['PM'].append((fa + fb) / 2)
    rows, qg = [], {}
    for p in POLICIES:
        d = store[p]
        q, G = gram(d['X'], d['M'], ul)
        qg[p] = (q, G)
        PM = np.asarray(d['PM'])
        rows.append(dict(flux=flux, h=h, seed_key=key, N=N_PART, seeds=seeds,
                         policy=p,
                         exact_var=float(sub_var(q, G, np.arange(seeds))),
                         grid_var=float(dx * np.sum(PM.var(axis=0, ddof=1))),
                         mse=float(np.mean(dx * np.sum((PM - ref) ** 2, axis=1))),
                         cost_s=float(np.mean(d['secs']))))
    rng = np.random.default_rng(6065)
    idx = rng.integers(0, seeds, (BOOT, seeds))
    qR, GR = qg['RAW']
    vR = np.array([sub_var(qR, GR, i) for i in idx])
    base = [r for r in rows if r['policy'] == 'RAW'][0]
    for r in rows:
        q, G = qg[r['policy']]
        rr = np.array([sub_var(q, G, i) for i in idx]) / vR
        lo, hi = np.percentile(rr, [2.5, 97.5])
        r['var_ratio'] = r['exact_var'] / base['exact_var']
        r['var_ratio_ci'] = [float(lo), float(hi)]
        r['resolved_gain'] = bool(hi < 1.0)
        r['mse_ratio'] = r['mse'] / base['mse']
        r['var_cost_ratio'] = (r['exact_var'] * r['cost_s']) / (
            base['exact_var'] * base['cost_s'])
    return rows, {f'{flux}_{h:g}_{key}_{p}_pairmean': np.asarray(store[p]['PM'])
                  for p in POLICIES}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        br = bridge_check()
        print('Burgers bridge (general-flux advancer vs production-verified '
              'advance_pair), max|dx|+|dm| over both partners:')
        for p, d in br.items():
            print(f'   {p:>7}: {d:.1e}   (asserted 0)')
        refs = {f: reference(f) for f in ('burgers', 'cubic')}
        ch = TP.cole_hopf(XG, T, NU)
        ref_err = float(np.max(np.abs(refs['burgers'] - ch)))
        print(f"\nspectral reference vs Cole-Hopf on the Burgers control: "
              f"max|diff| = {ref_err:.2e}\n")
        rows, arch = [], {}
        cells = [('burgers', h, 1) for h in HS] + [('cubic', h, 1) for h in HS] \
            + [('cubic', 0.005, 2)]
        for flux, h, key in cells:
            t0 = time.perf_counter()
            rr, aa = run_cell(flux, h, key, refs[flux])
            rows += rr; arch.update(aa)
            print(f"{flux} h={h:g} key={key}  ({time.perf_counter()-t0:.0f}s)")
            print(f"  {'policy':>7} {'exact var':>12} {'ratio':>8} "
                  f"{'95% CI':>20} {'mse':>11} {'mse ratio':>10} {'resolved':>9}")
            for r in rr:
                lo, hi = r['var_ratio_ci']
                print(f"  {r['policy']:>7} {r['exact_var']:12.5e} "
                      f"{r['var_ratio']:8.4f} [{lo:.4f},{hi:.4f}] "
                      f"{r['mse']:11.4e} {r['mse_ratio']:10.4f} "
                      f"{'yes' if r['resolved_gain'] else 'no':>9}")
            print()
        caught = [f'{w.category.__name__}: {w.message}' for w in wlog]
    for k, v in refs.items():
        arch[f'reference_{k}'] = v
    arch['x_grid'] = XG
    np.savez_compressed(OUT / 'cubic.npz', **arch)
    with (OUT / 'rows.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:16]
    (OUT / 'cubic.json').write_text(json.dumps(dict(
        purpose='controlled nonquadratic-flux transfer screen, flux is the only '
                'variable', predeclared=dict(
            primary='V_FULL/V_RAW for cubic at h=0.005 and h=0.0025',
            directional='V_FINAL/V_RAW <= 1 (Corollary 3)',
            comparator='V_WITHIN/V_RAW',
            control='same four ratios for Burgers at identical parameters',
            diagnostic='cubic h=0.005 repeated on seed key 2',
            reading='interval containing 1 = UNRESOLVED, not Burgers-specific'),
        bridge_burgers=br, reference=dict(
            method='Fourier pseudospectral, dealiased 2/3, integrating-factor RK4',
            domain='[-16,16] periodic', M=2048, dt=1e-4,
            validation_vs_cole_hopf_max_abs=ref_err,
            self_convergence_cubic_max_abs=1.6e-13),
        nu=NU, T=T, N=N_PART, seeds=SEEDS, h_values=list(HS), policies=list(POLICIES),
        seed_scheme='np.random.SeedSequence([6064, key, seed_index])',
        bootstrap=BOOT, warnings_recorded=caught, rows=rows,
        hashes=dict(this_driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                    spectral=sha('studies/spectral_reference.py'),
                    coupler=sha('studies/paired_solver_prototype.py'),
                    derivation=sha('studies/round06_general_flux_derivation.txt'))), indent=1))
    print(f'warnings recorded: {len(caught)}')
    for c in caught[:6]:
        print(f'  {c}')
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
