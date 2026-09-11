"""Round-8: does sign-aware coupling add value beyond, or alongside, an
inexpensive known-expectation control?

PREDECLARED BEFORE ANY RESULT WAS INSPECTED
  problem       two-pulse, nu=0.1, T=1, N=1600, h=0.005 (K=200)
  fluxes        burgers f=u^2/2 and cubic f=u^3/3, same inclusive schedule
  training      seed key 8001, 16 seeds, used ONLY to fit one scalar c per
                (flux, policy); frozen thereafter
  evaluation    seed key 8002, 64 fresh seeds, disjoint from training
  observable    terminal pair-mean field on a COMMON grid and window for every
                arm: x in [-8, 8], 801 points; grid/window convergence reported
  arms          RAW, FULL, WITHIN (context);
                RAW+CV(c=1), FULL+CV(c=1);
                RAW+CV(c*), FULL+CV(c*);
                RAW+RB, FULL+RB
  metrics       integrated centred field variance; complete cost including
                bookkeeping, mu_H evaluation and training (one-off and
                amortised, reuse count stated); variance x cost; MSE against
                the validated spectral reference
  reading       this asks whether coupling helps WITHOUT a reference and
                whether it still helps WITH one. A control beating FULL does
                not demote the method, and FULL beating this one heat control
                does not establish superiority over deviational methods.

THE CONTROL. U_CV = Ubar - c (Hbar - mu_H) with H the identity-tracked heat
surrogate. This is the standard correlated-sampling control variate; its closest
particle-method realisation is Al-Mohssen & Hadjiconstantinou (2010) Eq. (2.1),
R^VR = R - R_eq + <R_eq>. It is NOT LVDSMC and nothing Boltzmann-specific is
imported. For fixed c it preserves the expected terminal field of the ORIGINAL
DISCRETE SOLVER, including its discretisation bias; mu_H uses the discrete
initial particles, never the continuum profile and never the spectral solution.

c* MINIMISES INTEGRATED VARIANCE:
    d/dc int Var(U - c(H - mu_H)) dx = 0  =>  c* = int Cov(U,H) dx / int Var(H) dx,
one scalar per (flux, policy), fitted on the training batch only.
"""
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
from studies.round08_estimators import (advance_pair_instrumented, pair_field,
                                        heat_pair_field, mu_heat, rb_pair_field)
from studies.study_round06_cubic import advance_pair_flux
from studies.round06_spectral_reference import solve, evaluate
from studies import twopulse_reference as TP

OUT = ROOT / 'output/round08_controls_2026_09_10'
NU, T, N_PART, H_STEP = 0.1, 1.0, 1600, 0.005
K_STEPS = round(T / H_STEP)
FLUXES = ('burgers', 'cubic')
POLICIES = ('RAW', 'FULL', 'WITHIN')
TRAIN_KEY, EVAL_KEY, TRAIN_SEEDS, EVAL_SEEDS = 8001, 8002, 16, 64
XG = np.linspace(-8.0, 8.0, 801)
BOOT = 4000


def batch(flux, policy, key, seeds, x0, m0, ul, grid=XG, timed=False):
    """Run `seeds` paired replicas, returning U, H and RB fields plus timings."""
    U, H, RB, t_solve, t_U, t_H, t_RB = [], [], [], [], [], [], []
    for s in range(seeds):
        r = np.random.default_rng(np.random.SeedSequence([key, s]))
        t0 = time.perf_counter()
        o = advance_pair_instrumented((x0, m0, ul), NU, H_STEP, K_STEPS, r,
                                      policy, flux)
        t1 = time.perf_counter(); t_solve.append(t1 - t0)
        u = pair_field(o, grid); t2 = time.perf_counter(); t_U.append(t2 - t1)
        h = heat_pair_field(o, x0, m0, grid)
        t3 = time.perf_counter(); t_H.append(t3 - t2)
        rb = rb_pair_field(o, NU, H_STEP, grid)
        t_RB.append(time.perf_counter() - t3)
        U.append(u); H.append(h); RB.append(rb)
    return (np.array(U), np.array(H), np.array(RB),
            dict(solve_instr=float(np.mean(t_solve)), U=float(np.mean(t_U)),
                 H=float(np.mean(t_H)), RB=float(np.mean(t_RB))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        x0, m0, ul, _ = TP.initialize(N_PART)
        dx = float(XG[1] - XG[0])

        # ---------- fixtures ----------
        print("FIXTURES")
        xs, ms, uls, _ = TP.initialize(400)
        for pol in POLICIES:
            a = advance_pair_instrumented((xs, ms, uls), NU, H_STEP, 40,
                                          np.random.default_rng(3), pol, 'burgers')
            b = advance_pair_flux((xs, ms, uls), NU, H_STEP, 40,
                                  np.random.default_rng(3), pol, 'burgers')
            d = max(float(np.max(np.abs(a['xA'] - b[0][0]))),
                    float(np.max(np.abs(a['mA'] - b[0][1]))),
                    float(np.max(np.abs(a['xB'] - b[1][0]))),
                    float(np.max(np.abs(a['mB'] - b[1][1]))))
            assert d == 0.0
        print(f"  instrumented paths vs production-bridged advancer, positions "
              f"and signed masses, both partners: 0.0e+00 for all policies")
        o = advance_pair_instrumented((xs, ms, uls), NU, H_STEP, 200,
                                      np.random.default_rng(9), 'FULL', 'burgers',
                                      transport=False)
        muS = mu_heat(xs, ms, uls, NU, T, XG)
        pure = float(np.max(np.abs(pair_field(o, XG)
                                   - (heat_pair_field(o, xs, ms, XG) - muS) - muS)))
        print(f"  pure-heat fixture, c=1: max|U - (H - mu_H) - mu_H| = {pure:.1e}")
        assert pure < 1e-15

        t0 = time.perf_counter()
        _, _, uh, kk = solve('burgers', T, NU, M=2048, dt=1e-4)
        refs = {'burgers': evaluate(uh, kk, XG, 2048)}
        _, _, uh, kk = solve('cubic', T, NU, M=2048, dt=1e-4)
        refs['cubic'] = evaluate(uh, kk, XG, 2048)
        t_ref_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        muH = mu_heat(x0, m0, ul, NU, T, XG)
        t_muH = time.perf_counter() - t0
        print(f"  mu_H evaluation cost (one-off): {t_muH*1e3:.1f} ms; "
              f"spectral references built in {t_ref_build:.1f}s (NOT charged to "
              f"any arm: reference is for MSE only)\n")

        rows, arch, coeffs = [], {}, {}
        for flux in FLUXES:
            # ---------- training batch: one scalar c per (flux, policy) ----------
            t_train = {}
            for pol in ('RAW', 'FULL'):
                tt = time.perf_counter()
                Ut, Ht, _, _ = batch(flux, pol, TRAIN_KEY, TRAIN_SEEDS, x0, m0, ul)
                D = Ht - muH
                cov = float(dx * np.sum(((Ut - Ut.mean(0)) * (D - D.mean(0))
                                         ).mean(0) * TRAIN_SEEDS / (TRAIN_SEEDS - 1)))
                var = float(dx * np.sum(D.var(axis=0, ddof=1)))
                coeffs[(flux, pol)] = cov / var
                t_train[pol] = time.perf_counter() - tt
            print(f"{flux}: trained c* = " + ", ".join(
                f"{p} {coeffs[(flux, p)]:.4f}" for p in ('RAW', 'FULL'))
                + f"   (16 seeds, key {TRAIN_KEY}, frozen)")

            # ---------- evaluation batch ----------
            E = {}
            for pol in POLICIES:
                E[pol] = batch(flux, pol, EVAL_KEY, EVAL_SEEDS, x0, m0, ul)
            # plain-solver timing for arms that need no instrumentation
            tp = []
            for s in range(16):
                r = np.random.default_rng(np.random.SeedSequence([EVAL_KEY, s]))
                q = time.perf_counter()
                advance_pair_flux((x0, m0, ul), NU, H_STEP, K_STEPS, r, 'RAW', flux)
                tp.append(time.perf_counter() - q)
            t_plain = float(np.mean(tp))

            fields = {}
            for pol in POLICIES:
                U, Hf, RB, tt = E[pol]
                fields[pol] = U
                fields[f'{pol}+RB'] = RB
                base = tt['solve_instr'] + tt['U']
                cost_plain = t_plain + tt['U']
                rows.append(dict(flux=flux, arm=pol, cost=cost_plain,
                                 note='no control'))
                if pol != 'WITHIN':
                    for tag, c in (('c=1', 1.0), ('c*', coeffs[(flux, pol)])):
                        fields[f'{pol}+CV({tag})'] = U - c * (Hf - muH)
                        extra = t_muH / EVAL_SEEDS
                        if tag == 'c*':
                            extra += t_train[pol] / EVAL_SEEDS
                        rows.append(dict(flux=flux, arm=f'{pol}+CV({tag})',
                                         cost=base + tt['H'] + extra,
                                         note=f'c={c:.4f}; mu_H and training '
                                              f'amortised over {EVAL_SEEDS} replicas'))
                    rows.append(dict(flux=flux, arm=f'{pol}+RB',
                                     cost=t_plain + tt['RB'],
                                     note='Gaussian-CDF at all 801 probes, timed'))
                else:
                    fields.pop(f'{pol}+RB')

            # ---------- metrics, common observable, paired bootstrap ----------
            rng = np.random.default_rng(8003)
            idx = rng.integers(0, EVAL_SEEDS, (BOOT, EVAL_SEEDS))
            V0 = np.array([dx * np.sum(fields['RAW'][i].var(axis=0, ddof=1))
                           for i in idx])
            print(f"\n{flux}  ({EVAL_SEEDS} eval seeds, key {EVAL_KEY}, "
                  f"grid [-8,8] x 801)")
            print(f"  {'arm':>16} {'int var':>11} {'ratio':>7} {'95% CI':>18} "
                  f"{'cost s':>8} {'var*cost':>10} {'vc ratio':>9} {'MSE':>11}")
            base_v = float(dx * np.sum(fields['RAW'].var(axis=0, ddof=1)))
            base_c = [r for r in rows if r['flux'] == flux and r['arm'] == 'RAW'][0]['cost']
            for r in rows:
                if r['flux'] != flux:
                    continue
                F = fields[r['arm']]
                v = float(dx * np.sum(F.var(axis=0, ddof=1)))
                mse = float(np.mean(dx * np.sum((F - refs[flux]) ** 2, axis=1)))
                vb = np.array([dx * np.sum(F[i].var(axis=0, ddof=1)) for i in idx])
                lo, hi = np.percentile(vb / V0, [2.5, 97.5])
                r.update(int_var=v, var_ratio=v / base_v,
                         var_ratio_ci=[float(lo), float(hi)], mse=mse,
                         var_cost=v * r['cost'],
                         var_cost_ratio=(v * r['cost']) / (base_v * base_c))
                print(f"  {r['arm']:>16} {v:11.4e} {r['var_ratio']:7.4f} "
                      f"[{lo:.4f},{hi:.4f}] {r['cost']:8.4f} {r['var_cost']:10.3e} "
                      f"{r['var_cost_ratio']:9.4f} {mse:11.4e}")
            for k, v in fields.items():
                arch[f'{flux}_{k}'] = v.astype(np.float32)
            print()

        # ---------- grid / window convergence on the common observable ----------
        print("grid and window convergence (16 eval seeds, RAW and FULL+CV(c=1))")
        for w, n in ((6.0, 401), (8.0, 401), (8.0, 801), (8.0, 1601), (10.0, 801)):
            g = np.linspace(-w, w, n); d = float(g[1] - g[0])
            U, Hf, _, _ = batch('burgers', 'RAW', EVAL_KEY, 16, x0, m0, ul, grid=g)
            mg = mu_heat(x0, m0, ul, NU, T, g)
            vr = float(d * np.sum(U.var(axis=0, ddof=1)))
            vc = float(d * np.sum((U - (Hf - mg)).var(axis=0, ddof=1)))
            print(f"   window [-{w:g},{w:g}] n={n:5d}: RAW {vr:.5e}   "
                  f"RAW+CV(c=1) {vc:.5e}   ratio {vc/vr:.4f}")
        caught = [f'{w.category.__name__}: {w.message}' for w in wlog]

    arch['x_grid'] = XG
    arch['mu_H'] = muH
    arch['init_x'] = x0
    arch['init_m'] = m0
    for f in FLUXES:
        arch[f'reference_{f}'] = refs[f]
    np.savez_compressed(OUT / 'fields.npz', **arch)
    with (OUT / 'rows.csv').open('w') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:16]
    (OUT / 'controls.json').write_text(json.dumps(dict(
        predeclared=dict(problem='two-pulse', nu=NU, T=T, N=N_PART, h=H_STEP,
                         fluxes=list(FLUXES), policies=list(POLICIES),
                         train=dict(key=TRAIN_KEY, seeds=TRAIN_SEEDS),
                         evaluate=dict(key=EVAL_KEY, seeds=EVAL_SEEDS),
                         grid='[-8,8], 801 points, common to every arm'),
        control='U - c(H - mu_H); closest particle realisation is '
                'Al-Mohssen & Hadjiconstantinou 2010 Eq. (2.1). NOT LVDSMC.',
        c_star={f'{f}|{p}': coeffs[(f, p)] for (f, p) in coeffs},
        c_star_formula='int Cov(U,H) dx / int Var(H) dx, one scalar per '
                       '(flux, policy), fitted on the training batch only',
        mu_H='exact mean of the heat surrogate for the DISCRETE initial '
             'particles; no continuum profile and no spectral solution enters',
        cost_note=f'mu_H and training amortised over {EVAL_SEEDS} replicas; '
                  f'spectral reference cost is NOT charged to any arm',
        bootstrap=dict(replicates=BOOT, kind='paired percentile bootstrap over '
                       'evaluation seeds; an inference method with assumptions, '
                       'not certified coverage'),
        rows=rows, warnings_recorded=caught,
        hashes=dict(estimators=sha('studies/round08_estimators.py'),
                    driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                    solver=sha('relaxation_gbmc.py'))), indent=1))
    print(f"\nwarnings recorded: {len(caught)}")
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
