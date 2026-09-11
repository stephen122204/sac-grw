"""Round-9: corrected estimator-specific costs and one transport-aware control.

TWO ROUND-8 COST DEFECTS, BOTH CONFIRMED BY INSPECTION AND CORRECTED HERE.

 (C1) study_round08_controls.py measured the plain solver time with policy
      'RAW' only (its line 155) and assigned that number to FULL, WITHIN and
      every RB arm (lines 166, 180). FULL evaluates sign products and WITHIN
      runs a per-sign-class matching loop, so their plain costs are not RAW's.
      Corrected: every policy is timed separately, in interleaved repetitions.

 (C2) The coefficient-training batch called batch(), which computes the
      Gaussian-CDF conditional-expectation field for every training sample
      (line 77) even though fitting uses only U and the surrogate. That work
      was charged to the trained-control arms. Corrected: training runs only
      the work fitting needs, and is timed as such.

Costs are composed from separately measured components; one-off costs (analytic
means, coefficient training) are amortised over an explicitly named reuse count
and are also reported un-amortised. Per-replicate times are archived.

NEW COMPARATOR: the frozen-initial-velocity control of studies/round09_estimators.py.
It is a standard control variate with an analytically known discrete mean, added
to test whether the heat control's weakness came from omitting leading transport.
It is not a complete nonlinear surrogate and stands for no other control.
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
from studies.round09_estimators import (initial_velocity_by_identity,
                                        frozen_pair_field, mu_frozen)
from studies.study_round06_cubic import advance_pair_flux
from studies.round06_spectral_reference import solve, evaluate
from studies import twopulse_reference as TP

OUT = ROOT / 'output/round09_costs_control_2026_09_10'
R8 = ROOT / 'output/round08_controls_2026_09_10/fields.npz'
NU, T, N_PART, H_STEP = 0.1, 1.0, 1600, 0.005
K_STEPS = round(T / H_STEP)
FLUXES = ('burgers', 'cubic')
POLICIES = ('RAW', 'FULL', 'WITHIN')
TRAIN_KEY, EVAL_KEY, TRAIN_SEEDS, EVAL_SEEDS = 8001, 8002, 16, 64
XG = np.linspace(-8.0, 8.0, 801)
BOOT, TIMING_REPS = 4000, 24


def seed(key, s):
    return np.random.default_rng(np.random.SeedSequence([key, s]))


def eval_batch(flux, pol, x0, m0, ul, v0, key=EVAL_KEY, seeds=EVAL_SEEDS, grid=XG):
    U, H, F, RB = [], [], [], []
    for s in range(seeds):
        o = advance_pair_instrumented((x0, m0, ul), NU, H_STEP, K_STEPS,
                                      seed(key, s), pol, flux)
        U.append(pair_field(o, grid))
        H.append(heat_pair_field(o, x0, m0, grid))
        F.append(frozen_pair_field(o, x0, m0, v0, T, grid))
        RB.append(rb_pair_field(o, NU, H_STEP, grid))
    return tuple(np.array(a) for a in (U, H, F, RB))


def train_c(flux, pol, kind, x0, m0, ul, v0, mu):
    """Fit ONE scalar on the disjoint training batch, doing only the work the
    fit needs (no conditional-expectation fields). Returns (c, wall time)."""
    t0 = time.perf_counter()
    U, D = [], []
    for s in range(TRAIN_SEEDS):
        o = advance_pair_instrumented((x0, m0, ul), NU, H_STEP, K_STEPS,
                                      seed(TRAIN_KEY, s), pol, flux)
        U.append(pair_field(o, XG))
        D.append((heat_pair_field(o, x0, m0, XG) if kind == 'H'
                  else frozen_pair_field(o, x0, m0, v0, T, XG)) - mu)
    U, D = np.array(U), np.array(D)
    dx = float(XG[1] - XG[0])
    cov = float(dx * np.sum(((U - U.mean(0)) * (D - D.mean(0))).mean(0)
                            * TRAIN_SEEDS / (TRAIN_SEEDS - 1)))
    var = float(dx * np.sum(D.var(axis=0, ddof=1)))
    return cov / var, time.perf_counter() - t0, U, D


def timings(flux, x0, m0, ul, v0, grid=XG, reps=TIMING_REPS):
    """Estimator-specific component times. Every policy timed separately;
    repetition order shuffled so no component sits systematically first."""
    rng = np.random.default_rng(9001)
    acc = {k: [] for k in
           [f'{p}_{w}' for p in POLICIES for w in ('plain', 'prefinal', 'instr')]
           + ['U', 'H', 'F', 'RB']}
    for r in range(reps):
        for p in rng.permutation(list(POLICIES)):
            t0 = time.perf_counter()
            advance_pair_flux((x0, m0, ul), NU, H_STEP, K_STEPS, seed(EVAL_KEY, r), p, flux)
            acc[f'{p}_plain'].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            advance_pair_instrumented((x0, m0, ul), NU, H_STEP, K_STEPS,
                                      seed(EVAL_KEY, r), p, flux, track=False)
            acc[f'{p}_prefinal'].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            o = advance_pair_instrumented((x0, m0, ul), NU, H_STEP, K_STEPS,
                                          seed(EVAL_KEY, r), p, flux)
            acc[f'{p}_instr'].append(time.perf_counter() - t0)
            for w in rng.permutation(['U', 'H', 'F', 'RB']):
                t0 = time.perf_counter()
                if w == 'U':
                    pair_field(o, grid)
                elif w == 'H':
                    heat_pair_field(o, x0, m0, grid)
                elif w == 'F':
                    frozen_pair_field(o, x0, m0, v0, T, grid)
                else:
                    rb_pair_field(o, NU, H_STEP, grid)
                acc[w].append(time.perf_counter() - t0)
    return {k: float(np.median(v)) for k, v in acc.items()}, acc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        x0, m0, ul, _ = TP.initialize(N_PART)
        dx = float(XG[1] - XG[0])
        old = np.load(R8)
        refs = {}
        for f in FLUXES:
            _, _, uh, kk = solve(f, T, NU, M=2048, dt=1e-4)
            refs[f] = evaluate(uh, kk, XG, 2048)
        t0 = time.perf_counter(); muH = mu_heat(x0, m0, ul, NU, T, XG)
        t_muH = time.perf_counter() - t0
        rows, arch, meta = [], {}, {}

        for flux in FLUXES:
            v0 = initial_velocity_by_identity(x0, m0, ul, flux)
            t0 = time.perf_counter(); muF = mu_frozen(x0, m0, v0, ul, NU, T, XG)
            t_muF = time.perf_counter() - t0
            tm, raw_t = timings(flux, x0, m0, ul, v0)
            print(f"\n{flux}: corrected per-policy component times (median of "
                  f"{TIMING_REPS} interleaved reps, ms)")
            print("   " + "  ".join(f"{p}: plain {tm[f'{p}_plain']*1e3:.1f} "
                                    f"prefinal {tm[f'{p}_prefinal']*1e3:.1f} "
                                    f"instr {tm[f'{p}_instr']*1e3:.1f}" for p in POLICIES))
            print(f"   fields (ms): U {tm['U']*1e3:.2f}  H {tm['H']*1e3:.2f}  "
                  f"F {tm['F']*1e3:.2f}  RB {tm['RB']*1e3:.2f}   "
                  f"| one-off mu_H {t_muH*1e3:.1f}  mu_F {t_muF*1e3:.1f}")
            print(f"   round-8 assigned RAW's plain time to every policy; the "
                  f"FULL/RAW plain ratio is {tm['FULL_plain']/tm['RAW_plain']:.3f}, "
                  f"WITHIN/RAW {tm['WITHIN_plain']/tm['RAW_plain']:.3f}")

            coef, t_train = {}, {}
            for pol in ('RAW', 'FULL'):
                for kind, mu in (('H', muH), ('F', muF)):
                    c, tt, _, _ = train_c(flux, pol, kind, x0, m0, ul, v0, mu)
                    coef[(pol, kind)] = c; t_train[(pol, kind)] = tt
            print(f"   trained c*: " + "  ".join(
                f"{p}/{k} {coef[(p,k)]:+.4f}" for p in ('RAW','FULL') for k in ('H','F')))

            fields = {}
            for pol in POLICIES:
                U, H, F, RB = eval_batch(flux, pol, x0, m0, ul, v0)
                fields[pol] = U
                rows.append(dict(flux=flux, arm=pol,
                                 cost=tm[f'{pol}_plain'] + tm['U'], oneoff=0.0))
                if pol != 'WITHIN':
                    fields[f'{pol}+RB'] = RB
                    rows.append(dict(flux=flux, arm=f'{pol}+RB',
                                     cost=tm[f'{pol}_prefinal'] + tm['RB'], oneoff=0.0))
                    for kind, D, mu, tmu in (('H', H, muH, t_muH), ('F', F, muF, t_muF)):
                        for tag, c in (('c=1', 1.0), ('c*', coef[(pol, kind)])):
                            fields[f'{pol}+CV{kind}({tag})'] = U - c * (D - mu)
                            one = tmu + (t_train[(pol, kind)] if tag == 'c*' else 0.0)
                            rows.append(dict(
                                flux=flux, arm=f'{pol}+CV{kind}({tag})',
                                cost=tm[f'{pol}_instr'] + tm['U']
                                     + tm['H' if kind == 'H' else 'F']
                                     + one / EVAL_SEEDS,
                                oneoff=one))
                # replay check against the round-8 archive
                k8 = f'{flux}_{pol}'
                if k8 in old:
                    d = float(np.max(np.abs(U.astype(np.float32) - old[k8])))
                    meta[f'replay_{flux}_{pol}'] = d
            print(f"   replay vs round-8 archive (float32): " + "  ".join(
                f"{p} {meta[f'replay_{flux}_{p}']:.1e}" for p in POLICIES))

            rng = np.random.default_rng(9002)
            idx = rng.integers(0, EVAL_SEEDS, (BOOT, EVAL_SEEDS))
            V0b = np.array([dx * np.sum(fields['RAW'][i].var(axis=0, ddof=1)) for i in idx])
            base_v = float(dx * np.sum(fields['RAW'].var(axis=0, ddof=1)))
            base_c = [r for r in rows if r['flux'] == flux and r['arm'] == 'RAW'][0]['cost']
            print(f"\n  {'arm':>18} {'int var':>11} {'ratio':>7} {'95% CI':>18} "
                  f"{'cost ms':>8} {'vc ratio':>9} {'MSE':>11}")
            for r in rows:
                if r['flux'] != flux:
                    continue
                Fq = fields[r['arm']]
                v = float(dx * np.sum(Fq.var(axis=0, ddof=1)))
                vb = np.array([dx * np.sum(Fq[i].var(axis=0, ddof=1)) for i in idx])
                lo, hi = np.percentile(vb / V0b, [2.5, 97.5])
                r.update(int_var=v, var_ratio=v / base_v,
                         var_ratio_ci=[float(lo), float(hi)],
                         mse=float(np.mean(dx * np.sum((Fq - refs[flux]) ** 2, axis=1))),
                         var_cost_ratio=(v * r['cost']) / (base_v * base_c))
                print(f"  {r['arm']:>18} {v:11.4e} {r['var_ratio']:7.4f} "
                      f"[{lo:.4f},{hi:.4f}] {r['cost']*1e3:8.2f} "
                      f"{r['var_cost_ratio']:9.4f} {r['mse']:11.4e}")
            for k, val in fields.items():
                arch[f'{flux}_{k}'] = val.astype(np.float32)
            arch[f'{flux}_mu_F'] = muF
            for k, val in raw_t.items():
                arch[f'{flux}_time_{k}'] = np.array(val)
            meta[flux] = dict(components_s=tm,
                              c_star={f'{p}/{k}': coef[(p, k)] for (p, k) in coef},
                              train_s={f'{p}/{k}': t_train[(p, k)] for (p, k) in t_train},
                              mu_H_s=t_muH, mu_F_s=t_muF)

        # probe-count sensitivity with the ACTUAL estimators on each grid
        print("\nprobe-count sensitivity, actual estimators (burgers, FULL), 24 seeds")
        v0 = initial_velocity_by_identity(x0, m0, ul, 'burgers')
        for n in (201, 401, 801, 1601):
            g = np.linspace(-8.0, 8.0, n); d = float(g[1] - g[0])
            U, _, _, RB = eval_batch('burgers', 'FULL', x0, m0, ul, v0, seeds=24, grid=g)
            tmn, _ = timings('burgers', x0, m0, ul, v0, grid=g, reps=6)
            cu = tmn['FULL_plain'] + tmn['U']; cr = tmn['FULL_prefinal'] + tmn['RB']
            vu = float(d * np.sum(U.var(axis=0, ddof=1)))
            vr = float(d * np.sum(RB.var(axis=0, ddof=1)))
            print(f"   probes {n:5d}: var ratio RB/FULL {vr/vu:.4f}  cost ratio "
                  f"{cr/cu:.3f}  var*cost ratio {(vr*cr)/(vu*cu):.4f}")
            meta[f'probes_{n}'] = dict(var_ratio=vr / vu, cost_ratio=cr / cu,
                                       var_cost_ratio=(vr * cr) / (vu * cu))
        caught = [f'{w.category.__name__}: {w.message}' for w in wlog]

    arch['x_grid'] = XG; arch['mu_H'] = muH
    np.savez_compressed(OUT / 'fields.npz', **arch)
    with (OUT / 'rows.csv').open('w') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / 'costs_control.json').write_text(json.dumps(dict(
        corrections=['C1 per-policy plain/prefinal/instr times, interleaved',
                     'C2 training does only the work fitting needs (no RB field)'],
        reuse_count=EVAL_SEEDS,
        amortisation='analytic means and coefficient training divided by the '
                     'reuse count; un-amortised one-off cost in the oneoff column',
        timing='median of 24 interleaved repetitions per component',
        new_control='frozen initial velocity, exact discrete mean mu_F',
        meta=meta, rows=rows, warnings_recorded=caught,
        hashes=dict(r8=hashlib.sha256((ROOT/'studies/round08_estimators.py').read_bytes()).hexdigest()[:16],
                    r9=hashlib.sha256((ROOT/'studies/round09_estimators.py').read_bytes()).hexdigest()[:16],
                    driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16])),
        indent=1))
    print(f"\nwarnings recorded: {len(caught)}\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
