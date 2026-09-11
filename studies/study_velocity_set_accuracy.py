"""Full-field accuracy of the velocity-set choice in gradient relaxation particles.

On {-a,+a} the mean constraint E[V|u]=f'(u) has a UNIQUE solution, so the
displacement variance a^2-f'(u)^2 is forced.  On {-a,0,+a} the constraint
leaves a free parameter and the variance can be minimised:

    move at a*sign(s) with probability |s|/a, otherwise stay,
    variance  a|s| - s^2  <  a^2 - s^2   for |s| < a.

This driver measures whether that reduction shows up in the computed solution,
against the mean-transport reference and the variance-compensated variants.
"""
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import (advance_rbgbmc_particles,
                             initialize_tanh_shock_particles,
                             reconstruct_cumulative_field)
from studies.study_smooth_transient import initialize_gaussian_gradient_particles

OUT = ROOT / 'output/velocity_set_2026_09_09'
N, S = 6400, 50
ARMS = {
    'mean':            dict(conditional_mean_transport=True),
    'two_speed':       dict(),
    'two_speed_comp':  dict(compensate_transport_variance=True),
    'three_speed':     dict(min_variance_three_speed=True),
    'three_speed_comp': dict(min_variance_three_speed=True,
                             compensate_transport_variance=True),
}
CASES = ([dict(name=f'shock_a2_h{h:g}', problem='shock', a=2., nu=.5, dt=h*.5, T=2.5)
          for h in (.02, .005)] +
         [dict(name=f'gaussian_a{a:g}_dt0.005', problem='gaussian', a=a, nu=.1,
               dt=.005, T=1.) for a in (2., 4.)])


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    names = list(ARMS)
    for ci, cfg in enumerate(CASES):
        dest = OUT / cfg['name']; dest.mkdir(exist_ok=True)
        if (dest / 'summary.json').exists():
            continue
        if cfg['problem'] == 'shock':
            x = np.linspace(-10., 14., 400)
            ref = -np.tanh((x - 2.) / (2 * cfg['nu']))
            initial = initialize_tanh_shock_particles(N, cfg['nu'], 1., 2.)
        else:
            z = np.load(ROOT / 'output/final_prepublication_tests/'
                               'gbmc_smooth_transient/reference.npz')
            x, ref = z['x'], z['u_ref']
            initial = initialize_gaussian_gradient_particles(N)
        dx = float(x[1] - x[0]); steps = round(cfg['T'] / cfg['dt'])
        profiles = {a: [] for a in names}; records = []
        for s in range(S):
            rot = names[s % len(names):] + names[:s % len(names)]
            for arm in rot:
                rl = np.random.default_rng(np.random.SeedSequence([20260909, 77, ci, s, 1]))
                rb = np.random.default_rng(np.random.SeedSequence([20260909, 77, ci, s, 2]))
                t0 = time.perf_counter()
                run = advance_rbgbmc_particles(*initial, cfg['nu'], cfg['a'], cfg['dt'],
                                               steps, rl, rng_brownian=rb, **ARMS[arm])
                el = time.perf_counter() - t0
                u = reconstruct_cumulative_field(run['x'], run['m'], initial[2], x)
                profiles[arm].append(u)
                records.append(dict(seed=s, arm=arm, seconds=el,
                                    error_l2=float(np.sqrt(dx * np.sum((u - ref) ** 2)))))
        rows = []
        for arm in names:
            v = np.asarray(profiles[arm]); avg = v.mean(axis=0)
            bias = float(np.sqrt(dx * np.sum((avg - ref) ** 2)))
            spread = float(np.sqrt(dx * np.mean(np.sum((v - avg) ** 2, axis=1))))
            errs = np.sqrt(dx * np.sum((v - ref) ** 2, axis=1))
            rows.append(dict(arm=arm, bias=bias, spread=spread,
                             total=float(np.sqrt(np.mean(errs ** 2))),
                             seconds=float(np.median([r['seconds'] for r in records
                                                      if r['arm'] == arm]))))
        rng = np.random.default_rng(4242); idx = rng.integers(0, S, (5000, S))
        e2 = {a: dx * np.sum((np.asarray(profiles[a]) - ref) ** 2, axis=1) for a in names}
        ratios = {}
        for arm in names:
            for base in ('two_speed', 'mean'):
                if arm == base:
                    continue
                b = np.sqrt(e2[arm][idx].mean(axis=1) / e2[base][idx].mean(axis=1))
                ratios[f'{arm}_over_{base}'] = dict(
                    ratio=float(np.sqrt(e2[arm].mean() / e2[base].mean())),
                    interval=np.percentile(b, [2.5, 97.5]).tolist())
        np.savez_compressed(dest / 'profiles.npz', x=x, reference=ref, **profiles)
        with (dest / 'per_run.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
        (dest / 'summary.json').write_text(json.dumps(
            dict(config=cfg, N=N, S=S, rows=rows, ratios=ratios), indent=2))
        print(f"== {cfg['name']}")
        for r in rows:
            print(f"   {r['arm']:>17} bias={r['bias']:.5f} spread={r['spread']:.5f} "
                  f"total={r['total']:.5f}  t={r['seconds']:.3f}s")
        for k in ('three_speed_over_two_speed', 'three_speed_over_mean',
                  'two_speed_comp_over_two_speed', 'three_speed_comp_over_mean'):
            if k in ratios:
                r = ratios[k]
                print(f"   {k:>30} = {r['ratio']:.3f} "
                      f"[{r['interval'][0]:.3f}, {r['interval'][1]:.3f}]")
        sys.stdout.flush()


if __name__ == '__main__':
    main()
