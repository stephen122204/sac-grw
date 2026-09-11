"""Compare full-field accuracy of original, compensated, and mean transport.

Uses the carried-velocity schedule in the paper. Brownian normals are paired
by sorted rank; compensation changes their amplitude. Times exclude fitting
and output. Each cell saves profiles and realization errors before proceeding.
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

OUT = ROOT / 'output/compensated_accuracy_2026_09_09'
N, S = 6400, 50
ARMS = ('original', 'compensated', 'mean')
CASES = ([dict(name=f'shock_a2_h{h:g}', problem='shock', a=2., nu=.5,
               dt=h*.5, T=2.5) for h in (.02,.01,.005)] +
         [dict(name=f'gaussian_a{a:g}_dt{dt:g}', problem='gaussian', a=a,
               nu=.1, dt=dt, T=1.) for a in (2.,4.) for dt in (.005,.00125)])


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    design = dict(N=N, S=S, cases=CASES, primary='RMS L2 error against exact reference',
                  secondary=['ensemble-mean error', 'spread', 'runtime'],
                  pairing='normals by sorted rank within case and seed',
                  schedule='transport, reconstruct, sample next velocity, diffuse')
    design_path = OUT / 'design.json'
    if design_path.exists():
        assert json.loads(design_path.read_text()) == design
    else:
        design_path.write_text(json.dumps(design, indent=2))
    for ci, cfg in enumerate(CASES):
        dest=OUT / cfg['name']
        dest.mkdir(exist_ok=True)
        if (dest/'summary.json').exists():
            continue
        if cfg['problem']=='shock':
            x=np.linspace(-10.,14.,400)
            ref=-np.tanh((x-2.)/(2*cfg['nu']))
            initial=initialize_tanh_shock_particles(N,cfg['nu'],1.,2.)
        else:
            z=np.load(ROOT/'output/final_prepublication_tests/gbmc_smooth_transient/reference.npz')
            x,ref=z['x'],z['u_ref']
            initial=initialize_gaussian_gradient_particles(N)
        dx=float(x[1]-x[0]); steps=round(cfg['T']/cfg['dt'])
        profiles={a:[] for a in ARMS}; records=[]
        # Rotate arm order to reduce a fixed ordering effect on measured time.
        for s in range(S):
            for arm in ARMS[s%3:]+ARMS[:s%3]:
                rl=np.random.default_rng(np.random.SeedSequence([20260909,ci,s,1]))
                rb=np.random.default_rng(np.random.SeedSequence([20260909,ci,s,2]))
                start=time.perf_counter()
                run=advance_rbgbmc_particles(*initial,cfg['nu'],cfg['a'],cfg['dt'],
                    steps,rl,rng_brownian=rb,conditional_mean_transport=arm=='mean',
                    compensate_transport_variance=arm=='compensated')
                elapsed=time.perf_counter()-start
                u=reconstruct_cumulative_field(run['x'],run['m'],initial[2],x)
                profiles[arm].append(u)
                records.append(dict(seed=s,arm=arm,seconds=elapsed,
                                    error_l2=float(np.sqrt(dx*np.sum((u-ref)**2)))))
        rows=[]
        for arm in ARMS:
            v=np.asarray(profiles[arm]); avg=v.mean(axis=0)
            bias=float(np.sqrt(dx*np.sum((avg-ref)**2)))
            spread=float(np.sqrt(dx*np.mean(np.sum((v-avg)**2,axis=1))))
            errors=np.sqrt(dx*np.sum((v-ref)**2,axis=1))
            total=float(np.sqrt(np.mean(errors**2)))
            assert abs(total**2-bias**2-spread**2)<1e-12
            rows.append(dict(arm=arm,bias=bias,spread=spread,total=total,
                             seconds=float(np.median([r['seconds'] for r in records if r['arm']==arm]))))
        # Seed-paired uncertainty for the relative RMS improvement.
        rng=np.random.default_rng(12345)
        idx=rng.integers(0,S,(5000,S))
        e2={a:dx*np.sum((np.asarray(profiles[a])-ref)**2,axis=1) for a in ARMS}
        ratios={}
        for base in ('original','mean'):
            b=np.sqrt(e2['compensated'][idx].mean(axis=1)/e2[base][idx].mean(axis=1))
            ratios[base]=dict(ratio=float(np.sqrt(e2['compensated'].mean()/e2[base].mean())),
                              interval=np.percentile(b,[2.5,97.5]).tolist())
        np.savez_compressed(dest/'profiles.npz',x=x,reference=ref,**profiles)
        with (dest/'per_run.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
        result=dict(config=cfg,N=N,S=S,rows=rows,compensated_ratios=ratios)
        (dest/'summary.json').write_text(json.dumps(result,indent=2))
        print(cfg['name'],json.dumps(result),flush=True)


if __name__=='__main__':
    main()
