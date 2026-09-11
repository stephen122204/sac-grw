"""Check numerical resolution of two response-model predictions."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.effective_equation import solve_modified,cole_hopf_constant_nu
from analysis.predict import fit_ratio,d_vel_over_nu
OUT=ROOT/'output/compensated_accuracy_2026_09_09/response_resolution.json'
results=json.loads(OUT.read_text()) if OUT.exists() else []
for a,h,tau in [(2.,.005,2.),(4.,.02,1.)]:
    w=np.linspace(-24.,24.,400);D=d_vel_over_nu(a,h)
    for dx in [.02,.01,.005]:
        if any(r.get("alpha")==a and r.get("h")==h and r.get("tau")==tau and r.get("dx")==dx for r in results):
            continue
        x,u=solve_modified(tau,h,a,Xmax=144.,dX=dx)
        g=(fit_ratio(w,np.interp(w,x,u))[2]-1)/D
        results.append(dict(alpha=a,h=h,tau=tau,dx=dx,g=g))
        OUT.write_text(json.dumps(results,indent=2));print(results[-1],flush=True)
# Independent constant-viscosity field check against Cole--Hopf.
a,h,tau=2.,.005,2.;D=d_vel_over_nu(a,h)
x,u=solve_modified(tau,h,a,Xmax=144.,dX=.01,delta_fn=lambda u:np.full_like(u,D))
w=np.linspace(-24.,24.,400)
err=np.max(np.abs(np.interp(w,x,u)-cole_hopf_constant_nu(w,tau,1+D)))
results=[r for r in results if r.get("check")!="constant_coefficient_Cole_Hopf_max_error"]
results.append(dict(check='constant_coefficient_Cole_Hopf_max_error',value=float(err)))
OUT.write_text(json.dumps(results,indent=2));print(results[-1],flush=True)
