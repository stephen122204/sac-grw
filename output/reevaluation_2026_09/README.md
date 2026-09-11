# Reevaluation study, 2026-09-08

Self-contained addition. **Nothing under `output/final_prepublication_tests/`
was rerun or modified, and `reproduce.py`, `expected_values.json`, the study
drivers of the published studies, and `relaxation_gbmc.py` are untouched**, so
`python reproduce.py verify` still checks exactly the published values.

New files:

```
analysis/effective_equation.py       modified-equation solver, exact steady
                                     profile, exact Cole-Hopf reference
analysis/predict.py                  parameter-free predictions through the
                                     same tanh fit as the particle profiles
analysis/make_reevaluation_figure.py principal figure and summary table
analysis/particle_count_check.py     particle-count sensitivity of the ratio
studies/study_response_time.py       Studies A, B, C
```

## Run commands

Environment as in the repository README (Python 3.11, `requirements-lock.txt`).
All commands are run from the repository root.

```bash
# 1. Register the parameter-free predictions (seconds).  Rerunning this
#    overwrites registered_predictions.json with identical content.
python - <<'EOF'
import json, sys, platform, subprocess, numpy, scipy
sys.path.insert(0, '.')
from analysis.predict import predict
cases  = [(f'A_a{a:g}', a, 0.005, t) for a in (2.0, 4.0)
          for t in (0.5, 1.0, 2.0, 3.5, 5.0, 10.0, 20.0)]
cases += [('C1', 3.0, 0.005, 3.5), ('C2', 2.0, 0.020, 5.0),
          ('C3', 1.5, 0.010, 2.0), ('C4', 4.0, 0.020, 1.0)]
out = []
for nm, at, dtt, taut in cases:
    p = predict(at, dtt, taut); p['case'] = nm; out.append(p)
print(json.dumps(out, indent=1)[:400])
EOF

# 2. Study A: time development at fixed viscosity          (~50 min, S=400)
python studies/study_response_time.py --mode A --seeds 400 \
    --out output/reevaluation_2026_09/response_time

# 3. Study B: exact-scaling equivalence check              (~1 min,  S=20)
python studies/study_response_time.py --mode B --seeds 20 \
    --out output/reevaluation_2026_09/response_time

# 4. Study C: held-out configurations                      (~8 min,  S=400)
python studies/study_response_time.py --mode C --seeds 400 \
    --out output/reevaluation_2026_09/response_time

# 5. Particle-count sensitivity                            (~25 min, S=120)
python analysis/particle_count_check.py 120

# 6. Principal figure and summary table                    (~1 min)
python analysis/make_reevaluation_figure.py
```

Seeds are value-keyed `SeedSequence` streams
(`SeedSequence([930000, <config key>, seed])` for velocity uniforms,
`SeedSequence([940000, <config key>, seed])` for Brownian normals), so every
cell is reproducible and arms within a cell share the Brownian stream.

## Contents

| file | what it holds |
|---|---|
| `registered_predictions.json` | the predictions, with code version and package versions, written before Studies A and C ran |
| `response_time/A_at2_*`, `A_at4_*` | Study A per-run CSV, realization profiles, paired summary |
| `response_time/C1..C4_*` | Study C, same layout |
| `response_time/B_scaling_check.json` | the exact-equivalence check |
| `particle_count_check.json` | `g` at `N = 1600, 6400, 25600` |
| `transient_prediction.json` | effective-equation prediction for the archived Gaussian hump |
| `reevaluation_summary.json` | every cell, both models, and the aggregate error statistics |
| `figures/reevaluation_response.pdf` | the principal figure |
| `modeA.log`, `modeC.log`, `particle_count.log` | run logs |

## Per-run CSV columns

`A, nu, a, at, dt, dtt, taut, n_steps, arm, seed, A_hat, xc_hat, nu_hat,
fit_rms, at_bound, l2_vs_exact, runtime_s`

Realization-level, one row per (arm, seed, snapshot time). `arm` is
`cond_mean` or `two_speed`. Snapshot times within a seed come from one
trajectory and are correlated; the bootstrap resamples whole seeds.

## If the extension is adopted

`reproduce.py` would gain `response`, `response-heldout`, `response-scaling`
and `particle-count` targets, `expected_values.json` would gain the new pinned
values, and `PROVENANCE.md` would gain an entry. None of that has been done, so
the published reproduction path is unchanged.
