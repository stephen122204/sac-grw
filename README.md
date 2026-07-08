# RB-GBMC — Heat, Burgers, FitzHugh–Nagumo

Companion code for *Gradient Random Walk Methods for Diffusive PDEs and a
Relaxation–Brownian Particle Scheme for Viscous Burgers' Equation*
(Abkin & Daripa). Paper link: to be added.

Gradient random walk (GRW) particle solvers for the heat and scalar
FitzHugh–Nagumo equations, and a relaxation–Brownian gradient Monte Carlo
(RB-GBMC) particle scheme for viscous Burgers' equation, together with the
studies behind every table and figure in the paper.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # numpy, matplotlib; scipy for full reruns

# Check every checked-in paper number without rerunning anything (seconds):
python reproduce.py verify

# Rerun a study with the paper configuration (needs scipy; ~10-20 min each):
python reproduce.py t6            # production GBMC N-refinement
python reproduce.py --help        # all targets

# Single exploratory simulations from JSON configs:
python main.py configs/burgers_stationary_shock.json
```

Note: function defaults inside the study scripts are exploration settings.
The paper configurations live in the `reproduce.py` entry points — in
particular, the T2 traveling-shock paper run uses S=30 seeds
(`python reproduce.py t2`), not the S=10 function default. All ensemble
studies use base seed 42 with the same seed list paired across N.

## What was used in the paper

| Paper artifact | Command | Data |
|---|---|---|
| Table 1 (heat bias/spread/total) | `python reproduce.py t4` | `output/final_prepublication_tests/heat_extended/` |
| Table 2 + FHN Δt study | `python reproduce.py t5` | `output/final_prepublication_tests/fhn_extended/` |
| Tables 5–6 + production figures | `python reproduce.py t6` | `output/final_prepublication_tests/gbmc_production_n_refinement/` and `regen_data/` |
| Table 7 (Δt bias) | `python reproduce.py t1` | `output/final_prepublication_tests/gbmc_dt_bias/` |
| Table 8 + traveling-shock figures | `python reproduce.py t2` | `output/final_prepublication_tests/gbmc_traveling_shock/` |
| Cole–Hopf plateau figure + diagnostics | `python reproduce.py t3` | `output/final_prepublication_tests/cole_hopf_plateau/` |

`regen_data/` holds the corrected tanh-fit regeneration (same 50 seeds,
error columns bit-identical to the checked-in study output; only the fitted
viscosity/center columns differ) that backs Table 6, the fitted-viscosity
figure, and the two `_fixed` bias–spread–total figures used in the draft.
`expected_values.json` pins the values `reproduce.py verify` checks.
