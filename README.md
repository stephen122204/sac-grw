# RB-GBMC: Relaxation–Brownian Gradient Monte Carlo for Viscous Burgers

This repository contains computer code for reproducing the numerical results
described in the manuscript *Gradient Random Walk Methods for Diffusive PDEs
and a Relaxation–Brownian Particle Scheme for Viscous Burgers' Equation* by
Stephen Abkin and Prabir Daripa.

**Paper:** link to be added (arXiv preprint forthcoming).

## Getting Started

```bash
git clone https://github.com/stephen122204/heat_burgers_fhn.git
cd heat_burgers_fhn
git checkout rb-gbmc-paper2
```

## Reproducing Numerical Results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Check every number reported in the paper against the checked-in study
outputs (runs in seconds, no simulation):

```bash
python reproduce.py verify
```

Rerun any study at its exact paper configuration (requires scipy;
about 10–20 minutes each):

```bash
python reproduce.py t4    # heat convergence          (Table 1)
python reproduce.py t5    # FitzHugh–Nagumo           (Table 2)
python reproduce.py t3    # Cole–Hopf diagnostics     (Section 6)
python reproduce.py t6    # production RB-GBMC sweep  (Tables 5–6)
python reproduce.py t1    # time-step bias            (Table 7)
python reproduce.py t2    # traveling shock, S=30     (Table 8)
```

All ensemble studies use base seed 42 with the same seed list paired across
particle counts. Function defaults inside the study scripts are exploration
settings; the paper configurations live in the `reproduce.py` entry points.
Study outputs are under `output/final_prepublication_tests/`, and
`regen_data/` holds the corrected tanh-fit data behind Table 6 and the
fitted-viscosity figure.

## Running Your Own Experiments

The section above is for convenience in reproducing the paper. To run the
solvers with your own inputs, copy a JSON config from `configs/`, edit it,
and run:

```bash
python main.py configs/burgers_relaxation_gbmc.json   # RB-GBMC solver
python main.py configs/heat_step_dirichlet.json       # heat GRW
python main.py configs/fhn_grw_steady.json            # scalar FHN GRW
```

The main fields are `diff_constant` (α, D, or ν), `time_step`, `total_time`,
`num_points` (particle count), `domain_size`, and `boundary_conditions`
(Dirichlet reflects particles and keeps their mass; Neumann reflects and
negates it). Each equation adds an initial-condition block: heat takes a
step, uniform-gradient, or Gaussian-cloud profile; FHN takes the logistic
front (`steady_solution`), a linear ramp, or a Heaviside step; Burgers takes
a stationary shock, traveling wave, or step, with `burgers_mode` selecting
`cole_hopf_grw` or `relaxation_gbmc`. The RB-GBMC solver additionally needs
`relaxation_speed_a` (must exceed the shock amplitude), runs on the whole
line, and currently supports the stationary-shock initialization only.
`config_template.jsonc` documents every field with comments; comparison
figures are saved under `output/`. To check a custom run against an exact
solution where one exists, use `python verify_solver.py --equation burgers
--config <your_config>.json`.

## Citation

See `CITATION.cff`.
