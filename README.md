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
fitted-viscosity figure. Single exploratory runs:
`python main.py configs/burgers_stationary_shock.json`.

## Citation

See `CITATION.cff`.
