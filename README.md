# RB-GBMC: Relaxation–Brownian Gradient Monte Carlo for Viscous Burgers

This repository contains computer code for reproducing the numerical results
described in the paper *A Relaxation–Brownian Gradient Particle Method for
Viscous Burgers' Equation, with Multi-Seed Convergence Studies of Gradient
Random Walk Methods* by Stephen Abkin and Prabir Daripa.

**Paper:** link to be added (arXiv preprint forthcoming).

## Getting Started

```bash
git clone https://github.com/stephen122204/heat_burgers_fhn.git
cd heat_burgers_fhn
git checkout rb-gbmc-paper2
```

## Reproducing Numerical Results

```bash
python -m venv .venv && source .venv/bin/activate   # tested with Python 3.11
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
python reproduce.py t6    # production RB-GBMC sweep  (Tables 3–4)
python reproduce.py t1    # time-step bias            (Table 5)
python reproduce.py t2    # traveling shock, S=30     (Table 6)
```

Regenerate the paper figures from the checked-in study data (seconds, no rerun):

```bash
python reproduce.py figures   # -> output/final_prepublication_tests/paper_figures/
```

All ensemble studies use base seed 42; the same seed identifiers are reused at
each particle count for reproducibility, which is not a strict
common-random-number coupling across N. Function defaults inside the study scripts are exploration
settings; the paper configurations live in the `reproduce.py` entry points.
Study outputs are under `output/final_prepublication_tests/`; see its
`PROVENANCE.md` for the output layout, tolerance policy, and environment.

## Running Your Own Experiments

The section above is for convenience in reproducing the paper. To run the
solvers with your own inputs, copy a JSON config from `configs/`, edit it,
and run:

```bash
python main.py configs/burgers_relaxation_gbmc.json   # RB-GBMC solver
python main.py configs/heat_step_dirichlet.json       # heat GRW
python main.py configs/fhn_grw_steady.json            # scalar FHN GRW
```

Every config field is documented with comments in `config_template.jsonc`.
Comparison figures are saved under `output/`, and a custom run can be
checked against an exact solution where one exists via
`python verify_solver.py --equation burgers --config <your_config>.json`.

## Citation

See `CITATION.cff`.
