# SAC-GRW

Code for **Sign-Aware Antithetic Coupling for Signed Gradient-Particle Methods**,
by Stephen Abkin and Prabir Daripa.

The method pairs two gradient-particle simulations by spatial rank and uses
mass signs to choose reflected or synchronous diffusion increments. This
repository supports two uses: reproducing the paper's numerical summaries and
figures from archived experiments, and running the coupled solver with other
parameters.

## Install

Use Python 3.12:

```bash
git clone https://github.com/stephen122204/sac-grw.git
cd sac-grw
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-coupling.txt -r requirements-test.txt
python -m pytest -q
```

## Reproduce the paper

```bash
python reproduce.py paper
```

This rebuilds the reported numerical summaries in
`reproduction/analysis/numbers.json`, checks them against archived particle fields
and held-out errors, and creates the five figures in `reproduction/figures/`.
No manuscript files or TeX installation are needed.

To rebuild and check the numbers without generating figures:

```bash
python reproduce.py verify-paper
```

These commands analyze the archived experiments under `output/`; they do not
repeat the simulations or measure new timings. Each summary in `numbers.json`
records the archive it was read from, and Appendix F of the paper maps every
table and figure to its archive. The drivers that produced the archives are in
`studies/`, with their settings and output paths in the scripts. Use a separate
clone for full reruns because the drivers overwrite the archived outputs.
Timings depend on the machine.

## Run other parameters

```bash
python run_coupling.py --profile gaussian --flux burgers --policy FULL --particles 400 --pairs 8 --nu 0.1 --dt 0.005 --time 1 --seed 42 --output output/custom/burgers
python run_coupling.py --profile gaussian --flux cubic --policy WITHIN --particles 400 --pairs 8 --nu 0.1 --dt 0.005 --time 1 --seed 42 --output output/custom/cubic
python run_coupling.py --help
```

Outputs are the individual paired fields, their mean and standard errors in
`fields.npz`, and parameters, estimated sampling variance, and runtime in
`summary.json`. This example uses its own deterministic initial step field and
is not a rerun of a paper benchmark. It does not estimate discretization bias
or certify an accuracy target.

`FULL` switches by mass sign; `RAW` reflects by spatial rank; `WITHIN` reflects
within each sign class; `FINAL` switches only at the last stage. The `heat`,
`burgers`, and `cubic` fluxes and the `gaussian` and `shock` initial profiles are
supported. Use an even particle count and a final time divisible by the time
step. For other initial particles or flux derivatives, use `advance_pair` in
`coupled_gradient_particles.py`.

## Code and data

- `coupled_gradient_particles.py`: coupled mean-transport stepper.
- `relaxation_gbmc.py`: particle initialization, reconstruction, and base solver.
- `config.py`: simulation configuration used by the drivers.
- `reproduction/analysis/`: scripts that rebuild the paper's numbers and figures,
  with `numbers.json` and the Figure 1 state.
- `reproduction/evidence/`: replayed particle fields behind Table 1 and Figure 2.
- `output/`: the archived experiments the paper reports, one directory per study.
- `studies/`: the drivers that produced those archives and the modules they import.
- `tests/`: solver and coupling checks.

The repository contains only the code and data used by the paper.

Citation details are in [CITATION.cff](CITATION.cff). No DOI has been assigned.

## Acknowledgments

**Principal Investigator:** [Professor Prabir Daripa](https://artsci.tamu.edu/mathematics/contact/profiles/prabir-daripa.html) — Texas A&M University, Department of Mathematics.

Other projects from the Daripa Research Group are available on the
[group's GitHub page](https://github.com/Daripa-Research-Group).
