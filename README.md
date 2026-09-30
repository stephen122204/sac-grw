# Sign-Switched Antithetic Coupling for Signed Gradient-Particle Methods

Python software and numerical data for the paper of the same name
by Stephen Abkin and Prabir Daripa. The examples cover the heat equation,
viscous Burgers' equation, and a scalar equation with a cubic flux.

The method pairs two gradient-particle simulations by spatial rank and uses
particle mass signs to choose reflected or synchronous diffusion increments.
The repository supports two uses:

1. reproduce the reported numerical summaries and six figures from the saved experiments, and
2. run the coupled solver with different parameters.

## Install

Clone the repository:

```bash
git clone https://github.com/stephen122204/sac-grw.git
cd sac-grw
```

Create a Python 3.12 environment:

```bash
python3.12 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell, use:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Windows Command Prompt, use:

```bat
.venv\Scripts\activate.bat
```

Then install the pinned dependencies:

```bash
python -m pip install -r requirements-coupling.txt
```

The reproduction environment uses Python 3.12.14.

## Reproduce the Paper

Rebuild the numerical summaries and all six figures from the committed data:

```bash
python reproduce.py paper
```

The summaries are saved in `reproduction/analysis/numbers.json`, and the
figures in `reproduction/figures/`. Each summary identifies its source data.
This command recalculates the statistics and plots from the saved experiments;
new simulations and timings are run with the commands below.

Check the coupled solver and independently recompute the principal variance,
error, and target-attainment results:

```bash
python reproduce.py verify
```

## Run a Modified Case

Run the sign-switched method on a Gaussian profile with Burgers transport:

```bash
python run_coupling.py --profile gaussian --flux burgers --policy FULL --particles 400 --pairs 8 --nu 0.1 --dt 0.005 --time 1 --seed 42 --output output/custom/burgers
```

Change the flux or pairing method to compare another case:

```bash
python run_coupling.py --profile gaussian --flux cubic --policy WITHIN --particles 400 --pairs 8 --nu 0.1 --dt 0.005 --time 1 --seed 42 --output output/custom/cubic
python run_coupling.py --help
```

The available pairing methods are:

| Option | Method |
| --- | --- |
| `FULL` | Spatial-rank matching with the mass-sign switch |
| `RAW` | Spatial-rank matching with reflected increments |
| `WITHIN` | Matching within each mass sign with reflected increments |
| `FINAL` | Spatial-rank reflection with switching only at the final stage |

Use `--flux heat`, `burgers`, or `cubic`, and `--profile gaussian` or `shock`.
The particle count must be even, at least two pairs are needed for a variance
estimate, and the final time must be a multiple of the time step.

Each run saves paired fields, their mean, and standard errors in `fields.npz`.
Its `summary.json` records the parameters, sampling variance, and execution
time. These examples use a deterministic step approximation of the initial
profile and report sampling error. Evaluating total error also requires a
reference solution and a discretization-bias assessment.

Generated files are saved below `output/custom/`. Reusing an output path
replaces that run's files. Execution times depend on hardware.
For other initial particles or flux derivatives, call `advance_pair` in
`coupled_gradient_particles.py`.

## Repository Layout

- `coupled_gradient_particles.py`: coupled particle update.
- `gradient_particles.py`: single-simulation update, initialization, and field reconstruction.
- `run_coupling.py`: run a case with your own parameters.
- `reproduce.py`, `checks/`: reproduce and check the numerical results.
- `reproduction/analysis/`: summary and figure calculations.
- `reproduction/evidence/`: paired particle fields used in the principal comparisons.
- `output/`: saved paper experiments, grouped by descriptive study names.
- `studies/`: original experiment drivers and their shared helpers. Some depend on intermediate research files; use `reproduce.py` for reproduction from this archive and `run_coupling.py` for new cases.

The archived execution times come from the original experiments. Their
recorded environments and timing procedures are retained with the data.
The pinned dependencies above specify the environment for rebuilding the
summaries and figures.

## Citation

The first release is [SAC-GRW-v1.0](https://github.com/stephen122204/sac-grw/releases/tag/v1.0).
The archived software and data are available on
[Zenodo](https://doi.org/10.5281/zenodo.23055907).

Abkin, S., and Daripa, P. (2026). *Sign-Switched Antithetic Coupling for Signed
Gradient-Particle Methods* (Version 1.0) [Computer software]. Zenodo.
[https://doi.org/10.5281/zenodo.23055907](https://doi.org/10.5281/zenodo.23055907).

Software citation metadata are in [CITATION.cff](CITATION.cff).
The software and data are distributed under the [MIT license](LICENSE).

## Acknowledgments

**Principal Investigator:** [Professor Prabir Daripa](https://artsci.tamu.edu/mathematics/contact/profiles/prabir-daripa.html) — Texas A&M University, Department of Mathematics.

Other projects from the Daripa Research Group are available on the
[group's GitHub page](https://github.com/Daripa-Research-Group).
