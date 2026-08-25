# RB-GBMC: Relaxation-Brownian Gradient Monte Carlo for Viscous Burgers' Equation

Python software and reproducible numerical examples for the paper
*A Relaxation-Brownian Gradient Monte Carlo Algorithm for Viscous Burgers'
Equation* by Stephen Abkin and Prabir Daripa.

The repository supports two uses:

1. reproduce the reported tables and figures, and
2. rerun the study scripts with modified parameters for new cases.

## Install

Release archive and repository links will be added at public release.
From this directory, create a Python 3.11 environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` recreates the archived environment (Python 3.11.4,
numpy 1.26.4, scipy 1.17.1, matplotlib 3.10.0). `requirements.txt` allows
compatible version ranges instead. On Windows PowerShell, activate with
`.\.venv\Scripts\Activate.ps1`.

## Check the Installation

Confirm the environment works before the longer runs (takes seconds):

```bash
pytest -q
```

## Reproduce the Paper

Check the 164 pinned numerical values against the archived study outputs,
then rebuild the nine manuscript figures from the committed data. Neither
command reruns a simulation, and both complete in seconds:

```bash
python reproduce.py verify
python reproduce.py figures
```

Rerun any study from scratch with `python reproduce.py <target>`. A rerun
overwrites that study's subdirectory under
`output/final_prepublication_tests/` (the archived paper-output directory).
Approximate wall-clock times on a laptop:

| Target | Study | Time |
|---|---|---|
| `t6` | stationary particle refinement | 1--2 min |
| `multinu` | fixed-step multi-viscosity sweep | 1--2 min |
| `pilot` | transport-velocity timing pilot | 2--4 min |
| `transient` | smooth transient (computes its reference on first run) | 3--5 min |
| `multinu-scaled` | viscosity-scaled time-step sweep | ~10 min |
| `ta`, `adt`, `ablation`, `t1`, `t2` | speed, speed-step, control, time-step, traveling studies | 5--20 min each |
| `studies` | all ten manuscript studies in sequence | 1--2 hours |

Reproducibility: every run is seed-deterministic (base seed 42 or value-keyed
`SeedSequence` streams). On the archived environment, study reruns reproduced
every non-runtime field of the committed outputs bit for bit in our checks,
and figure regeneration is byte-identical. The legacy target `all` runs only
the original four studies and is kept for compatibility.

## Run a Modified Case

The resumable studies accept parameter flags and a scratch output directory:

```bash
python studies/study_multiviscosity_sweep.py --nu 0.1 0.05 --seeds 10 --out /tmp/x
```

Each resumable study stores a configuration fingerprint and refuses to resume
if the design changed, so incompatible cells are never combined; interrupted
or corrupted cells are detected and regenerated. Custom-parameter runs write
valid outputs but will not match the pinned verification values, which
describe the published configuration only.

## Repository Layout

- `relaxation_gbmc.py`: the shared particle update used by every study.
- `studies/`: the ten paper study drivers.
- `figure_scripts/`: regenerates the nine manuscript figures from committed
  data.
- `output/final_prepublication_tests/`: archived per-run tables,
  realization-level profiles, summaries, the transient's quadrature
  reference, and `PROVENANCE.md`.
- `expected_values.json`, `reproduce.py`: reproduction and verification entry
  point.
- `tests/`: software checks (58 tests).

## Citation

The code-and-data archive DOI will be added at public release. See
`CITATION.cff` for citation metadata.

## Acknowledgments

The authors thank Oliver Stalker for providing an early version of the
Python code.

**Principal Investigator:** [Professor Prabir Daripa](https://artsci.tamu.edu/mathematics/contact/profiles/prabir-daripa.html) — Texas A&M University, Department of Mathematics

Other projects from the Daripa Research Group are available on the
[group's GitHub page](https://github.com/Daripa-Research-Group).
