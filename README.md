# RB-GBMC: Relaxation-Brownian Gradient Monte Carlo for Viscous Burgers' Equation

Reproducible Python implementation and numerical studies of a
relaxation-Brownian gradient-particle method for viscous Burgers' equation.

## Overview

RB-GBMC is a stochastic gradient-particle algorithm for viscous Burgers'
equation `u_t + u u_x = nu u_xx`. Signed particles carry portions of the
spatial derivative `u_x`, and sorting followed by cumulative summation
recovers the solution. Two-speed relaxation labels represent the nonlinear
transport: each particle moves at one of two fixed speeds, with probabilities
chosen so that its conditional-mean velocity equals the local Burgers speed.
Independent Brownian increments supply a separately prescribed physical
viscosity. This repository contains the implementation, the archived
seed-level study outputs, the manuscript figures, and the verification
checks accompanying the paper:

> Stephen Abkin and Prabir Daripa,
> *A Relaxation-Brownian Gradient Monte Carlo Algorithm for Viscous Burgers'
> Equation* (manuscript; preprint and archive identifiers to be added at
> release).

## What this repository contributes

- Implements the complete relaxation-Brownian gradient-particle update
  (transport, sort, reconstruct, verify, redraw, diffuse) in one shared
  stepping routine used by every study.
- Reproduces the stationary, relaxation-speed, time-step, multi-viscosity,
  timing-pilot, traveling-shock, and smooth-transient studies reported in
  the paper.
- Maps every reported manuscript value to archived outputs through explicit
  checks: an expected-values verifier and a claim-level provenance registry
  in the manuscript project.

Principal findings, stated with their scope in the paper: seed-to-seed
spread is consistent with the Monte Carlo rate over the tested range; the
fitted effective viscosity approaches the prescribed value under particle
and time-step refinement; a paired conditional-mean control attributes the
measured speed-dependent broadening to sampled-label transport; the paired
excess approaches the analytic label scale as the shock layer sharpens; and
the same speed- and step-dependent excess appears on a smooth nonstationary
transient.

## Quick verification (seconds)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt

python reproduce.py verify   # check archived numerical results (no reruns)
pytest -q                    # software checks (no study reruns)
python reproduce.py figures  # regenerate all nine manuscript figures
```

- `requirements-lock.txt` recreates the archived environment
  (Python 3.11.4, numpy 1.26.4, scipy 1.17.1, matplotlib 3.10.0).
- `requirements.txt` permits compatible version ranges for general use; see
  `output/final_prepublication_tests/PROVENANCE.md` for the reproduction
  policy across environments.
- `verify` checks 164 pinned quantities in the archived summaries without
  rerunning any ensemble.
- `figures` rebuilds the nine manuscript figures from checked-in data only.

## Reproduce the studies

`python reproduce.py <target>` reruns a study from scratch and overwrites
its subdirectory under `output/final_prepublication_tests/` (the archived
paper-output directory). Reruns are seed-deterministic: on the archived
environment they reproduced every non-runtime field of the stored outputs in
our checks (only wall-clock columns change).

| Target | Study and the question it answers | Approx. runtime |
|---|---|---|
| `t6` | stationary particle refinement: does seed-to-seed spread shrink with N? | ~1-2 min |
| `ta` | relaxation-speed sensitivity: is the speed a substantive parameter? | ~5-20 min |
| `adt` | joint speed/time-step matrix: does refinement reduce the speed effect? | ~5-20 min |
| `ablation` | conditional-mean transport control: is the broadening due to sampled labels? | ~5-20 min |
| `multinu` | fixed-step multi-viscosity sweep: how does the label excess depend on viscosity? | ~1-2 min |
| `multinu-scaled` | viscosity-scaled step sweep: what remains when dt/nu is fixed? | ~10 min |
| `pilot` | transport-velocity timing diagnostic for the control residual | ~few min |
| `t1` | time-step refinement at fixed particle count | ~5-20 min |
| `t2` | traveling shock: does the method transport a moving profile? | ~5-20 min |
| `transient` | smooth nonstationary transient against a Cole-Hopf reference | ~few min |
| `studies` | all ten manuscript studies in sequence (`studies-all` is an alias) | hours |
| `figures` | rebuild the nine figures from archived data (no reruns) | seconds |

The legacy target `all` runs only the original four studies (`t6 ta t1 t2`)
and is kept for compatibility.

## Reproducibility design

- Seeds are explicit: base seed 42 with consecutive identifiers, or
  value-keyed `SeedSequence` streams for the multi-viscosity, pilot, and
  transient studies, so subsets and reorderings reproduce identical cells.
- Pairing is stated per study: arms sharing Brownian streams at a fixed
  configuration are paired through common random numbers, aligned by sorted
  rank; comparisons across particle counts, step counts, or viscosities are
  not paired.
- Manuscript fits are strict: every reported fitted value comes from SciPy
  `curve_fit` (recorded per run where applicable), and a fit failure aborts
  the study rather than falling back to a cruder estimate.
- Figure regeneration is deterministic (fixed PDF metadata date), and the
  manuscript checker requires the shipped figures to match the regenerated
  files byte for byte.
- The archives include realization-level profiles, so the published tables
  and the realization-level bootstrap intervals can be recomputed from the
  archive alone; the test suite does exactly that.

## Repository structure

- `relaxation_gbmc.py` — the common particle update used by every study.
- `studies/` — study drivers: stationary and traveling shocks, the joint
  speed/step matrix, the conditional-mean control, the fixed- and
  scaled-step multi-viscosity sweeps, the timing pilot, and the smooth
  transient.
- `figure_scripts/` — regenerates the nine manuscript figures from archived
  data.
- `output/final_prepublication_tests/` — the archived paper-output
  directory: per-run tables, realization-level profiles, summaries, the
  transient's quadrature reference with its documented tolerance, and
  `PROVENANCE.md`.
- `expected_values.json` + `reproduce.py verify` — the numerical gate.
- `tests/` — software checks (58 tests: update order, exact signed-mass
  conservation, split random streams, strict fitting, resume safety, seed
  identity, archive completeness).

## Custom runs and resume behavior

The multi-viscosity, scaled-step, pilot, and transient studies are resumable
by cell: each completed cell stores its per-run rows and a profile array,
and `manifest.json` records a configuration fingerprint. A resume is refused
when the fingerprint differs, so changing the seed count, particle count,
time step, final time, reconstruction count, viscosity list, arm list,
window design, fit-bound rule, or seed scheme cannot silently combine
incompatible cells. Interrupted, corrupted, or incomplete cells are detected
and regenerated.

```bash
python studies/study_multiviscosity_sweep.py --nu 0.1 0.05   # subset of cells
python studies/study_multiviscosity_sweep.py --seeds 10      # smaller ensemble
python studies/study_multiviscosity_sweep.py --out /tmp/x    # scratch output
```

Seeds for these studies are keyed by parameter value, never by list
position, and a subset invocation recomputes the summary over every
completed cell rather than shrinking it. Viscosities outside the canonical
list are refused: changing the design is a new study. Custom-parameter runs
write valid outputs but will not match the pinned verification values, which
describe the published configuration only.

## Data archive, license, and release status

The archived outputs in `output/final_prepublication_tests/` are the data
record for the paper. At public release this repository gains a version tag,
a LICENSE file (currently `NOASSERTION` in `CITATION.cff`, pending the
authors' choice), and a code-and-data DOI; the paper's Code and Data
Availability statement will then cite that DOI.

Suggested repository metadata: name `rb-gbmc-viscous-burgers`; description
"Reproducible Python implementation and numerical studies of a
relaxation-Brownian gradient-particle method for viscous Burgers' equation";
topics `burgers-equation`, `monte-carlo`, `particle-methods`,
`numerical-analysis`, `stochastic-methods`,
`partial-differential-equations`, `python`, `reproducible-research`.

## Authors

Stephen Abkin and Prabir Daripa (Department of Mathematics, Texas A&M
University). Citation metadata is in `CITATION.cff`; see the manuscript for
acknowledgments and declarations.
