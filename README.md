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
python -m pip install -r requirements-coupling.txt
```

## Reproduce the paper

```bash
python reproduce.py paper
```

This rebuilds the reported numerical summaries in
`reproduction/analysis/numbers.json` and creates the five figures in
`reproduction/figures/`.
No manuscript files or TeX installation are needed.

These commands analyze the archived experiments under `output/`. They do not
repeat the simulations or measure new timings. Each summary in `numbers.json`
records the archive it was read from. The evidence map below connects the
paper's results with those archives. The experiment scripts in
`studies/` document the calculations, but some require intermediate inputs
that are not included in this release. They are not a supported clean-clone
rerun workflow. Use the reproduction commands above for the published data
and `run_coupling.py` for new simulations.
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

`FULL` switches by mass sign, `RAW` reflects by spatial rank, `WITHIN` reflects
within each sign class, and `FINAL` switches only at the last stage. The `heat`,
`burgers`, and `cubic` fluxes and the `gaussian` and `shock` initial profiles are
supported. Use an even particle count and a final time divisible by the time
step. For other initial particles or flux derivatives, use `advance_pair` in
`coupled_gradient_particles.py`.

## Code and data

- `coupled_gradient_particles.py`: coupled mean-transport stepper.
- `gradient_particles.py`: single-simulation update, particle initialization, and field reconstruction.
- `reproduction/analysis/`: scripts that rebuild the paper's numbers and figures,
  with `numbers.json` and the Figure 1 state.
- `reproduction/evidence/`: recomputed particle fields behind Table 1 and Figure 2.
- `output/`: the archived experiments the paper reports, one directory per study.
- `studies/`: the drivers that produced those archives and the modules they import.

The supported reproduction commands select the data used by the paper.
The nonlinear results are recomputed from `reproduction/evidence/principal_fields.npz`.

Citation details are in [CITATION.cff](CITATION.cff). No DOI has been assigned.

## Evidence map

Paths below are relative to the repository root. `reproduction/analysis/numbers.json`
records the computed values and their source paths. Each directory under
`output/` is named for the study it holds.

| Result | Source data or calculation |
| --- | --- |
| Principal field comparison and spatial variance figure | `reproduction/evidence/principal_fields.npz`, with statistics in `reproduction/analysis/build_numbers.py` |
| Costs per pair | `output/principal_burgers_runs/costs.json`, and pair versus single in `output/heldout_work_experiment_a/compare.json` |
| Conditional final stage, controls, and flux comparison | `output/controls_and_costs/fields.npz` and `rows.csv` |
| Profiles and observation times | `output/profile_time_flux_sweep/mechanism.json` |
| Matched-pair distances | `output/matched_pair_separation/separation.json`, the `matched_state` rows. The `own_history` rows describe a different comparison |
| Work at a target, Experiment A | `output/heldout_work_experiment_a/compare.json` |
| Work at a target, Experiment B | `output/heldout_work_experiment_b/worktarget.json`, with the interleaved timing summary in `output/heldout_work_experiment_b/timing_check.json` |
| Pointwise variance ratios | `output/pointwise_observables/observable.json` and `f_intervals.json` |
| Nonlinear-observable mean squared errors | `reproduction/evidence/principal_fields.npz`, with the empirical MSE recomputed by `reproduction/analysis/build_numbers.py` |
| Bias crossover | `output/bias_crossover_cell/joint_cell.json` |
| Sign configurations | `output/sign_structure_map/applicability.json` |
| Final-stage identity checks | `output/final_stage_identity_checks/finalstage.json`, with averages in `output/final_stage_switch/finalstep.json` |
| One-sign identity and separated-state divergence | `output/coupling_divergence_seeds/divergence.json`, with earlier checks in `output/principal_burgers_runs/corrections.json` |
| Single-replica agreement with the original update | `output/single_replica_law_check/audit.json` |
| Coordinatewise-monotonicity counterexample | `output/monotonicity_counterexample/counterexample.json`. This does not establish a reversed terminal-variance ordering |
| Interval calibration | `output/interval_calibration/calibration.json`. The earlier common-shape calibration remains in `output/principal_burgers_runs/calibration.json` but is not the reported two-shape calculation |
| Construction illustration | `reproduction/analysis/fig1_state.npz` and `reproduction/analysis/make_figures.py` |

Experiment A records Python 3.11.4, NumPy 1.26.4, SciPy 1.17.1, and
macOS 26.6.2 on ARM64. Its reported execution times are medians over
32 complete estimates per method. The pinned requirements describe the
reproduction environment, which differs from the original experiment environment.
The later interleaved timing archive stores aggregate times without individual
repetitions or a full hardware record, so no timing confidence interval is available
from that file. Timing summaries should not be interpreted as portable benchmarks.

## Acknowledgments

**Principal Investigator:** [Professor Prabir Daripa](https://artsci.tamu.edu/mathematics/contact/profiles/prabir-daripa.html) — Texas A&M University, Department of Mathematics.

Other projects from the Daripa Research Group are available on the
[group's GitHub page](https://github.com/Daripa-Research-Group).
