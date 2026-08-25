# Paper 2 output provenance

The checked-in outputs in this directory are the archived data record for the
RB-GBMC manuscript.

| Study | Driver | Stored output |
|---|---|---|
| stationary particle sweep and viscosity fit | `studies/study_gbmc_production_n_refinement.py` | `gbmc_production_n_refinement/` |
| relaxation-speed sensitivity | `studies/study_relaxation_speed_sensitivity.py` | `gbmc_relaxation_speed_sensitivity/` |
| joint speed/time-step matrix | `studies/study_a_dt_interaction.py` | `gbmc_a_dt_interaction/` |
| conditional-mean transport control | `studies/study_conditional_mean_ablation.py` | `gbmc_conditional_mean_ablation/` |
| time-step sweep | `studies/study_t1_gbmc_dt_bias.py` | `gbmc_dt_bias/` |
| multi-viscosity sweep (fixed step) | `studies/study_multiviscosity_sweep.py` | `gbmc_multiviscosity_sweep/` |
| scaled-time-step companion | `studies/study_multiviscosity_scaled_dt.py` | `gbmc_multiviscosity_scaled_dt/` |
| transport-velocity timing pilot | `studies/study_ordering_pilot.py` | `gbmc_ordering_pilot/` |
| traveling shock | `studies/run_t2_S30.py` | `gbmc_traveling_shock/` |
| smooth nonstationary transient | `studies/study_smooth_transient.py` | `gbmc_smooth_transient/` |
| manuscript figures | `figure_scripts/regenerate_paper_figures.py` | `paper2_figures/` |

Two layers pin the reported results. `expected_values.json` with
`reproduce.py verify` checks 164 quantities in these archived summaries.
The manuscript project's claim-level provenance registry additionally maps
every number printed in the paper's numerical section to its archived source
file, field, permitted transformation, and printed rounding.

Directly computed error values and stored arrays use strict numerical
comparisons. Quantities obtained from SciPy nonlinear least squares use a
relative tolerance of `1e-8` to accommodate optimizer and linear-algebra
differences across compatible environments.

The archived configuration is Python 3.11.4 with NumPy 1.26.4, SciPy 1.17.1,
and Matplotlib 3.10.0; `requirements-lock.txt` pins it exactly. On that
configuration, study reruns reproduced every non-runtime field of these
archives in our checks. The release tag, license, and archive identifier are
added at public release.
