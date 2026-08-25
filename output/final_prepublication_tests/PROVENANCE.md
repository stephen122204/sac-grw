# Paper 2 output provenance

The checked-in outputs support the RB-GBMC manuscript only.

| Study | Driver | Stored output |
|---|---|---|
| stationary particle sweep and viscosity fit | `studies/study_gbmc_production_n_refinement.py` | `gbmc_production_n_refinement/` |
| relaxation-speed sensitivity | `studies/study_relaxation_speed_sensitivity.py` | `gbmc_relaxation_speed_sensitivity/` |
| time-step sweep | `studies/study_t1_gbmc_dt_bias.py` | `gbmc_dt_bias/` |
| traveling shock | `studies/run_t2_S30.py` | `gbmc_traveling_shock/` |
| manuscript figures | `figure_scripts/regenerate_paper_figures.py` | `paper2_figures/` |

`expected_values.json` pins the quantities printed in the paper. Directly
computed error values and stored arrays use strict numerical comparisons.
Quantities obtained from SciPy nonlinear least squares use a relative tolerance
of `1e-8` to accommodate optimizer and linear-algebra differences across
compatible environments.

The code was checked with Python 3.11, NumPy 1.26.4, SciPy 1.17.1, and
Matplotlib 3.10.0. The project has no public remote or archival identifier at
this stage.
