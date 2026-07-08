# Provenance: `regen_data/`

Data and figures in this directory were **regenerated in the GRW-part2 repository at
commit `fb18341` (fb183416bf80ad3fc0b7f1e043a4f8d48c381d65) with the corrected tanh-fit
protocol**, using the same 50 seeds (42–91) as the original production run.

## Files

| File | Source (GRW-part2) |
|------|--------------------|
| `per_N_summary.csv` | `regen_output/gbmc_production/per_N_summary.csv` |
| `rates.json` | `regen_output/gbmc_production/rates.json` |
| `metadata.json` | `regen_output/gbmc_production/metadata.json` |
| `dt_bias_perdt_N6400.csv` | `regen_output/t1_gbmc_dt_bias/dt_bias_perdt_N6400.csv` |
| `production_gbmc_fitted_viscosity_vs_N_fixed.{pdf,png}` | `figuresv3_regen/` |
| `production_gbmc_bias_spread_total_vs_N_fixed.{pdf,png}` | `figuresv3_regen/` |
| `heat_bias_spread_total_vs_N_fixed.{pdf,png}` | `figuresv3_regen/` |

## Relationship to the checked-in study outputs

`per_N_summary.csv` here and
`output/final_prepublication_tests/gbmc_production_n_refinement/per_N_summary.csv`
were produced from the **same underlying particle simulations** (same seeds, same
N-sequence, same physical parameters):

- The `N`, `S_actual`, `n_failed`, `E_bias`, `E_spread`, `E_total`, and `identity_err`
  columns are **bit-identical** between the two files.
- Only the fitted columns (`xc_mean`, `xc_std`, `nu_mean`, `nu_std`) and the incidental
  `mean_rt` runtime column differ. The fitted columns **here supersede** the contaminated
  fits in the checked-in file: the original run's tanh fits were degraded by the fallback
  fit protocol, while this regeneration used the corrected scipy `curve_fit` tanh-fit
  protocol throughout.

## What this data backs in the manuscript

- **Table 6** (production GBMC N-refinement).
- The fitted-parameter convergence exponents in `rates.json`:
  `xc_std_slope = -0.5458` and `nu_std_slope = -0.5157`
  (the E_bias/E_spread/E_total slopes and CIs are unchanged from the checked-in run,
  e.g. spread slope -0.5099, CI [-0.5391, -0.4794]).
- The dt-study **recovered viscosities** in `dt_bias_perdt_N6400.csv`
  (`nu_fit` column): 0.5084, 0.5041, 0.5025, 0.5016, 0.5005 for
  dt = 0.01, 0.005, 0.0025, 0.00125, 0.000625 (exact nu = 0.5).
- The three `_fixed` draft figures used in the manuscript:
  `production_gbmc_fitted_viscosity_vs_N_fixed`,
  `production_gbmc_bias_spread_total_vs_N_fixed`,
  `heat_bias_spread_total_vs_N_fixed`.
