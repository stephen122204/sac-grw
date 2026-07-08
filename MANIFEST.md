# Manuscript artifact manifest

Maps every table and figure of the manuscript ("Gradient Random Walk Methods for
Diffusive PDEs and a Relaxation–Brownian Particle Scheme for Viscous Burgers'
Equation", Abkin & Daripa) to the study that produces it and the checked-in data
that backs it. `python reproduce.py <target>` reruns a study;
`python reproduce.py verify` checks the checked-in numbers without rerunning.

| Manuscript artifact | Study / target | Script | Checked-in data |
|---|---|---|---|
| Table 1 (heat GRW N-refinement) | `t4` | `studies/study_t4_heat_extended.py` | `output/final_prepublication_tests/heat_extended/{summary_by_N.csv,summary.json}` |
| Table 2 (FHN N-refinement) + FHN dt quartet | `t5` | `studies/study_t5_fhn_extended.py` | `output/final_prepublication_tests/fhn_extended/{summary_by_N.csv,summary.json}` (`dt_refinement` block) |
| Tables 5–6 (production GBMC N-refinement) + production figures | `t6` | `studies/study_gbmc_production_n_refinement.py` | `output/final_prepublication_tests/gbmc_production_n_refinement/{per_N_summary.csv,rates.json,metadata.json}`; **Table 6 fitted columns and the fitted-viscosity figure come from `regen_data/`** (corrected tanh-fit regeneration; see `regen_data/PROVENANCE.md`) |
| Table 7 (GBMC dt-bias, N=6400) | `t1` | `studies/study_t1_gbmc_dt_bias.py` | `output/final_prepublication_tests/gbmc_dt_bias/{dt_bias_summary_N6400.json,dt_bias_perdt_N6400.csv}`; recovered viscosities from `regen_data/dt_bias_perdt_N6400.csv` |
| Table 8 (traveling shock) + 2 traveling figures | `t2` (**S=30**) | `studies/run_t2_S30.py` (wraps `studies/study_t2_traveling_shock.py`) | `output/final_prepublication_tests/gbmc_traveling_shock/summary.json` (T=1 rows) |
| Cole–Hopf decomposition figure + Section-6 numbers | `t3` | `studies/study_t3_cole_hopf_plateau.py` | `output/final_prepublication_tests/cole_hopf_plateau/{plateau_decomposition.json,study_A..D*.json}` |

## Draft figures sourced from `regen_data/` (`_fixed` versions)

Three figures in the manuscript draft are the corrected `_fixed` versions carried in
`regen_data/`, not the figure files under `output/final_prepublication_tests/`:

- `regen_data/production_gbmc_fitted_viscosity_vs_N_fixed.{pdf,png}`
- `regen_data/production_gbmc_bias_spread_total_vs_N_fixed.{pdf,png}`
- `regen_data/heat_bias_spread_total_vs_N_fixed.{pdf,png}`

They were regenerated at GRW-part2 commit `fb18341` with the corrected tanh-fit
protocol (same seeds); provenance and the supersession relationship are documented
in `regen_data/PROVENANCE.md`.

## Notes

- `t2`: the paper run is S=30. `run_task2()`'s S=10 default is an exploration
  setting; always use `studies/run_t2_S30.py` (or `reproduce.py t2` / `reproduce.py all`)
  for paper numbers.
- `reproduce.py all` runs T1–T5 via `studies/run_prepublication_studies.py` (T2 at S=30)
  and regenerates `output/final_prepublication_tests/final_manifest/`; T6 must be
  run separately.
