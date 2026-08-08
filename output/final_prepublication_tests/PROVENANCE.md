# Canonical output provenance — Paper 2 (RB-GBMC)

All checked-in study outputs live under `output/final_prepublication_tests/`,
one subdirectory per study. `python reproduce.py verify` checks these against
`../../expected_values.json`; each `python reproduce.py t1..t6` rewrites its
study's subdirectory. Branch: `rb-gbmc-paper2`.

## Study → output → paper location
| target | subdirectory | paper |
|---|---|---|
| t4 | `heat_extended/` | Table 1 |
| t5 | `fhn_extended/` | Table 2 (uniform-grid / exact-logistic norm) |
| t6 | `gbmc_production_n_refinement/` | Tables 3–4 |
| t1 | `gbmc_dt_bias/` | Table 5 |
| t2 | `gbmc_traveling_shock/` | Table 6 |
| t3 | `cole_hopf_plateau/` | Section 6 |
| (summary) | — | Table 7 |

## Tolerance policy (see `expected_values.json`)
Directly computed quantities — `E_bias`, `E_spread`, `E_total`, stored arrays,
and the production spread slope — are checked strictly / bit-identically.
Quantities obtained from the nonlinear tanh `curve_fit` — the fitted viscosity
`nu` and the derived `nu_std` / `xc_std` slopes — are checked at `rtol=1e-8`:
their final floating-point digits vary at the ~1e-11 level across SciPy/BLAS
builds and optimizer runs. This is optimizer-level variation, not a change in the
numerical result; all printed table values, slopes, interpretations, and plotted
curves are unchanged at paper precision.

## Paper figures
`paper_figures/` holds the 11 title-less figures the paper displays, regenerated
from the data above (never from a fresh study run) by `python reproduce.py figures`
(script `figure_scripts/regenerate_paper_figures.py`). Each study subdirectory also
carries its own diagnostic copies of these plots; the `paper_figures/` versions are
the authoritative paper set and are what `8-8-26/figuresv3/` compiles.

## Notes
- `gbmc_dt_bias/dt_bias_perdt_N6400.csv`: the `nu_fit` / `xc_fit` columns require
  scipy. A no-scipy run leaves the fallback `nu_fit = 0.5`; the committed file
  carries the scipy fit, which rerunning t1 with scipy reproduces to `rtol=1e-8`.
- The former `regen_data/` split has been consolidated into this tree; the t6
  study now writes the corrected viscosity directly.
- `output/` (apart from these checked-in summaries) and `outputs/` hold
  regenerable and scratch data and are git-ignored.

## Environment
Verified with Python 3.11.4, numpy 1.26.4, scipy 1.17.1, matplotlib 3.10.0.
`requirements.txt` requires numpy>=1.26,<3 / matplotlib>=3.8 / scipy>=1.11, a
range that covers the verified build: the directly computed quantities are
bit-identical across these numpy builds, and the `curve_fit`-derived quantities
agree within the `rtol=1e-8` policy above.
