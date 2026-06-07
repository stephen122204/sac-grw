# Final Pre-Publication Testing Summary

## Overview
- **Total figures generated:** 38/38
- **Total wall time:** 0.0s

---
## Task 1: GBMC Time-Step Bias (High-N)

- **Status:** complete
- **N:** 800   **Seeds:** 5
- **Bias slope vs dt:** 0.0863
- **95% CI:** [-0.4517, 0.3260]
- **Conclusion:** bias_below_spread: E_bias < 2*E_spread at all dt; bias remains below ensemble-estimation noise

> **Interpretation:** If E_bias remains below E_spread at all dt, the Lie-splitting
> bias is overwhelmed by particle noise at this N and S; a larger N or S is required.
> If a clean slope emerges with CI > 0, the Lie-splitting bias is measurable.

---
## Task 2: Traveling-Shock GBMC Validation

- **Status:** complete
- **N-refinement slope (T=1.0):** -0.5690
- **95% CI:** [-0.6044, -0.5192]

> **Interpretation:** A slope near -0.5 with CI below -0.25 would confirm O(N^{-1/2})
> convergence for the traveling shock, matching the stationary-shock result.

---
## Task 3: Cole-Hopf Error Plateau Diagnosis

- **Status:** complete
- **Observed plateau L2:** 0.4000
- **Primary cause:** combination_particle_reconstruction_and_differentiation
- **Differentiation error (exact phi):** 0.0007
- **Differentiation conclusion:** transform_differentiation_negligible
- **Boundary mismatch conclusion:** boundary_mismatch

> **Negative finding (previously established):** The Cole-Hopf GRW has a
> systematic L2 plateau at ~0.40. This study decomposes the source.
> The R²=0.54, slope CI touching zero confirms no clean convergence.

---
## Task 4: Extended Heat Convergence

- **Status:** complete
- **N sequence:** [500, 1000, 2000, 5000, 10000, 20000, 50000]   **Seeds:** 30
- **Total error slope:** -0.4568
- **95% CI:** [-0.4830, -0.4253]
- **Bias slope:** -0.2617
- **Spread slope:** -0.4676

> **No dt study:** Heat GRW is exact in time (Brownian increments are exact).
> Heat GRW is exact in time: Brownian increments are drawn exactly from N(0, 2*alpha*dt), with no discretization error in the stochastic dynamics. The only approximation is in the initial condition (number of particles N) and the histogram reconstruction. Therefore, the convergence study only needs N-refinement.

> **Interpretation:** E_spread should converge at ~O(N^{-1/2}).
> A persistent E_bias indicates systematic reconstruction error
> (histogram binning at the shock discontinuity).

---
## Task 5: Extended FHN Convergence

- **Status:** complete
- **N sequence:** [100, 200, 500, 1000, 2000, 5000]   **Seeds:** 30
- **Profile L2 slope:** -0.4789
- **Profile L2 95% CI:** [-0.5023, -0.4574]
- **Front center error slope:** -0.5238
- **Front speed error slope:** -0.5238
- **Aligned-profile error slope:** -0.4702

> **Interpretation:** The aligned-profile error isolates shape convergence
> from front-position error. A clean slope near -0.5 for the aligned error
> indicates the profile shape converges at the particle rate, while
> center error may converge faster (-1 to -2) due to averaging effects.

---
## Figure Inventory

| Task | Figure | PNG | PDF |
|------|--------|-----|-----|
| task1 | `gbmc_bias_vs_dt` | ✓ | ✓ |
| task1 | `gbmc_spread_total_vs_dt` | ✓ | ✓ |
| task1 | `gbmc_fitted_viscosity_vs_dt` | ✓ | ✓ |
| task1 | `gbmc_center_error_vs_dt` | ✓ | ✓ |
| task1 | `gbmc_profiles_selected_dt` | ✓ | ✓ |
| task1 | `gbmc_dt_runtime` | ✓ | ✓ |
| task2 | `gbmc_traveling_profiles_by_time` | ✓ | ✓ |
| task2 | `gbmc_traveling_center_vs_time` | ✓ | ✓ |
| task2 | `gbmc_traveling_center_error` | ✓ | ✓ |
| task2 | `gbmc_traveling_speed_error` | ✓ | ✓ |
| task2 | `gbmc_traveling_error_vs_N` | ✓ | ✓ |
| task2 | `gbmc_traveling_error_vs_dt` | ✓ | ✓ |
| task2 | `gbmc_traveling_sharp_layer` | ✓ | ✓ |
| task3 | `cole_hopf_error_vs_domain` | ✓ | ✓ |
| task3 | `cole_hopf_deterministic_transform_error` | ✓ | ✓ |
| task3 | `cole_hopf_particle_vs_deterministic_phi` | ✓ | ✓ |
| task3 | `cole_hopf_error_vs_output_grid` | ✓ | ✓ |
| task3 | `cole_hopf_plateau_decomposition` | ✓ | ✓ |
| task4 | `heat_bias_spread_total_vs_N` | ✓ | ✓ |
| task4 | `heat_spread_vs_N` | ✓ | ✓ |
| task4 | `heat_error_vs_output_grid` | ✓ | ✓ |
| task4 | `heat_profiles_selected_N` | ✓ | ✓ |
| task4 | `heat_runtime_vs_N` | ✓ | ✓ |
| task4 | `heat_error_vs_dt` | ✓ | ✓ |
| task5 | `fhn_profile_error_vs_N` | ✓ | ✓ |
| task5 | `fhn_front_error_vs_N` | ✓ | ✓ |
| task5 | `fhn_error_vs_dt` | ✓ | ✓ |
| task5 | `fhn_front_center_vs_time` | ✓ | ✓ |
| task5 | `fhn_front_speed_error` | ✓ | ✓ |
| task5 | `fhn_aligned_profile_error` | ✓ | ✓ |
| task5 | `fhn_profiles_selected_N` | ✓ | ✓ |
| task5 | `fhn_runtime_vs_N` | ✓ | ✓ |
| task6 | `production_gbmc_spread_vs_N` | ✓ | ✓ |
| task6 | `production_gbmc_bias_spread_total_vs_N` | ✓ | ✓ |
| task6 | `production_gbmc_center_width_std_vs_N` | ✓ | ✓ |
| task6 | `production_gbmc_profiles_selected_N` | ✓ | ✓ |
| task6 | `production_gbmc_fitted_viscosity_vs_N` | ✓ | ✓ |
| task6 | `production_gbmc_runtime_vs_N` | ✓ | ✓ |

---
## Key Decisions for Paper Writing

1. **GBMC convergence rate:** Report N-slope with 95% CI. If CI straddles -0.5,
   state `O(N^{-1/2})` is consistent but not proven.

2. **GBMC dt-bias:** If E_bias < E_spread at all dt, report as
   'bias below estimation precision; Lie-splitting bias not measurable at N=6400'.
   This is an honest negative finding, not a failure.

3. **Cole-Hopf plateau:** Report as confirmed negative finding with primary cause.
   Do not claim convergence for Cole-Hopf GRW.

4. **Heat GRW:** Report spread slope; document why no dt study is needed.
   If bias is non-negligible at small N, attribute to histogram binning at shock.

5. **FHN:** Report both profile and aligned-profile slopes. If they differ,
   the discrepancy is due to front-position variance (report both rates).

6. **Traveling shock:** If stop_triggered=True, report the physical reason
   (subcharacteristic violation, mass loss, etc.) as a finding.

---
_This summary was auto-generated by `generate_final_manifest.py`._
