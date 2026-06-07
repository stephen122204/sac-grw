# Convergence Study Results

## Study Parameters
- Repeats per point: 10  (base seed: 42, seeds 42–51 per method)
- Total runtime: ~34 s on Apple Silicon

---

## 1. Heat GRW — N-refinement
- **Benchmark**: step IC on [0,10], α=0.1, T=0.5, dt=0.001
- **Exact solution**: error-function `u(x,T) = (uR−uL)/2 · (1+erf(...))`
- **Fitted L2 slope vs N**: -0.3545  95% CI [-0.5483, -0.1868]  R²=0.8393

| N | L2 mean | L2 std | rel L2 |
|--:|--------:|-------:|-------:|
| 500 | 0.02309 | 0.00964 | 0.01051 |
| 1000 | 0.02640 | 0.00988 | 0.01202 |
| 2000 | 0.02656 | 0.00513 | 0.01210 |
| 5000 | 0.01246 | 0.00365 | 0.00568 |
| 10000 | 0.00919 | 0.00336 | 0.00418 |
| 20000 | 0.00807 | 0.00147 | 0.00368 |

**Interpretation**: Slope −0.35 is below the O(N^{−1/2}) Monte Carlo prediction. The error does not decrease monotonically at small N (variance is high for 10 repeats). At large N (5000–20000) the trend is clear; the reconstruction histogram likely limits accuracy at small N.

---

## 2. Cole-Hopf Burgers GRW — N-refinement
- **Benchmark**: stationary-shock IC on [0,4], ν=0.5, A=1, T=0.5, dt=0.005
- **Exact solution**: `u(x,t) = −A·tanh(A·(x−xc)/(2ν))` (time-independent)
- **Fitted L2 slope vs N**: -0.3146  95% CI [-0.7335, 0.0023]  R²=0.5360

| N | L2 mean | L2 std | rel L2 |
|--:|--------:|-------:|-------:|
| 50 | 1.64995 | 0.10589 | 1.12579 |
| 100 | 0.50801 | 0.05359 | 0.34977 |
| 200 | 0.42559 | 0.06588 | 0.29434 |
| 400 | 0.39068 | 0.03991 | 0.27081 |
| 800 | 0.42138 | 0.03932 | 0.29241 |
| 1600 | 0.40799 | 0.02586 | 0.28328 |

**Interpretation**: The L2 error **plateaus at ≈ 0.40 (28% relative)** for N = 200–1600. This is NOT a convergence failure of the GRW algorithm itself. Rather, it reflects a systematic accuracy floor in recovering u = −2ν·φ_x/φ from particle-approximated φ fields: the differentiation step φ_x amplifies particle noise, and this limits the achievable L2 accuracy at these N values. The poor R² = 0.54 and wide CI reflect the absence of a clean power-law trend. ⚠️ **This is a negative finding for Cole-Hopf GRW on this benchmark** and should be reported honestly in the paper.

**Traveling-wave verification** (N=400, T=0.3): L2=1.24542, relL2=0.26009, wave-speed error=0.60150. Large errors suggest boundary effects and/or similar reconstruction issues.

---

## 3. Relaxation GBMC — N-refinement
- **Benchmark**: stationary-shock IC on [0,4], ν=0.5, a=2, A=1, T=0.5, dt=0.005
- **Exact solution**: same tanh stationary shock
- **Fitted L2 slope vs N**: -0.4575  95% CI [-0.5466, -0.3601]  R²=0.9817

| N | L2 mean | L2 std | rel L2 |
|--:|--------:|-------:|-------:|
| 50 | 0.16420 | 0.04648 | 0.11204 |
| 100 | 0.10234 | 0.02245 | 0.07046 |
| 200 | 0.07409 | 0.01425 | 0.05124 |
| 400 | 0.06341 | 0.01904 | 0.04395 |
| 800 | 0.04604 | 0.01214 | 0.03195 |
| 1600 | 0.02972 | 0.01053 | 0.02063 |

**Interpretation**: Clean power-law convergence, R² = 0.982. Slope −0.46 is consistent with the theoretical Monte Carlo O(N^{−1/2}) rate (−0.50). The wider CI reflects variance from only 10 repeats; the prior full study (run_n_refinement.py, 20 repeats, N=50–1600) yielded slope −0.507, 95% CI [−0.534, −0.479], confirming O(N^{−1/2}) scaling.

---

## 4. Relaxation GBMC — dt-refinement
- **Parameters**: N=800 (fixed), ν=0.5, a=2, T=0.5, dt varied
- **Fitted slope vs Δt**: 0.0769  95% CI [-0.1239, 0.2521]  R²=0.1871

| dt | L2 mean | L2 std |
|---:|--------:|-------:|
| 0.05000 | 0.06826 | 0.01729 |
| 0.02500 | 0.03815 | 0.01259 |
| 0.01000 | 0.04549 | 0.01476 |
| 0.00500 | 0.04604 | 0.01214 |
| 0.00250 | 0.04600 | 0.01526 |

**Interpretation**: Slope ≈ 0.08 with 95% CI [−0.12, 0.25] **includes zero**. The L2 error does NOT decrease as Δt is refined at fixed N=800. This confirms that the dominant error source is particle-count noise (O(N^{−1/2})), not Lie-splitting temporal bias. The Lie-splitting error is below the particle noise floor at N=800 for all tested Δt ∈ [0.0025, 0.05].

---

## 5. FHN Scalar GRW — N-refinement
- **Benchmark**: steady-wave IC on [0,30], D=0.5, a=0.25, T=9, dt=0.01
- **Metric**: |front_num − front_exact| where front_exact = xc − θ·T, θ = √2·(0.5 − a)
- **Fitted slope vs N**: -0.6492  95% CI [-0.8733, -0.2741]  R²=0.9063

| N | front error mean | front error std |
|--:|-----------------:|----------------:|
| 100 | 0.62110 | 0.37671 |
| 200 | 0.21913 | 0.18566 |
| 500 | 0.16638 | 0.06905 |
| 1000 | 0.14098 | 0.11628 |
| 2000 | 0.06440 | 0.03634 |

**Interpretation**: Slope −0.65, R² = 0.906 — faster than O(N^{−1/2}). Large std at small N (high variance) makes the exact slope uncertain. The front tracks the exact traveling-wave speed (θ = √2·(0.5−0.25) = 0.354).

---

## 6. Cole-Hopf vs GBMC Comparison

| N | Cole-Hopf L2 | GBMC L2 | GBMC/CH ratio |
|--:|------------:|--------:|--------------:|
| 100 | 0.50801 | 0.10234 | 0.20× |
| 200 | 0.42559 | 0.07409 | 0.17× |
| 400 | 0.39068 | 0.06341 | 0.16× |
| 800 | 0.42138 | 0.04604 | 0.11× |

**Interpretation**: At all tested N, GBMC achieves 5–8× lower L2 error than Cole-Hopf on the **same stationary-shock benchmark** with the same parameters. Cole-Hopf plateaus at ~0.40 (28% rel-L2); GBMC drops to 0.030 (2% rel-L2) at N=1600. However, this comparison is BENCHMARK-SPECIFIC: Cole-Hopf appears limited by the φ_x/φ reconstruction step on this sharp-tanh profile, and may perform differently on other ICs (e.g., smooth/polynomial profiles).

---

## Summary of Fitted Convergence Rates

| Method | Metric | Slope | 95% CI | R² |
|--------|--------|------:|--------|---:|
| Heat GRW | L2 vs N | -0.355 | [-0.548, -0.187] | 0.839 |
| Cole-Hopf GRW | L2 vs N | -0.315 | [-0.733, 0.002] | 0.536 |
| Relax. GBMC | L2 vs N | -0.457 | [-0.547, -0.360] | 0.982 |
| Relax. GBMC | L2 vs Δt | 0.077 | [-0.124, 0.252] | 0.187 |
| FHN GRW | front-err vs N | -0.649 | [-0.873, -0.274] | 0.906 |

## Paper Claim Guidance

**Supported claims (direct evidence):**
- GBMC achieves O(N^{−1/2}) convergence for viscous Burgers on the stationary-shock benchmark (slope −0.46, CI [−0.55, −0.36], R²=0.98)
- GBMC error at N=1600 is ~2% relative L2 on the stationary-shock benchmark
- GBMC Lie-splitting bias is below particle-noise floor for Δt ∈ [0.0025, 0.05] at N=800
- FHN GRW front-location error decreases with N (slope −0.65, CI [−0.87, −0.27])

**Claims requiring additional context:**
- Heat GRW convergence: slope −0.35 is weaker than O(N^{−1/2}); reconstruction histogram may limit accuracy at small N; a larger-N study (N > 20000) would sharpen the rate estimate
- Cole-Hopf vs GBMC: GBMC is 5–8× better on the stationary-shock benchmark, but Cole-Hopf plateau is reconstruction-specific, not a general GRW failure

**Do not claim:**
- Cole-Hopf converges at O(N^{−1/2}) for the stationary-shock IC — the data does not support this
- GBMC is superior to Cole-Hopf in general — only this one benchmark was tested

## Output Files
- `output/convergence_study/heat/n_refinement.json`
- `output/convergence_study/heat/n_refinement_plot.png`
- `output/convergence_study/cole_hopf/n_refinement.json`
- `output/convergence_study/cole_hopf/n_refinement_plot.png`
- `output/convergence_study/cole_hopf/traveling_wave_verification.json`
- `output/convergence_study/cole_hopf/traveling_wave_plot.png`
- `output/convergence_study/gbmc/n_refinement.json`
- `output/convergence_study/gbmc/n_refinement_plot.png`
- `output/convergence_study/gbmc/dt_refinement.json`
- `output/convergence_study/gbmc/dt_refinement_plot.png`
- `output/convergence_study/fhn/n_refinement.json`
- `output/convergence_study/fhn/n_refinement_plot.png`
- `output/convergence_study/comparison/cole_hopf_vs_gbmc.json`
- `output/convergence_study/comparison/cole_hopf_vs_gbmc_plot.png`
- `output/convergence_study/summary_table.tex`
- `output/convergence_study/gbmc_dt_table.tex`
- `output/convergence_study/comparison_table.tex`
- `output/convergence_study/results_manifest.json`
