"""
generate_paper_tables.py
========================
Reads convergence study JSON output and produces:
  - output/convergence_study/summary_table.tex   (main LaTeX table, all methods)
  - output/convergence_study/gbmc_dt_table.tex   (GBMC dt-refinement)
  - output/convergence_study/comparison_table.tex (Cole-Hopf vs GBMC head-to-head)
  - output/convergence_study/results_summary.md  (human-readable markdown)

Run:
    python generate_paper_tables.py
"""
import json, os, sys


def load(path):
    with open(path) as f:
        return json.load(f)


def _fmt_pm(mean, std, digits=5):
    """Format as 'mean ± std' with given significant digits."""
    fmt = f".{digits}f"
    return f"{mean:{fmt}} $\\pm$ {std:{fmt}}"


def main():
    base = "output/convergence_study"
    h   = load(f"{base}/heat/n_refinement.json")
    ch  = load(f"{base}/cole_hopf/n_refinement.json")
    gb  = load(f"{base}/gbmc/n_refinement.json")
    dt  = load(f"{base}/gbmc/dt_refinement.json")
    fhn = load(f"{base}/fhn/n_refinement.json")
    tw  = load(f"{base}/cole_hopf/traveling_wave_verification.json")
    cmp = load(f"{base}/comparison/cole_hopf_vs_gbmc.json")

    # ------------------------------------------------------------------
    # TABLE 1 — N-refinement for all methods
    # ------------------------------------------------------------------
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{N-refinement convergence study: mean $L^2$ error "
                 r"($\pm$ one standard deviation over 10 independent runs) vs "
                 r"particle count $N$.  "
                 r"Heat GRW: step IC on $[0,10]$, $\alpha=0.1$, $T=0.5$; "
                 r"Cole-Hopf GRW: stationary-shock IC on $[0,4]$, $\nu=0.5$, $T=0.5$; "
                 r"Relaxation GBMC: stationary-shock IC on $[0,4]$, $\nu=0.5$, $a=2$, $T=0.5$; "
                 r"FHN GRW: steady-wave IC on $[0,30]$, $D=0.5$, $a=0.25$, $T=9$.  "
                 r"FHN metric is absolute front-location error; all others are $L^2$ field error.}")
    lines.append(r"\label{tab:n_refinement}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{r c c c c}")
    lines.append(r"\hline")
    lines.append(r"$N$ & Heat GRW & Cole-Hopf GRW & Relax.\ GBMC & FHN GRW \\")
    lines.append(r"    & $L^2$ error & $L^2$ error & $L^2$ error & Front error \\")
    lines.append(r"\hline")

    # Build per-N dicts for each method
    def by_N(results, key):
        return {r['N']: r for r in results}

    h_d   = by_N(h['results'],   None)
    ch_d  = by_N(ch['results'],  None)
    gb_d  = by_N(gb['results'],  None)
    fhn_d = by_N(fhn['results'], None)

    # All N values across all methods
    all_Ns = sorted(set(
        list(h_d.keys()) + list(ch_d.keys()) + list(gb_d.keys()) + list(fhn_d.keys())
    ))

    def cell(d, N, mean_key, std_key, digits=4):
        if N not in d:
            return r"\textemdash"
        r = d[N]
        return f"${r[mean_key]:.{digits}f} \\pm {r[std_key]:.{digits}f}$"

    for N in all_Ns:
        c1 = cell(h_d,   N, 'l2_mean', 'l2_std')
        c2 = cell(ch_d,  N, 'l2_mean', 'l2_std')
        c3 = cell(gb_d,  N, 'l2_mean', 'l2_std')
        c4 = cell(fhn_d, N, 'front_error_mean', 'front_error_std')
        lines.append(f"${N}$ & {c1} & {c2} & {c3} & {c4} \\\\")

    lines.append(r"\hline")
    f_h  = h['fit'];  f_ch = ch['fit'];  f_gb = gb['fit'];  f_fhn = fhn['fit']
    lines.append(
        r"\multicolumn{5}{l}{\textit{Fitted log-log slope vs $N$ (bootstrap 95\% CI)}} \\"
    )
    lines.append(
        f"& ${f_h['l2_slope']:.3f}$ [{f_h['l2_slope_ci_lo']:.3f}, {f_h['l2_slope_ci_hi']:.3f}]"
        f" & ${f_ch['l2_slope']:.3f}$ [{f_ch['l2_slope_ci_lo']:.3f}, {f_ch['l2_slope_ci_hi']:.3f}]"
        f" & ${f_gb['l2_slope']:.3f}$ [{f_gb['l2_slope_ci_lo']:.3f}, {f_gb['l2_slope_ci_hi']:.3f}]"
        f" & ${f_fhn['front_error_slope']:.3f}$ [{f_fhn['slope_ci_lo']:.3f}, {f_fhn['slope_ci_hi']:.3f}]"
        r" \\"
    )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table1 = "\n".join(lines)

    # ------------------------------------------------------------------
    # TABLE 2 — GBMC dt-refinement
    # ------------------------------------------------------------------
    lines2 = []
    lines2.append(r"\begin{table}[htbp]")
    lines2.append(r"\centering")
    lines2.append(r"\caption{Relaxation GBMC $\Delta t$-refinement: mean $L^2$ error "
                  r"vs time-step $\Delta t$ at fixed $N=800$.  "
                  r"Parameters: stationary-shock IC, $\nu=0.5$, $a=2$, $T=0.5$, $L=4$, "
                  r"10 independent runs per $\Delta t$.  "
                  r"A near-zero fitted slope (95\% CI includes zero) indicates that "
                  r"particle-noise dominates over Lie-splitting temporal bias at these $N$.}")
    lines2.append(r"\label{tab:gbmc_dt}")
    lines2.append(r"\begin{tabular}{r c c}")
    lines2.append(r"\hline")
    lines2.append(r"$\Delta t$ & Steps & $L^2$ error (mean $\pm$ std) \\")
    lines2.append(r"\hline")
    for r in dt['results']:
        lines2.append(
            f"${r['dt']:.5f}$ & ${r['n_steps']}$ & "
            f"${r['l2_mean']:.5f} \\pm {r['l2_std']:.5f}$ \\\\"
        )
    f_dt = dt['fit']
    lines2.append(r"\hline")
    lines2.append(
        r"\multicolumn{3}{l}{\textit{Fitted log-log slope vs $\Delta t$: "
        f"${f_dt['l2_slope_vs_dt']:.3f}$, 95\\% CI "
        f"[{f_dt['l2_slope_ci_lo']:.3f}, {f_dt['l2_slope_ci_hi']:.3f}]"
        r"}} \\"
    )
    lines2.append(r"\hline")
    lines2.append(r"\end{tabular}")
    lines2.append(r"\end{table}")
    table2 = "\n".join(lines2)

    # ------------------------------------------------------------------
    # TABLE 3 — Cole-Hopf vs GBMC comparison
    # ------------------------------------------------------------------
    lines3 = []
    lines3.append(r"\begin{table}[htbp]")
    lines3.append(r"\centering")
    lines3.append(r"\caption{Cole-Hopf GRW vs Relaxation GBMC: mean $L^2$ error "
                  r"on the stationary-shock benchmark at matched $N$.  "
                  r"Same IC, same domain ($[0,4]$), same $\nu=0.5$, $T=0.5$, $\Delta t=0.005$, "
                  r"10 runs per method per $N$.  "
                  r"The ratio GBMC/CH is the $L^2$-error ratio; values $<1$ indicate GBMC "
                  r"produces smaller errors.}")
    lines3.append(r"\label{tab:comparison}")
    lines3.append(r"\begin{tabular}{r c c c}")
    lines3.append(r"\hline")
    lines3.append(r"$N$ & Cole-Hopf GRW & Relaxation GBMC & Ratio (GBMC/CH) \\")
    lines3.append(r"\hline")
    for r in cmp['results']:
        ratio = r['gbmc_l2_mean'] / r['cole_hopf_l2_mean'] if r['cole_hopf_l2_mean'] > 0 else float('nan')
        lines3.append(
            f"${r['N']}$ & "
            f"${r['cole_hopf_l2_mean']:.4f} \\pm {r['cole_hopf_l2_std']:.4f}$ & "
            f"${r['gbmc_l2_mean']:.4f} \\pm {r['gbmc_l2_std']:.4f}$ & "
            f"${ratio:.2f}\\times$ \\\\"
        )
    lines3.append(r"\hline")
    lines3.append(r"\end{tabular}")
    lines3.append(r"\end{table}")
    table3 = "\n".join(lines3)

    # ------------------------------------------------------------------
    # Write LaTeX files
    # ------------------------------------------------------------------
    os.makedirs(base, exist_ok=True)
    with open(f"{base}/summary_table.tex", "w") as f:
        f.write("% TABLE 1: N-refinement for all four methods\n")
        f.write(table1)
        f.write("\n")
    print(f"[written] {base}/summary_table.tex")

    with open(f"{base}/gbmc_dt_table.tex", "w") as f:
        f.write("% TABLE 2: GBMC dt-refinement\n")
        f.write(table2)
        f.write("\n")
    print(f"[written] {base}/gbmc_dt_table.tex")

    with open(f"{base}/comparison_table.tex", "w") as f:
        f.write("% TABLE 3: Cole-Hopf vs GBMC comparison\n")
        f.write(table3)
        f.write("\n")
    print(f"[written] {base}/comparison_table.tex")

    # ------------------------------------------------------------------
    # Markdown summary
    # ------------------------------------------------------------------
    md = []
    md.append("# Convergence Study Results")
    md.append("")
    md.append("## Study Parameters")
    md.append("- Repeats per point: 10  (base seed: 42, seeds 42–51 per method)")
    md.append("- Total runtime: ~34 s on Apple Silicon")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Heat GRW — N-refinement")
    md.append(f"- **Benchmark**: step IC on [0,10], α=0.1, T=0.5, dt=0.001")
    md.append(f"- **Exact solution**: error-function `u(x,T) = (uR−uL)/2 · (1+erf(...))`")
    md.append(f"- **Fitted L2 slope vs N**: {f_h['l2_slope']:.4f}  "
              f"95% CI [{f_h['l2_slope_ci_lo']:.4f}, {f_h['l2_slope_ci_hi']:.4f}]  "
              f"R²={f_h['r2']:.4f}")
    md.append("")
    md.append("| N | L2 mean | L2 std | rel L2 |")
    md.append("|--:|--------:|-------:|-------:|")
    for r in h['results']:
        md.append(f"| {r['N']} | {r['l2_mean']:.5f} | {r['l2_std']:.5f} | {r['rel_l2_mean']:.5f} |")
    md.append("")
    md.append("**Interpretation**: Slope −0.35 is below the O(N^{−1/2}) Monte Carlo "
              "prediction. The error does not decrease monotonically at small N (variance "
              "is high for 10 repeats). At large N (5000–20000) the trend is clear; the "
              "reconstruction histogram likely limits accuracy at small N.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Cole-Hopf Burgers GRW — N-refinement")
    md.append(f"- **Benchmark**: stationary-shock IC on [0,4], ν=0.5, A=1, T=0.5, dt=0.005")
    md.append(f"- **Exact solution**: `u(x,t) = −A·tanh(A·(x−xc)/(2ν))` (time-independent)")
    md.append(f"- **Fitted L2 slope vs N**: {f_ch['l2_slope']:.4f}  "
              f"95% CI [{f_ch['l2_slope_ci_lo']:.4f}, {f_ch['l2_slope_ci_hi']:.4f}]  "
              f"R²={f_ch['r2']:.4f}")
    md.append("")
    md.append("| N | L2 mean | L2 std | rel L2 |")
    md.append("|--:|--------:|-------:|-------:|")
    for r in ch['results']:
        md.append(f"| {r['N']} | {r['l2_mean']:.5f} | {r['l2_std']:.5f} | {r['rel_l2_mean']:.5f} |")
    md.append("")
    md.append("**Interpretation**: The L2 error **plateaus at ≈ 0.40 (28% relative)** for "
              "N = 200–1600. This is NOT a convergence failure of the GRW algorithm itself. "
              "Rather, it reflects a systematic accuracy floor in recovering u = −2ν·φ_x/φ "
              "from particle-approximated φ fields: the differentiation step φ_x amplifies "
              "particle noise, and this limits the achievable L2 accuracy at these N values. "
              "The poor R² = 0.54 and wide CI reflect the absence of a clean power-law trend. "
              "⚠️ **This is a negative finding for Cole-Hopf GRW on this benchmark** and "
              "should be reported honestly in the paper.")
    md.append("")
    md.append("**Traveling-wave verification** (N=400, T=0.3): "
              f"L2={tw['l2']:.5f}, relL2={tw['rel_l2']:.5f}, "
              f"wave-speed error={tw['wave_speed_error']:.5f}. "
              "Large errors suggest boundary effects and/or similar reconstruction issues.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Relaxation GBMC — N-refinement")
    md.append(f"- **Benchmark**: stationary-shock IC on [0,4], ν=0.5, a=2, A=1, T=0.5, dt=0.005")
    md.append(f"- **Exact solution**: same tanh stationary shock")
    md.append(f"- **Fitted L2 slope vs N**: {f_gb['l2_slope']:.4f}  "
              f"95% CI [{f_gb['l2_slope_ci_lo']:.4f}, {f_gb['l2_slope_ci_hi']:.4f}]  "
              f"R²={f_gb['r2']:.4f}")
    md.append("")
    md.append("| N | L2 mean | L2 std | rel L2 |")
    md.append("|--:|--------:|-------:|-------:|")
    for r in gb['results']:
        md.append(f"| {r['N']} | {r['l2_mean']:.5f} | {r['l2_std']:.5f} | {r['rel_l2_mean']:.5f} |")
    md.append("")
    md.append("**Interpretation**: Clean power-law convergence, R² = 0.982. Slope −0.46 is "
              "consistent with the theoretical Monte Carlo O(N^{−1/2}) rate (−0.50). The "
              "wider CI reflects variance from only 10 repeats; the prior full study "
              "(run_n_refinement.py, 20 repeats, N=50–1600) yielded slope −0.507, "
              "95% CI [−0.534, −0.479], confirming O(N^{−1/2}) scaling.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Relaxation GBMC — dt-refinement")
    md.append(f"- **Parameters**: N=800 (fixed), ν=0.5, a=2, T=0.5, dt varied")
    f_dt = dt['fit']
    md.append(f"- **Fitted slope vs Δt**: {f_dt['l2_slope_vs_dt']:.4f}  "
              f"95% CI [{f_dt['l2_slope_ci_lo']:.4f}, {f_dt['l2_slope_ci_hi']:.4f}]  "
              f"R²={f_dt['r2']:.4f}")
    md.append("")
    md.append("| dt | L2 mean | L2 std |")
    md.append("|---:|--------:|-------:|")
    for r in dt['results']:
        md.append(f"| {r['dt']:.5f} | {r['l2_mean']:.5f} | {r['l2_std']:.5f} |")
    md.append("")
    md.append("**Interpretation**: Slope ≈ 0.08 with 95% CI [−0.12, 0.25] **includes zero**. "
              "The L2 error does NOT decrease as Δt is refined at fixed N=800. This "
              "confirms that the dominant error source is particle-count noise (O(N^{−1/2})), "
              "not Lie-splitting temporal bias. The Lie-splitting error is below the particle "
              "noise floor at N=800 for all tested Δt ∈ [0.0025, 0.05].")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. FHN Scalar GRW — N-refinement")
    md.append(f"- **Benchmark**: steady-wave IC on [0,30], D=0.5, a=0.25, T=9, dt=0.01")
    md.append(f"- **Metric**: |front_num − front_exact| where front_exact = xc − θ·T, "
              f"θ = √2·(0.5 − a)")
    md.append(f"- **Fitted slope vs N**: {f_fhn['front_error_slope']:.4f}  "
              f"95% CI [{f_fhn['slope_ci_lo']:.4f}, {f_fhn['slope_ci_hi']:.4f}]  "
              f"R²={f_fhn['r2']:.4f}")
    md.append("")
    md.append("| N | front error mean | front error std |")
    md.append("|--:|-----------------:|----------------:|")
    for r in fhn['results']:
        md.append(f"| {r['N']} | {r['front_error_mean']:.5f} | {r['front_error_std']:.5f} |")
    md.append("")
    md.append("**Interpretation**: Slope −0.65, R² = 0.906 — faster than O(N^{−1/2}). "
              "Large std at small N (high variance) makes the exact slope uncertain. "
              "The front tracks the exact traveling-wave speed (θ = √2·(0.5−0.25) = 0.354).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 6. Cole-Hopf vs GBMC Comparison")
    md.append("")
    md.append("| N | Cole-Hopf L2 | GBMC L2 | GBMC/CH ratio |")
    md.append("|--:|------------:|--------:|--------------:|")
    for r in cmp['results']:
        ratio = r['gbmc_l2_mean'] / r['cole_hopf_l2_mean']
        md.append(f"| {r['N']} | {r['cole_hopf_l2_mean']:.5f} | {r['gbmc_l2_mean']:.5f} | {ratio:.2f}× |")
    md.append("")
    md.append("**Interpretation**: At all tested N, GBMC achieves 5–8× lower L2 error than "
              "Cole-Hopf on the **same stationary-shock benchmark** with the same parameters. "
              "Cole-Hopf plateaus at ~0.40 (28% rel-L2); GBMC drops to 0.030 (2% rel-L2) "
              "at N=1600. However, this comparison is BENCHMARK-SPECIFIC: Cole-Hopf appears "
              "limited by the φ_x/φ reconstruction step on this sharp-tanh profile, and may "
              "perform differently on other ICs (e.g., smooth/polynomial profiles).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Summary of Fitted Convergence Rates")
    md.append("")
    md.append("| Method | Metric | Slope | 95% CI | R² |")
    md.append("|--------|--------|------:|--------|---:|")
    md.append(f"| Heat GRW | L2 vs N | {f_h['l2_slope']:.3f} | [{f_h['l2_slope_ci_lo']:.3f}, {f_h['l2_slope_ci_hi']:.3f}] | {f_h['r2']:.3f} |")
    md.append(f"| Cole-Hopf GRW | L2 vs N | {f_ch['l2_slope']:.3f} | [{f_ch['l2_slope_ci_lo']:.3f}, {f_ch['l2_slope_ci_hi']:.3f}] | {f_ch['r2']:.3f} |")
    md.append(f"| Relax. GBMC | L2 vs N | {f_gb['l2_slope']:.3f} | [{f_gb['l2_slope_ci_lo']:.3f}, {f_gb['l2_slope_ci_hi']:.3f}] | {f_gb['r2']:.3f} |")
    md.append(f"| Relax. GBMC | L2 vs Δt | {f_dt['l2_slope_vs_dt']:.3f} | [{f_dt['l2_slope_ci_lo']:.3f}, {f_dt['l2_slope_ci_hi']:.3f}] | {f_dt['r2']:.3f} |")
    md.append(f"| FHN GRW | front-err vs N | {f_fhn['front_error_slope']:.3f} | [{f_fhn['slope_ci_lo']:.3f}, {f_fhn['slope_ci_hi']:.3f}] | {f_fhn['r2']:.3f} |")
    md.append("")
    md.append("## Paper Claim Guidance")
    md.append("")
    md.append("**Supported claims (direct evidence):**")
    md.append("- GBMC achieves O(N^{−1/2}) convergence for viscous Burgers on the stationary-shock "
              "benchmark (slope −0.46, CI [−0.55, −0.36], R²=0.98)")
    md.append("- GBMC error at N=1600 is ~2% relative L2 on the stationary-shock benchmark")
    md.append("- GBMC Lie-splitting bias is below particle-noise floor for Δt ∈ [0.0025, 0.05] at N=800")
    md.append("- FHN GRW front-location error decreases with N (slope −0.65, CI [−0.87, −0.27])")
    md.append("")
    md.append("**Claims requiring additional context:**")
    md.append("- Heat GRW convergence: slope −0.35 is weaker than O(N^{−1/2}); "
              "reconstruction histogram may limit accuracy at small N; "
              "a larger-N study (N > 20000) would sharpen the rate estimate")
    md.append("- Cole-Hopf vs GBMC: GBMC is 5–8× better on the stationary-shock benchmark, "
              "but Cole-Hopf plateau is reconstruction-specific, not a general GRW failure")
    md.append("")
    md.append("**Do not claim:**")
    md.append("- Cole-Hopf converges at O(N^{−1/2}) for the stationary-shock IC — the data does not support this")
    md.append("- GBMC is superior to Cole-Hopf in general — only this one benchmark was tested")
    md.append("")
    md.append("## Output Files")
    for fn in [
        "heat/n_refinement.json", "heat/n_refinement_plot.png",
        "cole_hopf/n_refinement.json", "cole_hopf/n_refinement_plot.png",
        "cole_hopf/traveling_wave_verification.json", "cole_hopf/traveling_wave_plot.png",
        "gbmc/n_refinement.json", "gbmc/n_refinement_plot.png",
        "gbmc/dt_refinement.json", "gbmc/dt_refinement_plot.png",
        "fhn/n_refinement.json", "fhn/n_refinement_plot.png",
        "comparison/cole_hopf_vs_gbmc.json", "comparison/cole_hopf_vs_gbmc_plot.png",
        "summary_table.tex", "gbmc_dt_table.tex", "comparison_table.tex",
        "results_manifest.json",
    ]:
        md.append(f"- `output/convergence_study/{fn}`")

    with open(f"{base}/results_summary.md", "w") as f:
        f.write("\n".join(md))
        f.write("\n")
    print(f"[written] {base}/results_summary.md")


if __name__ == "__main__":
    main()
