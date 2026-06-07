"""
generate_final_manifest.py
==========================
Reads all pre-publication study outputs and generates:
  - results_manifest.json      (all numerical results, fitted rates, CIs)
  - results_manifest.csv
  - figure_manifest.csv        (all expected figures with existence check)
  - table_manifest.csv
  - FINAL_TESTING_SUMMARY.md   (human-readable summary for paper writing)

Usage:
    python generate_final_manifest.py
or called programmatically from run_prepublication_studies.py.
"""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = 'output/final_prepublication_tests'
OUT = os.path.join(BASE, 'final_manifest')

# All expected figures (name -> study)
EXPECTED_FIGURES = {
    'task1': [
        'gbmc_bias_vs_dt',
        'gbmc_spread_total_vs_dt',
        'gbmc_fitted_viscosity_vs_dt',
        'gbmc_center_error_vs_dt',
        'gbmc_profiles_selected_dt',
        'gbmc_dt_runtime',
    ],
    'task2': [
        'gbmc_traveling_profiles_by_time',
        'gbmc_traveling_center_vs_time',
        'gbmc_traveling_center_error',
        'gbmc_traveling_speed_error',
        'gbmc_traveling_error_vs_N',
        'gbmc_traveling_error_vs_dt',
        'gbmc_traveling_sharp_layer',
    ],
    'task3': [
        'cole_hopf_error_vs_domain',
        'cole_hopf_deterministic_transform_error',
        'cole_hopf_particle_vs_deterministic_phi',
        'cole_hopf_error_vs_output_grid',
        'cole_hopf_plateau_decomposition',
    ],
    'task4': [
        'heat_bias_spread_total_vs_N',
        'heat_spread_vs_N',
        'heat_error_vs_output_grid',
        'heat_profiles_selected_N',
        'heat_runtime_vs_N',
        'heat_error_vs_dt',
    ],
    'task5': [
        'fhn_profile_error_vs_N',
        'fhn_front_error_vs_N',
        'fhn_error_vs_dt',
        'fhn_front_center_vs_time',
        'fhn_front_speed_error',
        'fhn_aligned_profile_error',
        'fhn_profiles_selected_N',
        'fhn_runtime_vs_N',
    ],
}

TASK_DIRS = {
    'task1': os.path.join(BASE, 'gbmc_dt_bias'),
    'task2': os.path.join(BASE, 'gbmc_traveling_shock'),
    'task3': os.path.join(BASE, 'cole_hopf_plateau'),
    'task4': os.path.join(BASE, 'heat_extended'),
    'task5': os.path.join(BASE, 'fhn_extended'),
}


def _load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _check_figure(task_key, fig_name):
    d = TASK_DIRS.get(task_key, BASE)
    exists_png = os.path.isfile(os.path.join(d, fig_name + '.png'))
    exists_pdf = os.path.isfile(os.path.join(d, fig_name + '.pdf'))
    return exists_png, exists_pdf


def generate_manifest(results=None, timings=None, tasks_run=None):
    os.makedirs(OUT, exist_ok=True)

    # Load all summaries from disk (so this can run standalone)
    t1 = None
    t1_jsons = sorted(glob.glob(os.path.join(TASK_DIRS['task1'], 'dt_bias_summary_N*.json')))
    if t1_jsons:
        t1 = _load_json_safe(t1_jsons[-1])  # use last (largest N)

    t2 = _load_json_safe(os.path.join(TASK_DIRS['task2'], 'summary.json'))
    t3 = _load_json_safe(os.path.join(TASK_DIRS['task3'], 'plateau_decomposition.json'))
    t4 = _load_json_safe(os.path.join(TASK_DIRS['task4'], 'summary.json'))
    t5 = _load_json_safe(os.path.join(TASK_DIRS['task5'], 'summary.json'))

    # ---- Figure manifest ----
    fig_rows = []
    for task_key, figs in EXPECTED_FIGURES.items():
        for fig in figs:
            png_ok, pdf_ok = _check_figure(task_key, fig)
            fig_rows.append({
                'task': task_key,
                'figure': fig,
                'png_exists': png_ok,
                'pdf_exists': pdf_ok,
                'both_exist': png_ok and pdf_ok,
            })

    fig_csv = os.path.join(OUT, 'figure_manifest.csv')
    with open(fig_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['task', 'figure', 'png_exists', 'pdf_exists', 'both_exist'])
        w.writeheader(); w.writerows(fig_rows)

    n_figs_ok = sum(1 for r in fig_rows if r['both_exist'])
    n_figs_total = len(fig_rows)
    missing_figs = [r['figure'] for r in fig_rows if not r['both_exist']]

    # ---- Results manifest ----
    manifest = {
        'figures': {'total': n_figs_total, 'complete': n_figs_ok, 'missing': missing_figs},
        'timings_s': timings or {},
        'task1_gbmc_dt_bias': _extract_t1(t1),
        'task2_traveling_shock': _extract_t2(t2),
        'task3_cole_hopf_plateau': _extract_t3(t3),
        'task4_heat_extended': _extract_t4(t4),
        'task5_fhn_extended': _extract_t5(t5),
    }

    with open(os.path.join(OUT, 'results_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # ---- Results CSV ----
    csv_rows = []
    for task, key, val_dict in [
        ('task1', 'GBMC dt-bias', manifest['task1_gbmc_dt_bias']),
        ('task2', 'Traveling shock', manifest['task2_traveling_shock']),
        ('task3', 'Cole-Hopf plateau', manifest['task3_cole_hopf_plateau']),
        ('task4', 'Heat extended', manifest['task4_heat_extended']),
        ('task5', 'FHN extended', manifest['task5_fhn_extended']),
    ]:
        for metric, value in (val_dict or {}).items():
            csv_rows.append({'task': task, 'study': key, 'metric': metric, 'value': value})

    results_csv = os.path.join(OUT, 'results_manifest.csv')
    with open(results_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['task', 'study', 'metric', 'value'])
        w.writeheader(); w.writerows(csv_rows)

    # ---- Table manifest ----
    # Collects fitted-rate rows for inclusion in paper tables
    table_rows = [
        {'table': 'convergence_rates', 'task': 'T1_GBMC_dt_bias',
         'metric': 'E_bias_vs_dt_slope',
         'value': (manifest['task1_gbmc_dt_bias'] or {}).get('bias_slope_vs_dt', ''),
         'ci_lo': (manifest['task1_gbmc_dt_bias'] or {}).get('bias_slope_ci_lo', ''),
         'ci_hi': (manifest['task1_gbmc_dt_bias'] or {}).get('bias_slope_ci_hi', ''),
         'note': 'E_bias ~ dt^slope; CI from polyfit residuals'},
        {'table': 'convergence_rates', 'task': 'T2_traveling_shock',
         'metric': 'L2_vs_N_slope',
         'value': (manifest['task2_traveling_shock'] or {}).get('N_slope', ''),
         'ci_lo': (manifest['task2_traveling_shock'] or {}).get('N_slope_ci_lo', ''),
         'ci_hi': (manifest['task2_traveling_shock'] or {}).get('N_slope_ci_hi', ''),
         'note': 'L2 at T=1 ~ N^slope'},
        {'table': 'convergence_rates', 'task': 'T4_heat_extended',
         'metric': 'E_total_vs_N_slope',
         'value': (manifest['task4_heat_extended'] or {}).get('total_slope', ''),
         'ci_lo': (manifest['task4_heat_extended'] or {}).get('total_ci_lo', ''),
         'ci_hi': (manifest['task4_heat_extended'] or {}).get('total_ci_hi', ''),
         'note': 'E_total ~ N^slope; bias dominated by spread'},
        {'table': 'convergence_rates', 'task': 'T4_heat_extended',
         'metric': 'E_spread_vs_N_slope',
         'value': (manifest['task4_heat_extended'] or {}).get('spread_slope', ''),
         'ci_lo': (manifest['task4_heat_extended'] or {}).get('spread_ci_lo', ''),
         'ci_hi': (manifest['task4_heat_extended'] or {}).get('spread_ci_hi', ''),
         'note': 'Spread ~ N^slope (dominates total)'},
        {'table': 'convergence_rates', 'task': 'T5_fhn_extended',
         'metric': 'profile_L2_vs_N_slope',
         'value': (manifest['task5_fhn_extended'] or {}).get('profile_l2_slope', ''),
         'ci_lo': (manifest['task5_fhn_extended'] or {}).get('profile_l2_ci_lo', ''),
         'ci_hi': (manifest['task5_fhn_extended'] or {}).get('profile_l2_ci_hi', ''),
         'note': 'Profile L2 ~ N^slope'},
        {'table': 'convergence_rates', 'task': 'T5_fhn_extended',
         'metric': 'center_err_vs_N_slope',
         'value': (manifest['task5_fhn_extended'] or {}).get('center_err_slope', ''),
         'ci_lo': (manifest['task5_fhn_extended'] or {}).get('center_err_ci_lo', ''),
         'ci_hi': (manifest['task5_fhn_extended'] or {}).get('center_err_ci_hi', ''),
         'note': 'Front center error ~ N^slope'},
        {'table': 'negative_findings', 'task': 'T1_GBMC_dt_bias',
         'metric': 'bias_below_spread',
         'value': (manifest['task1_gbmc_dt_bias'] or {}).get('bias_below_spread', ''),
         'ci_lo': '', 'ci_hi': '',
         'note': 'E_bias < 2*E_spread at all dt; dt-bias sub-dominant'},
        {'table': 'negative_findings', 'task': 'T3_cole_hopf_plateau',
         'metric': 'plateau_primary_cause',
         'value': (manifest['task3_cole_hopf_plateau'] or {}).get('primary_cause', ''),
         'ci_lo': '', 'ci_hi': '',
         'note': 'Cole-Hopf L2 plateau: does not converge with N'},
        {'table': 'negative_findings', 'task': 'T4_heat_extended',
         'metric': 'E_bias_flat',
         'value': (manifest['task4_heat_extended'] or {}).get('bias_slope', ''),
         'ci_lo': (manifest['task4_heat_extended'] or {}).get('bias_ci_lo', ''),
         'ci_hi': (manifest['task4_heat_extended'] or {}).get('bias_ci_hi', ''),
         'note': 'Bias flat; total convergence driven entirely by spread'},
    ]

    table_csv = os.path.join(OUT, 'table_manifest.csv')
    with open(table_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['table', 'task', 'metric', 'value', 'ci_lo', 'ci_hi', 'note'])
        w.writeheader(); w.writerows(table_rows)

    # ---- Write FINAL_TESTING_SUMMARY.md ----
    _write_summary_md(manifest, missing_figs, timings or {})

    print(f"\n  Manifest written to {OUT}/")
    print(f"  Figures: {n_figs_ok}/{n_figs_total} complete")
    if missing_figs:
        print(f"  Missing figures: {missing_figs}")
    return manifest


def _fmt(x, d=4):
    if x is None or (isinstance(x, float) and (x != x)):  # nan
        return 'n/a'
    try:
        return f'{float(x):.{d}f}'
    except Exception:
        return str(x)


def _extract_t1(data):
    if not data:
        return {'status': 'not_run'}
    fit = data.get('fit', {})
    return {
        'status': 'complete',
        'N': fit.get('N'),
        'S': fit.get('S'),
        'bias_slope': fit.get('bias_slope'),
        'bias_slope_ci_lo': fit.get('bias_slope_ci_lo'),
        'bias_slope_ci_hi': fit.get('bias_slope_ci_hi'),
        'conclusion': fit.get('conclusion', 'not_available'),
    }


def _extract_t2(data):
    if not data:
        return {'status': 'not_run'}
    stop = data.get('stop_triggered', False)
    N_slope = data.get('N_slope_T_max')
    ci = data.get('N_slope_ci', [None, None])
    return {
        'status': 'stopped' if stop else 'complete',
        'stop_reasons': data.get('stop_reasons', []),
        'N_slope': N_slope,
        'N_slope_ci_lo': ci[0],
        'N_slope_ci_hi': ci[1],
        'params': data.get('params', {}),
    }


def _extract_t3(data):
    if not data:
        return {'status': 'not_run'}
    return {
        'status': 'complete',
        'plateau_l2_observed': data.get('plateau_l2_observed'),
        'primary_cause': data.get('primary_cause'),
        'differentiation_error': data.get('transform_differentiation', {}).get('error_exact_phi_num_grad'),
        'differentiation_conclusion': data.get('transform_differentiation', {}).get('conclusion'),
        'boundary_mismatch_conclusion': data.get('boundary_mismatch', {}).get('conclusion'),
    }


def _extract_t4(data):
    if not data:
        return {'status': 'not_run'}
    fit = data.get('fit', {})
    return {
        'status': 'complete',
        'total_slope': fit.get('total_slope'),
        'total_ci_lo': fit.get('total_ci', [None, None])[0],
        'total_ci_hi': fit.get('total_ci', [None, None])[1],
        'bias_slope': fit.get('bias_slope'),
        'spread_slope': fit.get('spread_slope'),
        'no_dt_study_reason': data.get('no_dt_study_reason', ''),
        'N_seq': data.get('params', {}).get('N_seq', []),
        'S': data.get('params', {}).get('S'),
    }


def _extract_t5(data):
    if not data:
        return {'status': 'not_run'}
    fit = data.get('fit', {})
    return {
        'status': 'complete',
        'profile_l2_slope': fit.get('profile_l2_slope'),
        'profile_l2_ci': fit.get('profile_l2_ci', [None, None]),
        'center_err_slope': fit.get('center_err_slope'),
        'center_err_ci': fit.get('center_err_ci', [None, None]),
        'speed_err_slope': fit.get('speed_err_slope'),
        'aligned_err_slope': fit.get('aligned_err_slope'),
        'N_seq': data.get('params', {}).get('N_seq', []),
        'S': data.get('params', {}).get('S'),
    }


def _write_summary_md(manifest, missing_figs, timings):
    t1 = manifest.get('task1_gbmc_dt_bias', {})
    t2 = manifest.get('task2_traveling_shock', {})
    t3 = manifest.get('task3_cole_hopf_plateau', {})
    t4 = manifest.get('task4_heat_extended', {})
    t5 = manifest.get('task5_fhn_extended', {})
    figs = manifest.get('figures', {})

    total_time = sum(timings.values()) if timings else 0

    lines = [
        "# Final Pre-Publication Testing Summary",
        "",
        "## Overview",
        f"- **Total figures generated:** {figs.get('complete', '?')}/{figs.get('total', '?')}",
        f"- **Total wall time:** {total_time:.1f}s",
        "",
    ]

    if missing_figs:
        lines += ["### Missing Figures", ""]
        for f in missing_figs:
            lines.append(f"- `{f}` (PNG or PDF missing)")
        lines.append("")

    # ---- Task 1 ----
    lines += [
        "---",
        "## Task 1: GBMC Time-Step Bias (High-N)",
        "",
        f"- **Status:** {t1.get('status', 'unknown')}",
        f"- **N:** {t1.get('N', '?')}   **Seeds:** {t1.get('S', '?')}",
        f"- **Bias slope vs dt:** {_fmt(t1.get('bias_slope'))}",
        f"- **95% CI:** [{_fmt(t1.get('bias_slope_ci_lo'))}, {_fmt(t1.get('bias_slope_ci_hi'))}]",
        f"- **Conclusion:** {t1.get('conclusion', 'not_available')}",
        "",
        "> **Interpretation:** If E_bias remains below E_spread at all dt, the Lie-splitting",
        "> bias is overwhelmed by particle noise at this N and S; a larger N or S is required.",
        "> If a clean slope emerges with CI > 0, the Lie-splitting bias is measurable.",
        "",
    ]

    # ---- Task 2 ----
    lines += [
        "---",
        "## Task 2: Traveling-Shock GBMC Validation",
        "",
        f"- **Status:** {t2.get('status', 'unknown')}",
    ]
    if t2.get('stop_reasons'):
        for r in t2.get('stop_reasons', []):
            lines.append(f"- **STOP reason:** {r}")
    lines += [
        f"- **N-refinement slope (T=1.0):** {_fmt(t2.get('N_slope'))}",
        f"- **95% CI:** [{_fmt(t2.get('N_slope_ci_lo'))}, {_fmt(t2.get('N_slope_ci_hi'))}]",
        "",
        "> **Interpretation:** A slope near -0.5 with CI below -0.25 would confirm O(N^{-1/2})",
        "> convergence for the traveling shock, matching the stationary-shock result.",
        "",
    ]

    # ---- Task 3 ----
    lines += [
        "---",
        "## Task 3: Cole-Hopf Error Plateau Diagnosis",
        "",
        f"- **Status:** {t3.get('status', 'unknown')}",
        f"- **Observed plateau L2:** {_fmt(t3.get('plateau_l2_observed'))}",
        f"- **Primary cause:** {t3.get('primary_cause', 'not_determined')}",
        f"- **Differentiation error (exact phi):** {_fmt(t3.get('differentiation_error'))}",
        f"- **Differentiation conclusion:** {t3.get('differentiation_conclusion', 'n/a')}",
        f"- **Boundary mismatch conclusion:** {t3.get('boundary_mismatch_conclusion', 'n/a')}",
        "",
        "> **Negative finding (previously established):** The Cole-Hopf GRW has a",
        "> systematic L2 plateau at ~0.40. This study decomposes the source.",
        "> The R²=0.54, slope CI touching zero confirms no clean convergence.",
        "",
    ]

    # ---- Task 4 ----
    lines += [
        "---",
        "## Task 4: Extended Heat Convergence",
        "",
        f"- **Status:** {t4.get('status', 'unknown')}",
        f"- **N sequence:** {t4.get('N_seq', '?')}   **Seeds:** {t4.get('S', '?')}",
        f"- **Total error slope:** {_fmt(t4.get('total_slope'))}",
        f"- **95% CI:** [{_fmt(t4.get('total_ci_lo'))}, {_fmt(t4.get('total_ci_hi'))}]",
        f"- **Bias slope:** {_fmt(t4.get('bias_slope'))}",
        f"- **Spread slope:** {_fmt(t4.get('spread_slope'))}",
        "",
        "> **No dt study:** Heat GRW is exact in time (Brownian increments are exact).",
        f"> {t4.get('no_dt_study_reason', '')}",
        "",
        "> **Interpretation:** E_spread should converge at ~O(N^{-1/2}).",
        "> A persistent E_bias indicates systematic reconstruction error",
        "> (histogram binning at the shock discontinuity).",
        "",
    ]

    # ---- Task 5 ----
    ci5 = t5.get('profile_l2_ci', [None, None])
    lines += [
        "---",
        "## Task 5: Extended FHN Convergence",
        "",
        f"- **Status:** {t5.get('status', 'unknown')}",
        f"- **N sequence:** {t5.get('N_seq', '?')}   **Seeds:** {t5.get('S', '?')}",
        f"- **Profile L2 slope:** {_fmt(t5.get('profile_l2_slope'))}",
        f"- **Profile L2 95% CI:** [{_fmt(ci5[0] if ci5 else None)}, "
            f"{_fmt(ci5[1] if ci5 else None)}]",
        f"- **Front center error slope:** {_fmt(t5.get('center_err_slope'))}",
        f"- **Front speed error slope:** {_fmt(t5.get('speed_err_slope'))}",
        f"- **Aligned-profile error slope:** {_fmt(t5.get('aligned_err_slope'))}",
        "",
        "> **Interpretation:** The aligned-profile error isolates shape convergence",
        "> from front-position error. A clean slope near -0.5 for the aligned error",
        "> indicates the profile shape converges at the particle rate, while",
        "> center error may converge faster (-1 to -2) due to averaging effects.",
        "",
    ]

    # ---- Figure inventory ----
    lines += [
        "---",
        "## Figure Inventory",
        "",
        "| Task | Figure | PNG | PDF |",
        "|------|--------|-----|-----|",
    ]
    # Re-check
    for task_key, figs_list in EXPECTED_FIGURES.items():
        for fig_name in figs_list:
            png_ok, pdf_ok = _check_figure(task_key, fig_name)
            lines.append(
                f"| {task_key} | `{fig_name}` | {'✓' if png_ok else '✗'} | {'✓' if pdf_ok else '✗'} |"
            )
    lines.append("")

    # ---- Key decisions for paper ----
    lines += [
        "---",
        "## Key Decisions for Paper Writing",
        "",
        "1. **GBMC convergence rate:** Report N-slope with 95% CI. If CI straddles -0.5,",
        "   state `O(N^{-1/2})` is consistent but not proven.",
        "",
        "2. **GBMC dt-bias:** If E_bias < E_spread at all dt, report as",
        "   'bias below estimation precision; Lie-splitting bias not measurable at N=6400'.",
        "   This is an honest negative finding, not a failure.",
        "",
        "3. **Cole-Hopf plateau:** Report as confirmed negative finding with primary cause.",
        "   Do not claim convergence for Cole-Hopf GRW.",
        "",
        "4. **Heat GRW:** Report spread slope; document why no dt study is needed.",
        "   If bias is non-negligible at small N, attribute to histogram binning at shock.",
        "",
        "5. **FHN:** Report both profile and aligned-profile slopes. If they differ,",
        "   the discrepancy is due to front-position variance (report both rates).",
        "",
        "6. **Traveling shock:** If stop_triggered=True, report the physical reason",
        "   (subcharacteristic violation, mass loss, etc.) as a finding.",
        "",
        "---",
        "_This summary was auto-generated by `generate_final_manifest.py`._",
    ]

    md_path = os.path.join(OUT, 'FINAL_TESTING_SUMMARY.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Summary written: {md_path}")


if __name__ == '__main__':
    generate_manifest()
