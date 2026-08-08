"""Regenerate the 11 paper figures, title-less, from the checked-in study data.

This script never reruns an ensemble study and never recomputes a seed. It reads
only the canonical summaries/CSVs/JSONs/arrays under
``output/final_prepublication_tests/`` and reproduces the plot commands from the
in-repo study scripts with every ``set_title``/``suptitle`` removed, matching the
title-less figures the paper compiles against (``8-8-26/figuresv3/``).

Run directly (``python figure_scripts/regenerate_paper_figures.py``) or via
``python reproduce.py figures``. Output PDFs are written to
``output/final_prepublication_tests/paper_figures/``.

Cosmetic-only deviations from the raw study recipes (no data changes):
  * fitted-viscosity figure uses the paper's ``\hat\nu`` label (as the caption does);
  * cole_hopf left-panel exact-phi bar is annotated ``7x10^-4``;
  * bias-spread-total figure drops the per-curve slope labels and the
    ``N^{-1/2}`` reference line and uses an ``Error (L^2)`` axis, matching the
    Paper-1 visual exemplar (the slopes/reference appear on the spread-vs-N figure).

Figure -> canonical data source:
  cole_hopf_plateau_decomposition        cole_hopf_plateau/{plateau_decomposition,study_B_deterministic}.json  (t3)
  heat_bias_spread_total_vs_N            heat_extended/summary_by_N.csv                                        (t4)
  fhn_profile_error_vs_N                 fhn_extended/summary_by_N.csv                                         (t5)
  fhn_front_error_vs_N                   fhn_extended/summary_by_N.csv                                         (t5)
  production_gbmc_spread_vs_N            gbmc_production_n_refinement/{per_N_summary.csv,rates.json}           (t6)
  production_gbmc_bias_spread_total_vs_N gbmc_production_n_refinement/{per_N_summary.csv,rates.json}           (t6)
  production_gbmc_profiles_selected_N    gbmc_production_n_refinement/production_profiles.npz                  (t6)
  production_gbmc_fitted_viscosity_vs_N  gbmc_production_n_refinement/per_N_summary.csv                        (t6)
  gbmc_bias_vs_dt                        gbmc_dt_bias/dt_bias_summary_N6400.json                               (t1)
  gbmc_traveling_error_vs_N              gbmc_traveling_shock/summary.json                                     (t2)
  gbmc_traveling_center_vs_time          gbmc_traveling_shock/per_run.csv (single seed 42, largest N)         (t2)
"""
import csv
import json
import os

# Fixed date for matplotlib PDF metadata so regenerated figures are byte-identical
# across runs (matplotlib honors SOURCE_DATE_EPOCH).
os.environ.setdefault('SOURCE_DATE_EPOCH', '1704067200')  # 2024-01-01 UTC

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'output', 'final_prepublication_tests')
OUT = os.path.join(DATA, 'paper_figures')

NU = 0.5  # prescribed viscosity for the stationary/traveling shock studies


def _csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _json(path):
    with open(path) as f:
        return json.load(f)


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name + '.pdf')


# --------------------------------------------------------------------------- #
# t3  cole_hopf_plateau_decomposition
# --------------------------------------------------------------------------- #
def cole_hopf_plateau_decomposition():
    dec = _json(os.path.join(DATA, 'cole_hopf_plateau', 'plateau_decomposition.json'))
    sB = _json(os.path.join(DATA, 'cole_hopf_plateau', 'study_B_deterministic.json'))
    err_numgrad = dec['transform_differentiation']['error_exact_phi_num_grad']
    err_noise_1pct = sB['perturbed_phi_noise_1e-02']
    plateau_l2 = dec['plateau_l2_observed']
    domain_l2 = {float(k): v for k, v in dec['boundary_mismatch']['error_by_domain'].items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sources = ['Diff. only\n(exact phi)', '1% phi noise', 'Observed\nplateau']
    values = [max(err_numgrad, 0), max(err_noise_1pct, 0), plateau_l2]
    colors = ['steelblue', 'orange', 'red']
    bars = axes[0].bar(sources, values, color=colors, alpha=0.8)
    axes[0].set_ylabel('L2 error')
    for i, (bar, val) in enumerate(zip(bars, values)):
        txt = r'$7\times10^{-4}$' if i == 0 else f'{val:.3f}'
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     txt, ha='center', fontsize=9)
    L_vals = sorted(domain_l2.keys())
    axes[1].plot(L_vals, [domain_l2[L] for L in L_vals], 'gs-', lw=1.5, ms=6)
    axes[1].axhline(plateau_l2, color='r', ls='--', label=f'Plateau ({plateau_l2})')
    axes[1].set_xlabel('Domain size L')
    axes[1].set_ylabel('L2 error')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, 'cole_hopf_plateau_decomposition')


# --------------------------------------------------------------------------- #
# t4  heat_bias_spread_total_vs_N
# --------------------------------------------------------------------------- #
def heat_bias_spread_total_vs_N():
    hr = _csv(os.path.join(DATA, 'heat_extended', 'summary_by_N.csv'))
    Nh = np.array([float(r['N']) for r in hr])
    Eb = np.array([float(r['E_bias']) for r in hr])
    Es = np.array([float(r['E_spread']) for r in hr])
    Et = np.array([float(r['E_total']) for r in hr])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(Nh, Eb, 'bs-', lw=1.8, ms=6, label=r'$E_\mathrm{bias}$')
    ax.loglog(Nh, Es, 'ro-', lw=1.8, ms=6, label=r'$E_\mathrm{spread}$')
    ax.loglog(Nh, Et, 'g^-', lw=1.8, ms=6, label=r'$E_\mathrm{total}$')
    N_ref = np.array([Nh.min(), Nh.max()])
    c0 = Et[0] * Nh[0]**0.5
    ax.loglog(N_ref, c0 * N_ref**(-0.5), 'k--', lw=1.2, label=r'$O(N^{-1/2})$')
    ax.set_xlabel('N')
    ax.set_ylabel('Error (L2)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    _save(fig, 'heat_bias_spread_total_vs_N')


# --------------------------------------------------------------------------- #
# t5  fhn_profile_error_vs_N  and  fhn_front_error_vs_N
# --------------------------------------------------------------------------- #
def _fhn_arrays():
    fr = _csv(os.path.join(DATA, 'fhn_extended', 'summary_by_N.csv'))
    Nf = np.array([float(r['N']) for r in fr])
    l2f = np.array([float(r['l2_mean']) for r in fr])
    cef = np.array([float(r['ce_mean']) for r in fr])
    spf = np.array([float(r['speed_err_mean']) for r in fr])
    return Nf, l2f, cef, spf


def fhn_profile_error_vs_N():
    Nf, l2f, _, _ = _fhn_arrays()
    Nrl = np.array([Nf.min(), Nf.max()])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(Nf, l2f, 'bo-', lw=1.8, ms=6)
    c0 = l2f[0] * Nf[0]**0.5
    ax.loglog(Nrl, c0 * Nrl**(-0.5), 'k--', lw=1.2, label=r'$O(N^{-1/2})$')
    ax.set_xlabel('N')
    ax.set_ylabel('L2 (vs reference)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    _save(fig, 'fhn_profile_error_vs_N')


def fhn_front_error_vs_N():
    Nf, _, cef, spf = _fhn_arrays()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].loglog(Nf, cef, 'rs-', lw=1.8, ms=6)
    axes[0].set_xlabel('N')
    axes[0].set_ylabel('|center error|')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[1].loglog(Nf, spf, 'g^-', lw=1.8, ms=6)
    axes[1].set_xlabel('N')
    axes[1].set_ylabel('|speed error|')
    axes[1].grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'fhn_front_error_vs_N')


# --------------------------------------------------------------------------- #
# t6  production stationary-shock figures
# --------------------------------------------------------------------------- #
def _t6():
    rows = _csv(os.path.join(DATA, 'gbmc_production_n_refinement', 'per_N_summary.csv'))
    rates = _json(os.path.join(DATA, 'gbmc_production_n_refinement', 'rates.json'))
    N = np.array([float(r['N']) for r in rows])
    return rows, rates, N


def production_gbmc_spread_vs_N():
    rows, rates, N = _t6()
    spr = np.array([float(r['E_spread']) for r in rows])
    coeffs = np.polyfit(np.log10(N), np.log10(spr), 1)
    spr_slope = float(coeffs[0])
    ci_lo, ci_hi = rates['spread_ci_lo'], rates['spread_ci_hi']
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N, spr, 'o-', color='steelblue', lw=2, ms=7,
              label=f'$E_{{spread}}$  slope={spr_slope:.3f}')
    xfit = np.array([N.min(), N.max()])
    ax.loglog(xfit, 10**np.polyval(coeffs, np.log10(xfit)), '--', color='steelblue',
              alpha=0.6, label=f'fit  CI=[{ci_lo:.3f},{ci_hi:.3f}]')
    i_mid = len(N) // 2
    ax.loglog(xfit, spr[i_mid] * (xfit / N[i_mid])**(-0.5), 'k:', lw=1.2,
              label=r'$N^{-1/2}$ reference')
    ax.set_xlabel('N (particles)')
    ax.set_ylabel(r'$E_{\mathrm{spread}}$')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    _save(fig, 'production_gbmc_spread_vs_N')


def production_gbmc_bias_spread_total_vs_N():
    # Paper-figure style (the Paper-1 visual exemplar): plain E_bias/E_spread/E_total
    # legend without per-curve slope annotations or an N^{-1/2} reference line (the
    # slopes and reference appear on the dedicated spread-vs-N figure), and an
    # "Error (L^2)" axis. The study's own diagnostic copy keeps the slope labels.
    rows, _, N = _t6()
    bias = np.array([float(r['E_bias']) for r in rows])
    spr = np.array([float(r['E_spread']) for r in rows])
    tot = np.array([float(r['E_total']) for r in rows])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N, tot, 's-', color='black', lw=2, ms=7, label=r'$E_\mathrm{total}$')
    ax.loglog(N, spr, 'o-', color='steelblue', lw=2, ms=7, label=r'$E_\mathrm{spread}$')
    ax.loglog(N, bias, '^-', color='firebrick', lw=2, ms=7, label=r'$E_\mathrm{bias}$')
    ax.set_xlabel('N')
    ax.set_ylabel(r'Error ($L^2$)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    _save(fig, 'production_gbmc_bias_spread_total_vs_N')


def production_gbmc_profiles_selected_N():
    npz = np.load(os.path.join(DATA, 'gbmc_production_n_refinement', 'production_profiles.npz'))
    x = npz['x']
    u_exact = npz['u_exact']
    N_plot = [N for N in [100, 400, 1600, 6400] if f'N{N}_mean' in npz]
    colors_ = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_plot)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, u_exact, 'k-', lw=2, label='Exact', zorder=10)
    for col, N in zip(colors_, N_plot):
        um = npz[f'N{N}_mean']
        us = npz[f'N{N}_std']
        ax.plot(x, um, '-', color=col, lw=1.5, label=f'N={N}', zorder=5)
        ax.fill_between(x, um - us, um + us, color=col, alpha=0.15)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, 'production_gbmc_profiles_selected_N')


def production_gbmc_fitted_viscosity_vs_N():
    rows, _, N = _t6()
    nu_means = np.array([float(r['nu_mean']) for r in rows])
    nu_stds = np.array([float(r['nu_std']) for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(N, nu_means, yerr=nu_stds, fmt='o-', color='purple', lw=2, ms=7,
                capsize=4, label=r'$\hat\nu$ (mean $\pm$ std)')
    ax.axhline(NU, color='k', ls='--', lw=1.5, label=r'Exact $\nu=0.5$')
    ax.set_xscale('log')
    ax.set_xlabel('N')
    ax.set_ylabel(r'$\hat\nu$')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    _save(fig, 'production_gbmc_fitted_viscosity_vs_N')


# --------------------------------------------------------------------------- #
# t1  gbmc_bias_vs_dt
# --------------------------------------------------------------------------- #
def gbmc_bias_vs_dt():
    t1 = _json(os.path.join(DATA, 'gbmc_dt_bias', 'dt_bias_summary_N6400.json'))
    per_dt = sorted(t1['per_dt'], key=lambda r: r['dt'])
    dt_arr = np.array([r['dt'] for r in per_dt])
    bias_arr = np.array([r['E_bias'] for r in per_dt])
    spr_arr = np.array([r['E_spread'] for r in per_dt])
    tot_arr = np.array([r['E_total'] for r in per_dt])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(dt_arr, bias_arr, 'bs-', lw=1.8, ms=6, label=r'$E_\mathrm{bias}$')
    ax.loglog(dt_arr, spr_arr, 'ro-', lw=1.8, ms=6, label=r'$E_\mathrm{spread}$')
    ax.loglog(dt_arr, tot_arr, 'g^-', lw=1.8, ms=6, label=r'$E_\mathrm{total}$')
    dt_ref = np.array([dt_arr.min(), dt_arr.max()])
    for exp, ls, lbl in [(1.0, '--', r'$O(\Delta t)$'), (0.5, ':', r'$O(\Delta t^{1/2})$')]:
        c0 = max(bias_arr) * (dt_arr.min() ** (-exp))
        ax.loglog(dt_ref, c0 * dt_ref**exp, ls, color='gray', lw=1.0, label=lbl)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel('Error (L2)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    _save(fig, 'gbmc_bias_vs_dt')


# --------------------------------------------------------------------------- #
# t2  gbmc_traveling_error_vs_N  and  gbmc_traveling_center_vs_time
# --------------------------------------------------------------------------- #
def gbmc_traveling_error_vs_N():
    t2 = _json(os.path.join(DATA, 'gbmc_traveling_shock', 'summary.json'))
    T_key = t2['params']['output_times'][-1]
    nr = sorted([r for r in t2['N_refinement'] if abs(r['t_out'] - T_key) < 1e-9],
                key=lambda r: r['N'])
    n_arr = np.array([float(r['N']) for r in nr])
    l2_arr = np.array([float(r['l2_mean']) for r in nr])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(n_arr, l2_arr, 'bo-', lw=1.8, ms=6, label=f'L2 (T={T_key})')
    n_ref = np.array([n_arr.min(), n_arr.max()])
    c0 = l2_arr[0] * n_arr[0]**0.5
    ax.loglog(n_ref, c0 * n_ref**(-0.5), 'k--', lw=1, label=r'$O(N^{-1/2})$')
    ax.set_xlabel('N')
    ax.set_ylabel('L2 error')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    _save(fig, 'gbmc_traveling_error_vs_N')


def gbmc_traveling_center_vs_time():
    pr = _csv(os.path.join(DATA, 'gbmc_traveling_shock', 'per_run.csv'))
    N_max = int(max(float(r['N']) for r in pr))
    trk = sorted([r for r in pr if int(float(r['N'])) == N_max and int(float(r['seed'])) == 42],
                 key=lambda r: float(r['t_out']))
    t_track = [float(r['t_out']) for r in trk]
    xc_num = [float(r['xc_num']) for r in trk]
    xc_ex = [float(r['xc_exact']) for r in trk]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(t_track, xc_ex, 'k-', lw=2, label='Exact center')
    ax1.plot(t_track, xc_num, 'bo-', lw=1.5, ms=6, label='GBMC center')
    ax1.set_xlabel('t')
    ax1.set_ylabel('Shock center')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(t_track, [abs(n - e) for n, e in zip(xc_num, xc_ex)], 'ro-', lw=1.5, ms=6)
    ax2.set_xlabel('t')
    ax2.set_ylabel('|center error|')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, 'gbmc_traveling_center_vs_time')


FIGURES = [
    cole_hopf_plateau_decomposition,
    heat_bias_spread_total_vs_N,
    fhn_profile_error_vs_N,
    fhn_front_error_vs_N,
    production_gbmc_spread_vs_N,
    production_gbmc_bias_spread_total_vs_N,
    production_gbmc_profiles_selected_N,
    production_gbmc_fitted_viscosity_vs_N,
    gbmc_bias_vs_dt,
    gbmc_traveling_error_vs_N,
    gbmc_traveling_center_vs_time,
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for fn in FIGURES:
        fn()
    print(f'\nDONE. {len(FIGURES)} title-less paper figures in {OUT}')


if __name__ == '__main__':
    main()
