#!/usr/bin/env python3
"""Verification and benchmark comparison for the GRW solver suite.

For each equation the script runs the primary solver and compares output against
a trusted benchmark:

  heat    -- exact analytical solution (error function), valid for step IC.
             Primary solver: direct GRW.
  burgers -- Cole-Hopf GRW only, vs exact stationary-shock or traveling-wave
             solution.  Supported ICs: stationary_shock, traveling_wave.
  fhn     -- scalar GRW vs exact traveling-wave solution
             u = 1/(1+exp(-(x+theta*t)/2)), multi-time snapshots.

Burgers and FHN comparisons label exact vs reference solutions explicitly.
The error metrics quantify GRW feasibility and limitations, not accuracy claims.
Outputs per equation (default: output/verify/<equation>): comparison_plot.png,
metrics.json, comparison_data.npz (with --save-data).  See --help for options.
"""

import argparse
import json
import os
import sys
from math import erf as _erf_scalar

import numpy as np
import matplotlib

if sys.platform.startswith("darwin"):
    matplotlib.use("MacOSX")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg_module
from config import SimulationConfig
from simulation import (
    simulate_heat_equation,
    simulate_burgers,
    simulate_fitzhugh_nagumo,
)


# Default config paths (relative to repo root)

_DEFAULT_CONFIGS = {
    "heat": "configs/heat_step_dirichlet.json",
    "burgers": "configs/burgers_stationary_shock.json",
    "burgers_rbmc": "configs/burgers_relaxation_gbmc.json",
    "fhn": "configs/fhn_grw_steady.json",
}


def _erf_vec(arr):
    """Element-wise erf using the stdlib scalar implementation (no scipy dependency)."""
    a = np.asarray(arr, dtype=float)
    return np.array([_erf_scalar(float(v)) for v in a.ravel()]).reshape(a.shape)


def exact_heat_step(x, T, x0, uL, uR, alpha):
    """
    Exact solution for the heat equation with Heaviside step IC on an infinite domain.

    u(x, T) = uL + (uR - uL) * 0.5 * (1 + erf((x - x0) / (2 * sqrt(alpha * T))))
    """
    return uL + (uR - uL) * 0.5 * (1.0 + _erf_vec((x - x0) / (2.0 * np.sqrt(alpha * T))))


def exact_burgers_traveling_wave(x, t, nu, x_center=0.0):
    """
    Exact traveling wave solution for Burgers' equation u_t + u*u_x = nu*u_xx.

    u(x, t) = 1 - 2*sqrt(nu) * tanh((x - x_center - t) / sqrt(nu))

    This is the exact infinite-domain solution for the IC
      u0(x) = 1 - 2*sqrt(nu) * tanh((x - x_center) / sqrt(nu)).
    The wave moves at unit speed c = 1.

    Derivation: inserting the ansatz f(x - ct) into Burgers' equation yields
    c = 1 (wave speed = mean value) and delta = sqrt(nu) (wave width).
    The solution is valid on an infinite domain; on a finite domain it is an
    approximation that degrades near the boundaries as the wave approaches them.

    :param x: array of spatial positions
    :param t: final time
    :param nu: kinematic viscosity
    :param x_center: initial wave center position
    :return: array of exact u values at (x, t)
    """
    return 1.0 - 2.0 * np.sqrt(nu) * np.tanh((x - x_center - t) / np.sqrt(nu))


def exact_burgers_stationary_shock(x, nu, x_center=None, amplitude=1.0):
    """
    Exact stationary-shock solution for Burgers' equation u_t + u*u_x = nu*u_xx.

    u(x, t) = -A * tanh(A * (x - x_center) / (2 * nu)) for all t >= 0

    where A = amplitude. This is an exact STATIONARY solution: the nonlinear
    advection term u*u_x and the diffusion term nu*u_xx cancel exactly for any
    amplitude A and viscosity nu. Verification:

      u_x = -A^2 / (2*nu) * sech^2(A*(x-xc)/(2*nu))
      u*u_x = A^3 * tanh(...) * sech^2(...) / (2*nu)
      u_xx = A^3 * tanh(...) * sech^2(...) / (2*nu^2)
      nu*u_xx = A^3 * tanh(...) * sech^2(...) / (2*nu) = u*u_x QED

    The Cole-Hopf phi0 for this IC satisfies phi0(0) = phi0(L) whenever xc = L/2
    and the domain is symmetric about xc, making the cumulative reconstruction in
    the GRW return exactly to its starting value -- a well-conditioned property for
    the Cole-Hopf GRW benchmark.

    :param x: array of spatial positions
    :param nu: kinematic viscosity
    :param x_center: shock centre; if None, defaults to domain midpoint
    :param amplitude: shock amplitude A (controls both wave height and width)
    :return: array of exact u values (independent of time t)
    """
    if x_center is None:
        x_center = 0.5 * (float(x.max()) + float(x.min()))
    return -amplitude * np.tanh(amplitude * (x - x_center) / (2.0 * nu))


def compute_metrics(numerical, reference, dx):
    """
    Standard error metrics between two arrays on a uniform grid with spacing dx.

    Returns a dict with: l1, l2, linf (max_abs_error), rel_l2, mean_signed, rmse.
    rel_l2 is None if the reference L2 norm is below 1e-12.
    linf and max_abs_error are the same value stored under two keys for readability.
    """
    diff = numerical - reference
    ref_norm = np.sqrt(np.sum(reference ** 2) * dx)
    linf = float(np.max(np.abs(diff)))
    return {
        "l1": float(np.sum(np.abs(diff)) * dx),
        "l2": float(np.sqrt(np.sum(diff ** 2) * dx)),
        "linf": linf,
        "max_abs_error": linf,
        "rel_l2": float(np.sqrt(np.sum(diff ** 2) * dx) / ref_norm)
                         if ref_norm > 1e-12 else None,
        "mean_signed": float(np.mean(diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def _fmt_val(v):
    """
    Adaptive formatter for console output.

    Values whose magnitude is >= 5e-4 are printed with 6 fixed decimal places.
    Smaller values are printed in 2-significant-figure scientific notation so
    they are never silently rounded to 0.000000.
    """
    if v is None:
        return "N/A"
    if v == 0.0:
        return "0.000000"
    return f"{v:.6f}" if abs(v) >= 5e-4 else f"{v:.2e}"


def _fmt_compact(v):
    """
    Compact adaptive formatter for figure titles and inline metric strings.

    Values >= 1e-3 use 4 fixed decimal places. Smaller values use 2-significant-
    figure scientific notation so tiny metrics remain readable in the figure title.
    """
    if v is None:
        return "N/A"
    if v == 0.0:
        return "0"
    return f"{v:.4f}" if abs(v) >= 1e-3 else f"{v:.2e}"


def _metrics_str(metrics):
    rel = f", relL2={_fmt_compact(metrics['rel_l2'])}" if metrics.get("rel_l2") is not None else ""
    return f"L2={_fmt_compact(metrics['l2'])}, max|err|={_fmt_compact(metrics['linf'])}{rel}"


def _print_metrics(metrics, label=""):
    if label:
        print(f"\n  {label}")
    print(f"    L1           : {_fmt_val(metrics['l1'])}")
    print(f"    L2           : {_fmt_val(metrics['l2'])}")
    print(f"    max|err|     : {_fmt_val(metrics['linf'])}")
    if metrics.get("rel_l2") is not None:
        print(f"    Relative L2  : {_fmt_val(metrics['rel_l2'])}")
    print(f"    Mean signed  : {_fmt_val(metrics['mean_signed'])}")
    print(f"    RMSE         : {_fmt_val(metrics['rmse'])}")


def plot_comparison(
    x, numerical, reference,
    title, ylabel, num_label, ref_label, metrics,
    output_path, ref_note=None, extra_curves=None,
):
    """
    Two-panel figure: left = overlay of numerical and reference, right = pointwise error.

    extra_curves: optional list of (x, y, label, linestyle) for additional left-panel
                  curves (used to show the v component in FHN plots).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(x, reference, color="black", linewidth=2, linestyle="-", label=ref_label)
    ax1.plot(x, numerical, color="crimson", linewidth=1.5, linestyle="--", label=num_label)

    if extra_curves:
        _colors = ["steelblue", "darkorange", "seagreen"]
        for i, (ex, ey, elabel, els) in enumerate(extra_curves):
            ax1.plot(ex, ey, color=_colors[i % len(_colors)],
                     linewidth=1.5, linestyle=els, label=elabel)

    ax1.set_xlabel("x")
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"{title}\n{_metrics_str(metrics)}", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    if ref_note:
        ax1.annotate(
            ref_note, xy=(0.02, 0.03), xycoords="axes fraction",
            fontsize=8, color="gray", fontstyle="italic",
        )

    error = numerical - reference
    ax2.plot(x, error, color="steelblue", linewidth=1.5)
    ax2.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("x")
    ax2.set_ylabel("error  (numerical - reference)")
    ax2.set_title("Pointwise error", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"  [verify] Saved plot    -> {output_path}")
    plt.close(fig)


def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=lambda v: None)
    print(f"  [verify] Saved metrics -> {path}")


def save_npz(x, numerical, reference, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, x=x, numerical=numerical, reference=reference,
             error=numerical - reference)
    print(f"  [verify] Saved data    -> {path}.npz")


# Heat verification  (exact analytical benchmark)

def run_heat(cfg, output_dir, do_save_data):
    print("\n" + "=" * 62)
    print("  Heat GRW -- verification vs exact error-function solution")
    print("=" * 62)

    alpha = cfg.diff_constant
    T = cfg.total_time
    L = cfg.domain_size
    N = cfg.num_points
    dt = cfg.time_step
    uL = float(cfg.boundary_conditions["LEFT"]["value"])
    uR = float(cfg.boundary_conditions["RIGHT"]["value"])

    # For a step IC, every glob starts at the same position (the jump location).
    x0 = float(cfg.initial_conditions[0][0])
    jump_height = float(uR - uL)
    exp_std = float(np.sqrt(2.0 * alpha * T))

    print(f"\n  Parameters")
    print(f"    alpha = {alpha},  T = {T},  dt = {dt}  ({int(T / dt)} steps)")
    print(f"    N = {N} globs,  domain [0, {L}]")
    print(f"    Step at x0 = {x0},  jump_height = {jump_height:.4f}")
    print(f"    Theoretical glob std: sqrt(2*alpha*T) = {exp_std:.6f}")

    print("\n  Running heat simulation ...", flush=True)
    initial_globs = [{"position": pos, "value": val}
                     for pos, val in cfg.initial_conditions]
    final_globs = simulate_heat_equation(initial_globs, cfg)
    print("  Done.")

    positions = np.array([g["position"] for g in final_globs])
    values = np.array([g["value"]    for g in final_globs])

    mean_pos = float(np.mean(positions))
    std_pos = float(np.std(positions))
    total_wt = float(np.sum(values))

    print(f"\n  Glob diagnostics")
    print(f"    Mean position  : {mean_pos:.6f}   (expected {x0:.4f} for sym. domain)")
    print(f"    Std  position  : {std_pos:.6f}   (expected {exp_std:.6f})")
    print(f"    Total weight   : {total_wt:.8f}   (expected {jump_height:.8f})")

    # Reconstruct u(x, T) by numerical integration of the glob list.
    nbins = min(400, max(200, N // 50))
    edges = np.linspace(0.0, L, nbins + 1)
    bin_w, _ = np.histogram(positions, bins=edges, weights=values)
    u_num = uL + np.cumsum(bin_w)
    x_grid = 0.5 * (edges[:-1] + edges[1:])
    dx = float(edges[1] - edges[0])

    u_exact = exact_heat_step(x_grid, T, x0, uL, uR, alpha)
    metrics = compute_metrics(u_num, u_exact, dx)

    metrics["glob_mean_position"] = mean_pos
    metrics["glob_std_position"] = std_pos
    metrics["theoretical_std"] = exp_std
    metrics["weight_conservation"] = total_wt
    metrics["expected_weight"] = jump_height
    metrics["num_globs"] = N
    metrics["alpha"] = alpha
    metrics["T"] = T

    _print_metrics(metrics, "Error metrics  (numerical vs exact error function)")

    title = (
        f"Heat GRW: numerical vs exact  (T={T}, alpha={alpha}, N={N})\n"
        f"Total weight={total_wt:.6f}, sigma_obs={std_pos:.4f}, "
        f"sigma_th={exp_std:.4f}"
    )
    plot_comparison(
        x_grid, u_num, u_exact,
        title=title,
        ylabel="u(x, T)",
        num_label=f"GRW  (N={N})",
        ref_label="Exact  (error function)",
        metrics=metrics,
        output_path=os.path.join(output_dir, "comparison_plot.png"),
    )
    save_metrics(metrics, os.path.join(output_dir, "metrics.json"))
    if do_save_data:
        save_npz(x_grid, u_num, u_exact, os.path.join(output_dir, "comparison_data"))


# Burgers verification (Cole-Hopf GRW only, exact benchmarks)

def _run_burgers_grw(cfg, diag_dir=None):
    """
    Run the active Burgers GRW solver (mode dispatched via config.burgers_mode).

    If diag_dir is provided and mode is cole_hopf_grw, saves intermediate
    diagnostic plots to diag_dir/cole_hopf_diagnostics.png.

    Returns (x_sorted, u_sorted) on a uniform or near-uniform grid.
    """
    if diag_dir is not None:
        cfg._diag_dir = diag_dir
    globs = [{"position": float(pos), "value": [float(val)]}
             for pos, val in cfg.initial_conditions]
    result = simulate_burgers(globs, cfg)
    if diag_dir is not None and hasattr(cfg, '_diag_dir'):
        del cfg._diag_dir
    x = np.array([g["position"] for g in result])
    u = np.array([g["value"][0] for g in result])
    order = np.argsort(x)
    return x[order], u[order]


def plot_burgers_cole_hopf(
    x_grid, u_num, u_ref,
    title, num_label, ref_label,
    metrics, output_path,
    ref_is_exact=True,
):
    """
    Clean two-panel Burgers Cole-Hopf verification figure.

    Layout mirrors the FHN verification style:
      Left panel: GRW reconstruction vs reference, clean legend, metrics subtitle.
      Right panel: pointwise error with zero line.

    :param x_grid: 1d array, spatial grid
    :param u_num: 1d array, GRW reconstruction
    :param u_ref: 1d array, reference solution
    :param title: str, figure suptitle
    :param num_label: str, legend label for GRW curve
    :param ref_label: str, legend label for reference curve
    :param metrics: dict from compute_metrics
    :param output_path: str, file path for saved PNG
    :param ref_is_exact: bool, whether the reference is exact (affects axis label)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=10, y=1.01)

    ref_style = "exact" if ref_is_exact else "reference"
    col_ref = "black"
    col_num = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

    ax1.plot(x_grid, u_ref, color=col_ref, linewidth=2.0, linestyle="-",
             label=ref_label)
    ax1.plot(x_grid, u_num, color=col_num, linewidth=1.5, linestyle="--",
             label=num_label)
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x, T)")
    ax1.set_title(
        f"GRW vs {ref_style}  |  {_metrics_str(metrics)}",
        fontsize=9,
    )
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.25)
    ax1.ticklabel_format(useOffset=False, axis="y", style="plain")

    error = u_num - u_ref
    ax2.plot(x_grid, error, color="steelblue", linewidth=1.4)
    ax2.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
    ax2.set_xlabel("x")
    ax2.set_ylabel("GRW \u2212 reference")
    ax2.set_title("Pointwise error", fontsize=9)
    ax2.grid(True, alpha=0.25)
    ax2.ticklabel_format(useOffset=False, axis="y", style="plain")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [verify] Saved plot    -> {output_path}")


# Burgers verification: relaxation GBMC

def _run_burgers_rbmc(cfg, output_dir, do_save_data):
    """
    Run relaxation GBMC verification for Burgers vs exact stationary-shock solution.

    Only stationary_shock IC is supported (solver raises NotImplementedError otherwise).
    Metrics are calculated from the raw cumulative-sum output (no smoothing).
    """
    from relaxation_gbmc import simulate_burgers_relaxation_gbmc

    ic_type = getattr(cfg, 'burgers_ic_type', '') or ''
    if ic_type != 'stationary_shock':
        print(f"  [Burgers RBMC] Only stationary_shock IC is supported "
              f"for relaxation_gbmc.  Got ic_type={ic_type!r}.  Exiting.")
        return

    print("\n" + "=" * 68)
    print("  Burgers -- Relaxation GBMC (BPC transport + Gradient Brownian MC)")
    print("  Reference: exact stationary shock  (unsmoothed raw-cumsum output)")
    print("=" * 68)

    N = cfg.num_points
    dt = cfg.time_step
    nu = cfg.diff_constant
    T = cfg.total_time
    L = cfg.domain_size
    amplitude = (float(cfg.burgers_ic_amplitude)
                 if getattr(cfg, 'burgers_ic_amplitude', None) is not None
                 else 1.0)
    xc = L / 2.0

    print(f"\n  Parameters")
    print(f"    nu = {nu},  T = {T},  dt = {dt}  ({int(T / dt)} steps)")
    print(f"    N = {N},  domain [0, {L}],  IC: stationary shock")
    print(f"    A = {amplitude:.4g},  x_c = {xc:.4g}")
    print(f"    Exact: u(x,t) = -A*tanh(A*(x-xc)/(2*nu))  [all t]")
    print(f"    Output: raw cumulative-sum reconstruction (no smoothing)")

    print(f"\n  Running Burgers relaxation GBMC ...", flush=True)
    grw_diag_dir = output_dir
    cfg._diag_dir = grw_diag_dir
    globs = [{"position": float(pos), "value": [float(val)]}
             for pos, val in cfg.initial_conditions]
    result = simulate_burgers_relaxation_gbmc(globs, cfg, _diag_dir=grw_diag_dir)
    if hasattr(cfg, '_diag_dir'):
        del cfg._diag_dir
    x_num = np.array([g["position"] for g in result])
    u_num = np.array([g["value"][0] for g in result])
    order = np.argsort(x_num)
    x_num, u_num = x_num[order], u_num[order]
    print("  Done.")

    x_grid = np.linspace(0.0, L, N)
    dx_grid = float(x_grid[1] - x_grid[0])

    u_ref = exact_burgers_stationary_shock(x_grid, nu,
                                           x_center=xc,
                                           amplitude=amplitude)
    ref_description = (f"exact stationary shock  "
                       f"(A={amplitude:.4g}, 2nu/A={2*nu/amplitude:.4g})")
    metrics_label = "Error metrics  (RBMC raw-cumsum vs exact stationary shock)"

    # Metrics from raw-cumsum output (no interpolation needed: output is on uniform grid)
    u_num_on_grid = np.interp(x_grid, x_num, u_num)
    metrics = compute_metrics(u_num_on_grid, u_ref, dx_grid)

    u_mid = 0.0  # stationary shock is antisymmetric about 0
    wave_loc_num = float(x_grid[np.argmin(np.abs(u_num_on_grid - u_mid))])
    wave_loc_ref = float(x_grid[np.argmin(np.abs(u_ref - u_mid))])
    wave_loc_diff = abs(wave_loc_num - wave_loc_ref)

    metrics.update({
        "wave_location_rbmc": wave_loc_num,
        "wave_location_ref": wave_loc_ref,
        "wave_location_diff": wave_loc_diff,
        "u_min": float(np.min(u_num_on_grid)),
        "u_max": float(np.max(u_num_on_grid)),
        "burgers_mode": "relaxation_gbmc",
        "burgers_ic_type": "stationary_shock",
        "output_smoothing": "none (raw cumulative-sum)",
    })

    _print_metrics(metrics, metrics_label)
    print(f"\n  Shock center")
    print(f"    RBMC: {wave_loc_num:.4f}")
    print(f"    Ref : {wave_loc_ref:.4f}")
    print(f"    Diff: {wave_loc_diff:.6f}")

    num_label = f"Relaxation GBMC  (N={N}, nu={nu}, dt={dt}, raw cumsum)"
    fig_title = (f"Burgers: Relaxation GBMC vs exact  (T={T}, nu={nu}, N={N})")
    plot_burgers_cole_hopf(
        x_grid, u_num_on_grid, u_ref,
        title=fig_title,
        num_label=num_label,
        ref_label=ref_description,
        metrics=metrics,
        output_path=os.path.join(output_dir, "comparison_plot.png"),
        ref_is_exact=True,
    )
    save_metrics(metrics, os.path.join(output_dir, "metrics.json"))
    if do_save_data:
        save_npz(x_grid, u_num_on_grid, u_ref,
                 os.path.join(output_dir, "comparison_data"))


def run_burgers(cfg, output_dir, do_save_data, ref_factor):
    mode = (getattr(cfg, 'burgers_mode', 'cole_hopf_grw') or 'cole_hopf_grw').lower()
    ic_type = getattr(cfg, 'burgers_ic_type', '') or ''

    if mode == 'relaxation_gbmc':
        return _run_burgers_rbmc(cfg, output_dir, do_save_data)

    # Only Cole-Hopf GRW is supported. Other modes are not routed.
    if mode != 'cole_hopf_grw' and mode != 'cole_hopf':
        print(f"  [Burgers] NOTE: mode '{mode}' is not supported. Using cole_hopf_grw.")
        mode = 'cole_hopf_grw'

    use_exact = ic_type in ('traveling_wave', 'stationary_shock')
    if not use_exact:
        print("  [Burgers] Only stationary_shock and traveling_wave ICs are supported.")
        print("  [Burgers] These have exact solutions for comparison. Exiting.")
        return

    print("\n" + "=" * 68)
    print("  Burgers -- Cole-Hopf GRW (Burgers -> heat equation via Cole-Hopf)")
    if ic_type == 'stationary_shock':
        ref_label_str = "exact stationary shock"
    else:
        ref_label_str = "exact traveling wave"
    print(f"  Reference: {ref_label_str}")
    print("=" * 68)

    N = cfg.num_points
    dt = cfg.time_step
    nu = cfg.diff_constant
    T = cfg.total_time
    L = cfg.domain_size

    print(f"\n  Parameters")
    print(f"    nu = {nu},  T = {T},  dt = {dt}  ({int(T / dt)} steps)")
    print(f"    N = {N} globs,  domain [0, {L}]")
    print(f"    Cole-Hopf: u = -2*nu*phi_x/phi,  phi solves phi_t = nu*phi_xx")
    if ic_type == 'stationary_shock':
        amplitude = (float(cfg.burgers_ic_amplitude)
                     if getattr(cfg, 'burgers_ic_amplitude', None) is not None
                     else None)
        A_str = f"A={amplitude:.4g}" if amplitude is not None else "A=?"
        print(f"    IC: stationary shock  ({A_str}, xc={L/2:.3g})")
        print(f"    Exact: u(x,t) = -A*tanh(A*(x-xc)/(2*nu))  [all t]")
    elif ic_type == 'traveling_wave':
        print(f"    IC: traveling wave  (xc={L/2:.3g}, wave speed c=1)")
        print(f"    Exact: u(x,t) = 1 - 2*sqrt(nu)*tanh((x-xc-t)/sqrt(nu))")

    print(f"\n  Running Burgers GRW (cole_hopf_grw) ...", flush=True)
    grw_diag_dir = output_dir
    x_num, u_num = _run_burgers_grw(cfg, diag_dir=grw_diag_dir)
    print("  Done.")

    x_grid = np.linspace(0.0, L, N)
    dx_grid = float(x_grid[1] - x_grid[0])

    if ic_type == 'stationary_shock':
        ic_x = np.array([pos for pos, _ in cfg.initial_conditions])
        ic_vals = np.array([val for _, val in cfg.initial_conditions])
        u_ref = np.interp(x_grid, ic_x, ic_vals)
        amplitude = (float(cfg.burgers_ic_amplitude)
                     if getattr(cfg, 'burgers_ic_amplitude', None) is not None
                     else float(np.max(np.abs(ic_vals))))
        ref_description = (
            f"exact stationary shock  "
            f"(A={amplitude:.4g}, width=2*nu/A={2*nu/amplitude:.4g})"
        )
        ref_short = "exact"
        ref_is_exact = True
        metrics_label = "Error metrics  (Cole-Hopf GRW vs exact stationary shock)"

    elif ic_type == 'traveling_wave':
        x_center = L / 2.0
        u_ref = exact_burgers_traveling_wave(x_grid, T, nu, x_center=x_center)
        ref_description = (
            f"exact traveling wave  "
            f"(c=1, shock width=sqrt(nu)={np.sqrt(nu):.4g})"
        )
        ref_short = "exact"
        ref_is_exact = True
        metrics_label = "Error metrics  (Cole-Hopf GRW vs exact traveling wave)"

    u_num_on_grid = np.interp(x_grid, x_num, u_num)
    metrics = compute_metrics(u_num_on_grid, u_ref, dx_grid)

    # Shock / wave center location (zero-crossing of u - mean(u)).
    u_mid = float(np.mean(u_ref))
    wave_loc_num = float(x_grid[np.argmin(np.abs(u_num_on_grid - u_mid))])
    wave_loc_ref = float(x_grid[np.argmin(np.abs(u_ref - u_mid))])
    wave_loc_diff = abs(wave_loc_num - wave_loc_ref)

    metrics.update({
        "wave_location_grw": wave_loc_num,
        "wave_location_ref": wave_loc_ref,
        "wave_location_diff": wave_loc_diff,
        "u_min": float(np.min(u_num_on_grid)),
        "u_max": float(np.max(u_num_on_grid)),
        "burgers_mode": "cole_hopf_grw",
        "burgers_ic_type": ic_type,
    })

    _print_metrics(metrics, metrics_label)
    print(f"\n  Wave / shock center")
    print(f"    GRW : {wave_loc_num:.4f}")
    print(f"    Ref : {wave_loc_ref:.4f}")
    print(f"    Diff: {wave_loc_diff:.6f}")

    # Figures.
    num_label = f"Cole-Hopf GRW  (N={N}, nu={nu}, dt={dt})"
    fig_title = (
        f"Burgers: Cole-Hopf GRW vs {ref_short}  "
        f"(T={T}, nu={nu}, N={N})"
    )
    plot_burgers_cole_hopf(
        x_grid, u_num_on_grid, u_ref,
        title=fig_title,
        num_label=num_label,
        ref_label=ref_description,
        metrics=metrics,
        output_path=os.path.join(output_dir, "comparison_plot.png"),
        ref_is_exact=ref_is_exact,
    )

    save_metrics(metrics, os.path.join(output_dir, "metrics.json"))
    if do_save_data:
        save_npz(x_grid, u_num_on_grid, u_ref,
                 os.path.join(output_dir, "comparison_data"))


# FitzHugh-Nagumo verification (primary solver: GRW particle method)
# FHN helpers: exact traveling wave and GRW runners

def exact_fhn_traveling_wave(x, t, a, x_center=0.0):
    """
    Exact traveling-wave solution for the scalar FHN equation.

    u(x, t) = 1 / (1 + exp(-(x + theta*t - x_center) / 2))
    theta = sqrt(2) * (0.5 - a)

    The wave front (u = 0.5) starts at x = x_center at t = 0 and moves at
    speed -theta in the x-direction (leftward for a < 0.5, rightward for a > 0.5).

    :param x: array-like, spatial coordinates
    :param t: float, time
    :param a: float, FHN threshold parameter
    :param x_center: float, initial position of the wave center (u = 0.5)
    :return: ndarray, exact u values at positions x, time t
    """
    theta = np.sqrt(2.0) * (0.5 - a)
    xi = np.asarray(x, dtype=float) + theta * float(t) - x_center
    return 1.0 / (1.0 + np.exp(-xi / 2.0))


def _run_fhn_grw_scalar(cfg, total_time_override=None, diag_dir=None):
    """
    Run the scalar GRW FHN solver.

    Returns (x_sorted, u_reconstructed) on a uniform output grid.
    u is reconstructed from the sorted glob list via cumulative summation
    (the GRW integration of the gradient globs).

    :param cfg: SimulationConfig
    :param total_time_override: if not None, use this total time instead of cfg.total_time
    :param diag_dir: optional path; forwarded as _diag_dir to the solver for
                                diagnostic figure output (front vs time, weight snapshots)
    :return: (x_grid, u_grid) arrays on a uniform output grid
    """
    ic_type = getattr(cfg, 'fhn_ic_type', '') or ''
    globs = [
        {'position': float(pos), 'value': float(val)}
        for pos, val in cfg.initial_conditions
    ]

    run_cfg = cfg
    if total_time_override is not None and total_time_override != cfg.total_time:
        run_cfg = SimulationConfig(
            equation_type=cfg.equation_type,
            domain_type=cfg.domain_type,
            domain_size=cfg.domain_size,
            boundary_conditions=cfg.boundary_conditions,
            diff_constant=cfg.diff_constant,
            time_step=cfg.time_step,
            total_time=float(total_time_override),
            num_points=cfg.num_points,
            initial_conditions=list(cfg.initial_conditions),
            reaction_term=cfg.reaction_term,
            a=cfg.a,
            b=cfg.b,
            tau=cfg.tau,
            fhn_ic_type=ic_type,
        )

    if diag_dir is not None:
        run_cfg._diag_dir = diag_dir

    result = simulate_fitzhugh_nagumo(globs, run_cfg)

    x_pos = np.array([g['position'] for g in result])
    w_val = np.array([float(g['value']) for g in result])
    order = np.argsort(x_pos)
    x_s = x_pos[order]
    w_s = w_val[order]

    L = cfg.domain_size
    n_grid = max(200, cfg.num_points)
    x_grid = np.linspace(0.0, L, n_grid)
    u_grid = np.zeros(n_grid)

    for j, xj in enumerate(x_grid):
        u_grid[j] = float(np.sum(w_s[x_s <= xj]))

    return x_grid, u_grid






def plot_fhn_scalar_grw(
    snap_times, snap_exact, snap_num,
    x_grid, title, ic_label,
    output_path,
    metrics_final,
):
    """
    Multi-time verification figure for the scalar FHN GRW.

    Layout: 2 rows x 2 columns (up to 4 time snapshots).
      Left column: numerical GRW vs exact traveling wave
      Right column: pointwise error

    :param snap_times: list of floats, snapshot times
    :param snap_exact: list of 1d arrays, exact solution at each snap time
    :param snap_num: list of 1d arrays, GRW reconstruction at each snap time
    :param x_grid: 1d array, common output grid
    :param title: str, figure suptitle
    :param ic_label: str, IC type label for annotations
    :param output_path: str, file path to save the PNG
    :param metrics_final: dict, error metrics at the final snapshot time
    """
    n_snap = len(snap_times)
    n_rows = (n_snap + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows + 0.8))
    if n_rows == 1:
        axes = np.array([axes])
    fig.suptitle(title, fontsize=10, y=0.99)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    col_num = colors[1] if len(colors) > 1 else 'crimson'
    col_exa = 'black'

    for idx, (t_snap, u_ex, u_gr) in enumerate(
            zip(snap_times, snap_exact, snap_num)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        err = u_gr - u_ex
        m = compute_metrics(u_gr, u_ex, float(x_grid[1] - x_grid[0]))

        ax.plot(x_grid, u_ex, color=col_exa, linewidth=2, linestyle='-', label='exact')
        ax.plot(x_grid, u_gr, color=col_num, linewidth=1.5, linestyle='--', label='GRW')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(
            f't = {t_snap:.1f}  |  {_metrics_str(m)}',
            fontsize=9,
        )
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(useOffset=False, axis='y', style='plain')

        # Small inset-style error curve on the same axis using a twin.
        ax2 = ax.twinx()
        ax2.plot(x_grid, err, color='gray', linewidth=0.9, linestyle=':', alpha=0.7)
        ax2.axhline(0.0, color='gray', linewidth=0.5, linestyle=':')
        ax2.set_ylabel('error', fontsize=7, color='gray')
        ax2.tick_params(axis='y', labelsize=7, colors='gray')
        ax2.ticklabel_format(useOffset=False, axis='y', style='sci', scilimits=(-2, 2))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f'  [verify] Saved plot     -> {output_path}')
    plt.close(fig)




def run_fhn(cfg, output_dir, do_save_data, ref_factor):
    """
    FHN verification: scalar GRW vs exact traveling-wave solution.

    Runs the GRW solver at multiple time snapshots (t = 0, T/3, 2T/3, T)
    and compares each against the analytic solution:
      u(x, t) = 1 / (1 + exp(-(x + theta*t - x_center) / 2))
      theta = sqrt(2) * (0.5 - a)
    """
    _run_fhn_scalar(cfg, output_dir, do_save_data)


def _run_fhn_scalar(cfg, output_dir, do_save_data):
    """
    Scalar GRW FHN verification.

    Runs the GRW solver at multiple time snapshots and compares each to the
    exact traveling-wave solution:
      u(x, t) = 1 / (1 + exp(-(x + theta*t - x_center) / 2))
      theta = sqrt(2) * (0.5 - a)
    """
    print("\n" + "=" * 62)
    print("  FitzHugh-Nagumo -- scalar GRW vs exact solution")
    print("  Primary : GRW gradient-side method")
    print("  Reference: exact traveling-wave solution (analytic)")
    print("=" * 62)

    N = cfg.num_points
    dt = cfg.time_step
    T = cfg.total_time
    L = cfg.domain_size
    D = cfg.diff_constant
    a_ = cfg.a if cfg.a is not None else 0.25
    ic_type = getattr(cfg, 'fhn_ic_type', '') or ''
    theta = np.sqrt(2.0) * (0.5 - a_)

    # Infer wave center from the IC: the midpoint value x_center.
    # For steady_solution, the first glob position should be near the left tail,
    # last near the right tail.  Median position estimates the center.
    x_ic = np.array([pos for pos, _ in cfg.initial_conditions])
    x_center = float(np.median(x_ic))

    print(f"\n  Parameters")
    print(f"    a={a_:.4g},  D={D:.4g},  theta = sqrt(2)*(0.5-a) = {theta:.4g}")
    print(f"    T = {T},  dt = {dt}  ({int(T / dt)} GRW steps)")
    print(f"    N = {N} globs,  domain [0, {L}]")
    print(f"    IC type: {ic_type},  wave center at x = {x_center:.3g}")

    # Snapshot times: 0, T/3, 2T/3, T  (capped at 4 times for the 2x2 layout).
    t_snaps = [0.0]
    n_panels = 4
    step = T / (n_panels - 1)
    for k in range(1, n_panels):
        t_snaps.append(round(k * step, 6))
    # Make sure the last entry is exactly T.
    t_snaps[-1] = T

    snap_exact = []
    snap_num = []
    snap_fronts = []  # (t_snap, grw_front, exact_front) for each non-zero snapshot
    x_grid_out = None

    for t_snap in t_snaps:
        u_ex = exact_fhn_traveling_wave(
            np.linspace(0.0, L, max(200, N)),
            t_snap, a_, x_center=x_center)
        snap_exact.append(u_ex)

        if t_snap == 0.0:
            # IC reconstruction from initial globs (no time stepping).
            x_s = np.sort(x_ic)
            w_ic = np.array([float(w) for _, w in cfg.initial_conditions])
            w_ic_s = w_ic[np.argsort(x_ic)]
            x_g = np.linspace(0.0, L, max(200, N))
            u_g = np.array([float(np.sum(w_ic_s[x_s <= xj])) for xj in x_g])
        else:
            # Pass diag_dir only for the final snapshot so we get a single
            # front-vs-time diagnostic figure covering the full run.
            is_final = (t_snap == T)
            ddir = output_dir if is_final else None
            print(f"\n  Running GRW to t = {t_snap:.3g} ...", flush=True)
            x_g, u_g = _run_fhn_grw_scalar(
                cfg, total_time_override=t_snap, diag_dir=ddir)
            print("  Done.")

        if x_grid_out is None:
            x_grid_out = x_g
        snap_num.append(u_g)

        # Per-snapshot front comparison (skip t=0).
        if t_snap > 0.0:
            snap_fronts.append((t_snap, u_g, u_ex))

    if x_grid_out is None:
        x_grid_out = np.linspace(0.0, L, max(200, N))

    # All snapshots on the same x_grid; re-interpolate if needed.
    for i in range(len(snap_num)):
        if len(snap_num[i]) != len(x_grid_out):
            snap_num[i] = np.interp(x_grid_out, np.linspace(0.0, L, len(snap_num[i])), snap_num[i])
        if len(snap_exact[i]) != len(x_grid_out):
            snap_exact[i] = np.interp(x_grid_out, np.linspace(0.0, L, len(snap_exact[i])), snap_exact[i])

    dx_g = float(x_grid_out[1] - x_grid_out[0])
    m_fin = compute_metrics(snap_num[-1], snap_exact[-1], dx_g)

    # Front location: where u = 0.5 (zero-crossing of u - 0.5).
    def _front_loc(u, x):
        diff = u - 0.5
        idx = np.where(np.diff(np.sign(diff)))[0]
        if len(idx) == 0:
            return float(x[np.argmin(np.abs(diff))])
        i = idx[0]
        if u[i+1] == u[i]:
            return float(x[i])
        return float(x[i] + (0.5 - u[i]) / (u[i+1] - u[i]) * (x[i+1] - x[i]))

    front_grw = _front_loc(snap_num[-1], x_grid_out)
    front_ex = _front_loc(snap_exact[-1], x_grid_out)
    front_diff = abs(front_grw - front_ex)

    combined_metrics = {
        "ic_type": ic_type,
        "a": float(a_),
        "theta": float(theta),
        "x_center": float(x_center),
        "T": float(T),
        "N_globs": int(N),
        "D": float(D),
        "dt": float(dt),
        "u_final": m_fin,
        "front_location_grw": front_grw,
        "front_location_exact": front_ex,
        "front_location_diff": front_diff,
    }

    # Per-snapshot front table.
    print(f"\n  Front location per snapshot  (u=0.5 crossing, exact speed = {theta:.4g})")
    print(f"    {'t':>6}   {'GRW front':>10}   {'exact front':>11}   {'error':>8}")
    print(f"    {'--':>6}   {'----------':>10}   {'-----------':>11}   {'-------':>8}")
    for (t_s, u_g_s, u_ex_s) in snap_fronts:
        fg = _front_loc(u_g_s, x_grid_out)
        fe = _front_loc(u_ex_s, x_grid_out)
        print(f"    {t_s:6.3g}   {fg:10.4f}   {fe:11.4f}   {fg-fe:+8.4f}")

    _print_metrics(m_fin, f"\n  u error metrics  (GRW vs exact, t = {T})")
    print(f"\n  Front location diagnostics (t = {T})")
    print(f"    Front (GRW)   : {front_grw:.4f}")
    print(f"    Front (exact) : {front_ex:.4f}")
    print(f"    Difference    : {_fmt_val(front_diff)}")

    ic_label = f"IC: {ic_type}"
    title = (
        f"FHN scalar GRW vs exact traveling wave  "
        f"(a={a_:.4g}, D={D:.4g}, theta={theta:.4g})\n"
        f"{ic_label}  |  t in [0, {T}]  |  "
        f"final: {_metrics_str(m_fin)}"
    )

    plot_path = os.path.join(output_dir, "comparison_plot.png")
    plot_fhn_scalar_grw(
        snap_times=t_snaps,
        snap_exact=snap_exact,
        snap_num=snap_num,
        x_grid=x_grid_out,
        title=title,
        ic_label=ic_label,
        output_path=plot_path,
        metrics_final=m_fin,
    )
    save_metrics(combined_metrics, os.path.join(output_dir, "metrics.json"))

    if do_save_data:
        path = os.path.join(output_dir, "comparison_data.npz")
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            path,
            x=x_grid_out,
            u_grw=snap_num[-1],
            u_exact=snap_exact[-1],
            u_error=snap_num[-1] - snap_exact[-1],
        )
        print(f"  [verify] Saved data    -> {path}")




def main():
    parser = argparse.ArgumentParser(
        description="Verification and benchmark comparison for the GRW solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python verify_solver.py\n"
            "  python verify_solver.py --equation heat\n"
            "  python verify_solver.py --equation burgers --ref-factor 8\n"
            "  python verify_solver.py --equation fhn --save-data\n"
        ),
    )
    parser.add_argument(
        "--equation", "-e",
        choices=["heat", "burgers", "burgers_rbmc", "fhn", "all"],
        default="all",
        help="Equation to verify (default: all; burgers_rbmc uses relaxation GBMC)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to JSON config (uses canonical default per equation if omitted)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: output/verify/<equation>)",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Also save comparison_data.npz with x, numerical, reference, error arrays",
    )
    parser.add_argument(
        "--ref-factor",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Resolution multiplier for Burgers/FHN reference solves "
            "(higher = more accurate reference, longer runtime; default: 5)"
        ),
    )
    args = parser.parse_args()

    equations = (["heat", "burgers", "fhn"]
                 if args.equation == "all"
                 else [args.equation])

    for eq in equations:
        cfg_path = args.config or _DEFAULT_CONFIGS.get(eq, _DEFAULT_CONFIGS["burgers"])
        output_dir = args.output_dir or os.path.join("output", "verify",
                                                      eq.replace("_rbmc", ""))

        print(f"\nLoading config: {cfg_path}")
        loaded_cfg = cfg_module.load_config_from_json(cfg_path)

        if eq == "heat":
            run_heat(loaded_cfg, output_dir, args.save_data)
        elif eq in ("burgers", "burgers_rbmc"):
            run_burgers(loaded_cfg, output_dir, args.save_data, args.ref_factor)
        elif eq == "fhn":
            run_fhn(loaded_cfg, output_dir, args.save_data, args.ref_factor)


if __name__ == "__main__":
    main()
