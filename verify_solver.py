#!/usr/bin/env python3
"""
verify_solver.py
================
Verification and benchmark comparison for the GRW solver suite.

For each equation the script runs the primary solver and compares output against
a trusted benchmark:

  heat    -- exact analytical solution (error function), valid for step IC.
             Primary solver: thesis-faithful GRW.

  burgers -- high-resolution FD reference (ref_factor x finer grid / smaller dt).
             Primary solver: experimental GRW-inspired Lagrangian particle method.
             Reference solver: standard finite-difference (simulate_burgers_fd).

  fhn     -- high-resolution FD reference (ref_factor x smaller dt, same grid).
             Primary solver: experimental GRW-inspired particle method.
             Reference solver: standard finite-difference (simulate_fitzhugh_nagumo_fd).

Burgers and FHN comparisons are against *numerical FD references*, not exact solutions.
The output labels this explicitly.  The purpose of the error metrics on main is to
quantify GRW feasibility and limitations, not to advertise accuracy.

Usage examples:
  python verify_solver.py                                         # run all three
  python verify_solver.py --equation heat
  python verify_solver.py --equation burgers --config configs/burgers_shock.json
  python verify_solver.py --equation fhn --output-dir output/fhn_check
  python verify_solver.py --equation heat --save-data
  python verify_solver.py --equation burgers --ref-factor 8

Outputs per equation (written to --output-dir, default: output/verify/<equation>):
  comparison_plot.png   two-panel figure: numerical vs reference + pointwise error
  metrics.json          all computed error metrics and equation-specific diagnostics
  comparison_data.npz   (optional, with --save-data) x grid, solutions, error arrays
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
    simulate_burgers_fd,
    simulate_fitzhugh_nagumo,
    simulate_fitzhugh_nagumo_fd,
)


# ---------------------------------------------------------------------------
# Default config paths (relative to repo root)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIGS = {
    "heat":    "configs/heat_step_dirichlet.json",
    "burgers": "configs/burgers_shock.json",
    "fhn":     "configs/fhn_oscillatory.json",
}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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
        "l1":            float(np.sum(np.abs(diff)) * dx),
        "l2":            float(np.sqrt(np.sum(diff ** 2) * dx)),
        "linf":          linf,
        "max_abs_error": linf,
        "rel_l2":        float(np.sqrt(np.sum(diff ** 2) * dx) / ref_norm)
                         if ref_norm > 1e-12 else None,
        "mean_signed":   float(np.mean(diff)),
        "rmse":          float(np.sqrt(np.mean(diff ** 2))),
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

    Values >= 1e-3 use 4 fixed decimal places.  Smaller values use 2-significant-
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

    ax1.plot(x, reference, color="black",    linewidth=2,   linestyle="-",  label=ref_label)
    ax1.plot(x, numerical, color="crimson",  linewidth=1.5, linestyle="--", label=num_label)

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
    plt.show(block=True)
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


# ---------------------------------------------------------------------------
# Heat verification  (exact analytical benchmark)
# ---------------------------------------------------------------------------

def run_heat(cfg, output_dir, do_save_data):
    print("\n" + "=" * 62)
    print("  Heat GRW -- verification vs exact error-function solution")
    print("=" * 62)

    alpha = cfg.diff_constant
    T     = cfg.total_time
    L     = cfg.domain_size
    N     = cfg.num_points
    dt    = cfg.time_step
    uL    = float(cfg.boundary_conditions["LEFT"]["value"])
    uR    = float(cfg.boundary_conditions["RIGHT"]["value"])

    # For a step IC, every glob starts at the same position (the jump location).
    x0          = float(cfg.initial_conditions[0][0])
    jump_height = float(uR - uL)
    exp_std     = float(np.sqrt(2.0 * alpha * T))

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
    values    = np.array([g["value"]    for g in final_globs])

    mean_pos = float(np.mean(positions))
    std_pos  = float(np.std(positions))
    total_wt = float(np.sum(values))

    print(f"\n  Glob diagnostics")
    print(f"    Mean position  : {mean_pos:.6f}   (expected {x0:.4f} for sym. domain)")
    print(f"    Std  position  : {std_pos:.6f}   (expected {exp_std:.6f})")
    print(f"    Total weight   : {total_wt:.8f}   (expected {jump_height:.8f})")

    # Reconstruct u(x, T) by numerical integration of the glob list.
    nbins = min(400, max(200, N // 50))
    edges = np.linspace(0.0, L, nbins + 1)
    bin_w, _ = np.histogram(positions, bins=edges, weights=values)
    u_num     = uL + np.cumsum(bin_w)
    x_grid    = 0.5 * (edges[:-1] + edges[1:])
    dx        = float(edges[1] - edges[0])

    u_exact = exact_heat_step(x_grid, T, x0, uL, uR, alpha)
    metrics = compute_metrics(u_num, u_exact, dx)

    metrics["glob_mean_position"]  = mean_pos
    metrics["glob_std_position"]   = std_pos
    metrics["theoretical_std"]     = exp_std
    metrics["weight_conservation"] = total_wt
    metrics["expected_weight"]     = jump_height
    metrics["num_globs"]           = N
    metrics["alpha"]               = alpha
    metrics["T"]                   = T

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


# ---------------------------------------------------------------------------
# Burgers verification
# Primary solver: GRW-inspired Lagrangian particle method
# Reference solver: high-resolution finite-difference (simulate_burgers_fd)
# ---------------------------------------------------------------------------

def _run_burgers_grw(cfg):
    """
    Run the GRW Lagrangian particle solver for Burgers.

    Returns (x_sorted, u_sorted) where x values are the final particle
    positions (scattered, not on a uniform grid).
    """
    globs = [{"position": float(pos), "value": [float(val)]}
             for pos, val in cfg.initial_conditions]
    result = simulate_burgers(globs, cfg)
    x = np.array([g["position"] for g in result])
    u = np.array([g["value"][0] for g in result])
    order = np.argsort(x)
    return x[order], u[order]


def _run_burgers_fd(cfg):
    """
    Run the FD reference solver for Burgers on a fixed uniform grid.

    Returns (x_sorted, u_sorted) where x is the original uniform grid.
    """
    globs = [{"position": float(pos), "value": [float(val)]}
             for pos, val in cfg.initial_conditions]
    result = simulate_burgers_fd(globs, cfg)
    x = np.array([g["position"] for g in result])
    u = np.array([g["value"][0] for g in result])
    order = np.argsort(x)
    return x[order], u[order]


def _burgers_ref_config(cfg, ref_factor):
    """
    Build a higher-resolution FD reference config for Burgers.

    Increases num_points by ref_factor and reduces time_step by the same factor
    to maintain a comparable CFL condition.  The IC is interpolated onto the
    finer grid.  This config is passed to _run_burgers_fd (not the GRW solver).
    """
    N_ref  = cfg.num_points * ref_factor
    dt_ref = cfg.time_step / ref_factor

    x_orig = np.array([float(pos) for pos, _ in cfg.initial_conditions])
    u_orig = np.array([float(val) for _, val in cfg.initial_conditions])
    x_fine = np.linspace(float(x_orig[0]), float(x_orig[-1]), N_ref)
    u_fine = np.interp(x_fine, x_orig, u_orig)
    ref_ic = list(zip(x_fine.tolist(), u_fine.tolist()))

    return SimulationConfig(
        equation_type=cfg.equation_type,
        domain_type=cfg.domain_type,
        domain_size=cfg.domain_size,
        boundary_conditions=cfg.boundary_conditions,
        diff_constant=cfg.diff_constant,
        time_step=dt_ref,
        total_time=cfg.total_time,
        num_points=N_ref,
        initial_conditions=ref_ic,
        reaction_term=cfg.reaction_term,
    )


def run_burgers(cfg, output_dir, do_save_data, ref_factor):
    print("\n" + "=" * 62)
    print("  Burgers -- GRW particle solver vs FD reference")
    print("  Primary : GRW Lagrangian particle method (experimental)")
    print("  Reference: standard FD, finer grid + smaller dt")
    print("=" * 62)

    N  = cfg.num_points
    dt = cfg.time_step
    nu = cfg.diff_constant
    T  = cfg.total_time
    L  = cfg.domain_size
    uL = float(cfg.boundary_conditions["LEFT"]["value"])
    uR = float(cfg.boundary_conditions["RIGHT"]["value"])

    print(f"\n  Parameters")
    print(f"    nu = {nu},  T = {T},  dt = {dt}  ({int(T / dt)} steps)")
    print(f"    N = {N} particles,  domain [0, {L}]")
    print(f"    GRW: advection step u*dt + diffusion step Normal(0, sqrt(2*nu*dt))")

    print("\n  Running Burgers GRW particle solver ...", flush=True)
    x_num, u_num = _run_burgers_grw(cfg)
    print("  Done.")

    ref_cfg = _burgers_ref_config(cfg, ref_factor)
    print(
        f"\n  Running Burgers FD reference  "
        f"(N={ref_cfg.num_points}, dt={ref_cfg.time_step:.5g}) ...",
        flush=True,
    )
    x_ref, u_ref = _run_burgers_fd(ref_cfg)
    print("  Done.")

    # Interpolate GRW result (scattered positions) onto the reference grid.
    u_num_on_ref = np.interp(x_ref, x_num, u_num)
    dx_ref = float(L) / ref_cfg.num_points
    metrics = compute_metrics(u_num_on_ref, u_ref, dx_ref)

    u_mid = 0.5 * (uL + uR)
    shock_num = float(x_num[np.argmin(np.abs(u_num - u_mid))]) if len(x_num) else None
    shock_ref = float(x_ref[np.argmin(np.abs(u_ref - u_mid))]) if len(x_ref) else None

    u_hi = max(uL, uR)
    u_lo = min(uL, uR)
    overshoot  = max(0.0, float(np.max(u_num)) - u_hi)
    undershoot = max(0.0, u_lo - float(np.min(u_num)))

    metrics["u_min"]                    = float(np.min(u_num))
    metrics["u_max"]                    = float(np.max(u_num))
    metrics["physical_upper_bound"]     = u_hi
    metrics["physical_lower_bound"]     = u_lo
    metrics["overshoot"]                = overshoot
    metrics["undershoot"]               = undershoot
    metrics["shock_location_grw"]       = shock_num
    metrics["shock_location_fd_ref"]    = shock_ref
    metrics["shock_location_diff"]      = (
        float(abs(shock_num - shock_ref))
        if (shock_num is not None and shock_ref is not None) else None
    )
    metrics["ref_factor"]      = ref_factor
    metrics["ref_num_points"]  = ref_cfg.num_points
    metrics["ref_time_step"]   = ref_cfg.time_step

    _print_metrics(metrics, "Error metrics  (GRW particle vs FD reference)")
    if shock_num is not None and shock_ref is not None:
        print(f"\n  Shock diagnostics")
        print(f"    Shock location (GRW)     : {shock_num:.4f}")
        print(f"    Shock location (FD ref)  : {shock_ref:.4f}")
        print(f"    Shock location diff      : {abs(shock_num - shock_ref):.6f}")
    print(f"\n  Solution range  (physical bounds: [{u_lo}, {u_hi}])")
    print(f"    u_min = {_fmt_val(metrics['u_min'])},  u_max = {_fmt_val(metrics['u_max'])}")
    print(f"    Overshoot   (u_max - {u_hi}) : +{_fmt_val(overshoot)}"
          + ("  <-- above physical bound" if overshoot > 1e-3 else ""))
    print(f"    Undershoot  ({u_lo} - u_min) : +{_fmt_val(undershoot)}"
          + ("  <-- below physical bound" if undershoot > 1e-3 else ""))

    bound_str = f"u in [{metrics['u_min']:.3f}, {metrics['u_max']:.3f}]"
    if overshoot > 1e-3 or undershoot > 1e-3:
        os_str = f"overshoot +{overshoot:.4f}" if overshoot > 1e-3 else ""
        us_str = f"undershoot -{undershoot:.4f}" if undershoot > 1e-3 else ""
        nonphys = ",  ".join(filter(None, [os_str, us_str]))
        bound_str += f"  [{nonphys}]"

    title = (
        f"Burgers: GRW particle vs FD reference  (T={T}, nu={nu}, N={N})\n"
        f"{bound_str}"
    )
    plot_comparison(
        x_ref, u_num_on_ref, u_ref,
        title=title,
        ylabel="u(x, T)",
        num_label=f"GRW particle  (N={N}, dt={dt})",
        ref_label=f"FD reference  (N={ref_cfg.num_points}, dt={ref_cfg.time_step:.5g})",
        metrics=metrics,
        output_path=os.path.join(output_dir, "comparison_plot.png"),
        ref_note="Reference = high-resolution FD solution  (not the GRW method)",
    )
    save_metrics(metrics, os.path.join(output_dir, "metrics.json"))
    if do_save_data:
        save_npz(x_ref, u_num_on_ref, u_ref, os.path.join(output_dir, "comparison_data"))


# ---------------------------------------------------------------------------
# FitzHugh-Nagumo verification
# Primary solver: GRW-inspired particle method
# Reference solver: high-resolution FD (simulate_fitzhugh_nagumo_fd)
# ---------------------------------------------------------------------------

def _run_fhn_grw(cfg):
    """
    Run the GRW-inspired particle solver for FHN.

    Returns (x_sorted, u_sorted, v_sorted) where x values are the final
    particle positions (scattered, not on a uniform grid).
    """
    globs = [{"position": float(pos), "value": list(val)}
             for pos, val in cfg.initial_conditions]
    result = simulate_fitzhugh_nagumo(globs, cfg)
    x  = np.array([g["position"] for g in result])
    uv = np.array([g["value"]    for g in result])
    order = np.argsort(x)
    return x[order], uv[order, 0], uv[order, 1]


def _run_fhn_fd(cfg):
    """
    Run the FD reference solver for FHN on a fixed uniform grid.

    Returns (x_sorted, u_sorted, v_sorted) where x is the uniform grid.
    """
    globs = [{"position": float(pos), "value": list(val)}
             for pos, val in cfg.initial_conditions]
    result = simulate_fitzhugh_nagumo_fd(globs, cfg)
    x  = np.array([g["position"] for g in result])
    uv = np.array([g["value"]    for g in result])
    order = np.argsort(x)
    return x[order], uv[order, 0], uv[order, 1]


def _fhn_ref_config(cfg, ref_factor):
    """
    Build a finer-timestep FHN config for the FD reference solve.

    The spatial grid is kept identical to the primary run so both solutions
    share the same x-coordinates (pointwise comparison without interpolation).
    Only dt is reduced (by ref_factor), which isolates temporal error.

    CFL stability for the explicit-FD reference solver:
      dt <= dx^2 / (2 * D)
    The reference dt_ref = dt / ref_factor makes this constraint more easily
    satisfied.  The CFL condition does NOT apply to the GRW particle solver.
    """
    dt_ref = cfg.time_step / ref_factor

    # Safety: warn if even the coarse dt violates the FD CFL limit.
    if cfg.diff_constant > 0.0:
        dx = cfg.domain_size / (cfg.num_points - 1)
        cfl_limit = dx**2 / (2.0 * cfg.diff_constant)
        if cfg.time_step > cfl_limit * 1.05:
            print(
                f"  [verify] WARNING: coarse dt={cfg.time_step:.5g} exceeds "
                f"CFL limit {cfl_limit:.5g} for D={cfg.diff_constant}, dx={dx:.5g}. "
                f"Results may be unstable."
            )

    return SimulationConfig(
        equation_type=cfg.equation_type,
        domain_type=cfg.domain_type,
        domain_size=cfg.domain_size,
        boundary_conditions=cfg.boundary_conditions,
        diff_constant=cfg.diff_constant,
        time_step=dt_ref,
        total_time=cfg.total_time,
        num_points=cfg.num_points,
        initial_conditions=list(cfg.initial_conditions),
        reaction_term=cfg.reaction_term,
        a=cfg.a,
        b=cfg.b,
        tau=cfg.tau,
    )


def plot_fhn_comparison(
    x, u_num, u_ref, v_num, v_ref,
    title, num_label, ref_label, m_u, m_v,
    output_path,
    diag_u=None,
    diag_v=None,
):
    """
    Four-panel figure for FHN verification.

    Layout (2 rows x 2 columns):
      [0,0] u: numerical vs reference overlay
      [0,1] u: pointwise error
      [1,0] v: numerical vs reference overlay
      [1,1] v: pointwise error

    diag_u / diag_v: optional dicts with keys
      peak_num, peak_ref, peak_loc_num, peak_loc_ref
    These are displayed as a compact annotation box on the overlay panels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=10, y=0.98)

    u_err = u_num - u_ref
    v_err = v_num - v_ref

    _diag_box = dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                     edgecolor="gray", alpha=0.85)

    def _overlay(ax, x, num, ref, ylabel, num_lbl, ref_lbl, metrics, diag=None):
        ax.plot(x, ref, color="black",   linewidth=2,   linestyle="-",  label=ref_lbl)
        ax.plot(x, num, color="crimson", linewidth=1.5, linestyle="--", label=num_lbl)
        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        ax.set_title(_metrics_str(metrics), fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        # Disable matplotlib's offset-notation on the y-axis so that values like
        # 1.1994 are not displayed as "+1.1994e0" or "1.199 + offset".
        ax.ticklabel_format(useOffset=False, axis="y", style="plain")

        if diag is not None:
            pn  = _fmt_compact(diag.get("peak_num"))
            pr  = _fmt_compact(diag.get("peak_ref"))
            pd  = _fmt_compact(diag.get("peak_num") - diag.get("peak_ref"))
            ln  = _fmt_compact(diag.get("peak_loc_num"))
            lr  = _fmt_compact(diag.get("peak_loc_ref"))
            ld  = _fmt_compact(abs(diag.get("peak_loc_num") - diag.get("peak_loc_ref")))
            txt = (f"peak:  num={pn},  ref={pr},  diff={pd}\n"
                   f"loc:   num={ln},  ref={lr},  diff={ld}")
            ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                    fontsize=7.5, verticalalignment="top",
                    fontfamily="monospace", bbox=_diag_box)

    def _error(ax, x, err, ylabel):
        ax.plot(x, err, color="steelblue", linewidth=1.5)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        ax.set_title("Pointwise error", fontsize=9)
        ax.grid(True, alpha=0.3)
        # Use scientific notation with explicit exponent on error axes so tiny
        # values like 1e-4 are shown as "1e-4" and not "0.0001" or offset.
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    _overlay(axes[0, 0], x, u_num, u_ref, "u(x, T)", num_label, ref_label, m_u, diag_u)
    _error(  axes[0, 1], x, u_err, "u error  (numerical - reference)")
    _overlay(axes[1, 0], x, v_num, v_ref, "v(x, T)",
             num_label.replace("u ", "v "), ref_label.replace("u ", "v "), m_v, diag_v)
    _error(  axes[1, 1], x, v_err, "v error  (numerical - reference)")

    fig.text(
        0.5, 0.005,
        "Reference = high-resolution FD solution  "
        "(simulate_fitzhugh_nagumo_fd, same grid, dt_ref = dt / ref_factor)",
        ha="center", fontsize=8, color="gray", fontstyle="italic",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"  [verify] Saved plot    -> {output_path}")
    plt.show(block=True)
    plt.close(fig)


def run_fhn(cfg, output_dir, do_save_data, ref_factor):
    print("\n" + "=" * 62)
    print("  FitzHugh-Nagumo -- GRW particle solver vs FD reference")
    print("  Primary : GRW-inspired particle method (experimental)")
    print("  Reference: FD solver, same spatial grid, dt_ref = dt / ref_factor")
    print("=" * 62)

    N  = cfg.num_points
    dt = cfg.time_step
    T  = cfg.total_time
    L  = cfg.domain_size
    D  = cfg.diff_constant
    dx = L / (N - 1)

    print(f"\n  Parameters")
    print(f"    a={cfg.a}, b={cfg.b}, tau={cfg.tau}, D={D}")
    print(f"    T = {T},  dt = {dt}  ({int(T / dt)} GRW steps)")
    print(f"    N = {N} particles,  dx (initial grid) = {dx:.5g},  domain [0, {L}]")
    if D > 0.0:
        cfl = dx**2 / (2.0 * D)
        ref_dt = dt / ref_factor
        print(f"    FD reference CFL limit: dx^2/(2D) = {cfl:.5g}")
        print(f"    FD reference dt = {ref_dt:.5g},  dt/CFL = {ref_dt/cfl:.3f}"
              + ("  OK" if ref_dt <= cfl else "  *** VIOLATED ***"))
        print(f"    (CFL does not apply to the GRW particle solver)")

    print("\n  Running FHN GRW particle solver ...", flush=True)
    x_num, u_num, v_num = _run_fhn_grw(cfg)
    print("  Done.")

    ref_cfg = _fhn_ref_config(cfg, ref_factor)
    print(
        f"\n  Running FHN FD reference  (dt={ref_cfg.time_step:.5g}) ...",
        flush=True,
    )
    x_ref, u_ref, v_ref = _run_fhn_fd(ref_cfg)
    print("  Done.")

    # GRW particles are scattered; interpolate onto the reference (uniform) grid.
    u_num_i = np.interp(x_ref, x_num, u_num)
    v_num_i = np.interp(x_ref, x_num, v_num)

    dx_grid = float(L) / (N - 1)
    m_u = compute_metrics(u_num_i, u_ref, dx_grid)
    m_v = compute_metrics(v_num_i, v_ref, dx_grid)

    # u peak: maximum of u (front of the traveling pulse)
    u_peak_num     = float(np.max(u_num))
    u_peak_ref     = float(np.max(u_ref))
    u_peak_loc_num = float(x_num[np.argmax(u_num)])
    u_peak_loc_ref = float(x_ref[np.argmax(u_ref)])

    # v peak: minimum of v (v dips during the action potential recovery tail)
    v_peak_num     = float(np.min(v_num))
    v_peak_ref     = float(np.min(v_ref))
    v_peak_loc_num = float(x_num[np.argmin(v_num)])
    v_peak_loc_ref = float(x_ref[np.argmin(v_ref)])

    combined_metrics = {
        "u": m_u,
        "v": m_v,
        "u_peak_grw":                   u_peak_num,
        "u_peak_fd_ref":                u_peak_ref,
        "u_peak_error":                 float(abs(u_peak_num - u_peak_ref)),
        "u_peak_error_rel":             float(abs(u_peak_num - u_peak_ref) / u_peak_ref)
                                        if abs(u_peak_ref) > 1e-12 else None,
        "u_peak_location_grw":          u_peak_loc_num,
        "u_peak_location_fd_ref":       u_peak_loc_ref,
        "u_peak_location_diff":         float(abs(u_peak_loc_num - u_peak_loc_ref)),
        "v_peak_grw":                   v_peak_num,
        "v_peak_fd_ref":                v_peak_ref,
        "v_peak_error":                 float(abs(v_peak_num - v_peak_ref)),
        "v_peak_location_grw":          v_peak_loc_num,
        "v_peak_location_fd_ref":       v_peak_loc_ref,
        "v_peak_location_diff":         float(abs(v_peak_loc_num - v_peak_loc_ref)),
        "ref_factor":                   ref_factor,
        "ref_time_step":                ref_cfg.time_step,
    }

    _print_metrics(m_u, "u error metrics  (GRW particle vs FD reference)")
    _print_metrics(m_v, "v error metrics  (GRW particle vs FD reference)")
    print(f"\n  FHN peak diagnostics")
    print(f"    u peak (GRW)            : {_fmt_val(u_peak_num)}")
    print(f"    u peak (FD ref)         : {_fmt_val(u_peak_ref)}")
    print(f"    u peak error            : {_fmt_val(abs(u_peak_num - u_peak_ref))}")
    if combined_metrics["u_peak_error_rel"] is not None:
        print(f"    u peak error (rel)      : {_fmt_val(combined_metrics['u_peak_error_rel'])}")
    print(f"    u peak location (GRW)   : {u_peak_loc_num:.4f}")
    print(f"    u peak location (FD)    : {u_peak_loc_ref:.4f}")
    print(f"    u peak location diff    : {_fmt_val(abs(u_peak_loc_num - u_peak_loc_ref))}")
    print(f"    v peak (GRW)            : {_fmt_val(v_peak_num)}")
    print(f"    v peak (FD ref)         : {_fmt_val(v_peak_ref)}")
    print(f"    v peak error            : {_fmt_val(abs(v_peak_num - v_peak_ref))}")
    print(f"    v peak location (GRW)   : {v_peak_loc_num:.4f}")
    print(f"    v peak location (FD)    : {v_peak_loc_ref:.4f}")
    print(f"    v peak location diff    : {_fmt_val(abs(v_peak_loc_num - v_peak_loc_ref))}")

    num_lbl = f"GRW particle  (N={N}, dt={dt})"
    ref_lbl = f"FD reference  (dt={ref_cfg.time_step:.5g})"
    title = (
        f"FHN: GRW particle vs FD reference  "
        f"(T={T}, a={cfg.a}, b={cfg.b}, tau={cfg.tau}, D={D})\n"
        f"u: {_metrics_str(m_u)}   |   v: {_metrics_str(m_v)}"
    )

    diag_u = dict(
        peak_num=u_peak_num, peak_ref=u_peak_ref,
        peak_loc_num=u_peak_loc_num, peak_loc_ref=u_peak_loc_ref,
    )
    diag_v = dict(
        peak_num=v_peak_num, peak_ref=v_peak_ref,
        peak_loc_num=v_peak_loc_num, peak_loc_ref=v_peak_loc_ref,
    )

    plot_fhn_comparison(
        x_ref, u_num_i, u_ref, v_num_i, v_ref,
        title=title,
        num_label=f"u {num_lbl}",
        ref_label=f"u {ref_lbl}",
        m_u=m_u, m_v=m_v,
        output_path=os.path.join(output_dir, "comparison_plot.png"),
        diag_u=diag_u,
        diag_v=diag_v,
    )
    save_metrics(combined_metrics, os.path.join(output_dir, "metrics.json"))
    if do_save_data:
        path = os.path.join(output_dir, "comparison_data.npz")
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            path,
            x=x_ref,
            u_numerical=u_num_i, u_reference=u_ref, u_error=u_num_i - u_ref,
            v_numerical=v_num_i, v_reference=v_ref, v_error=v_num_i - v_ref,
        )
        print(f"  [verify] Saved data    -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
        choices=["heat", "burgers", "fhn", "all"],
        default="all",
        help="Equation to verify (default: all)",
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

    equations = ["heat", "burgers", "fhn"] if args.equation == "all" else [args.equation]

    for eq in equations:
        cfg_path   = args.config or _DEFAULT_CONFIGS[eq]
        output_dir = args.output_dir or os.path.join("output", "verify", eq)

        print(f"\nLoading config: {cfg_path}")
        loaded_cfg = cfg_module.load_config_from_json(cfg_path)

        if eq == "heat":
            run_heat(loaded_cfg, output_dir, args.save_data)
        elif eq == "burgers":
            run_burgers(loaded_cfg, output_dir, args.save_data, args.ref_factor)
        elif eq == "fhn":
            run_fhn(loaded_cfg, output_dir, args.save_data, args.ref_factor)


if __name__ == "__main__":
    main()
