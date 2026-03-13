#!/usr/bin/env python3
"""
verify_solver.py
================
Verification and benchmark comparison for the GRW solver suite.

For each equation the script runs the primary solver and compares output against
a trusted benchmark:

  heat    -- exact analytical solution (error function), valid for step IC.
             Primary solver: thesis-faithful GRW.

  burgers -- three modes depending on config.burgers_mode:
    cole_hopf_grw (default): Cole-Hopf transform reduces Burgers to a heat equation
        solved by GRW.  For the traveling_wave IC, comparison is against the exact
        analytical traveling wave solution.  For other ICs, a high-resolution FD
        reference is used.
    direct_grw: diagnostic path reproducing thesis Section 5 noise discussion.
        Compares against FD reference; noise in the result is expected and intentional.
    lagrangian_grw: experimental Lagrangian particle method (operator splitting).
        Compares against high-resolution FD reference.

  fhn     -- high-resolution FD reference (ref_factor x smaller dt, same grid).
             Primary solver: experimental GRW-inspired particle method.
             Reference solver: standard finite-difference (simulate_fitzhugh_nagumo_fd).

Burgers and FHN comparisons label exact vs reference solutions explicitly.
The purpose of the error metrics on main is to quantify GRW feasibility and
limitations, not to advertise accuracy.

Usage examples:
  python verify_solver.py                                         # run all three
  python verify_solver.py --equation heat
  python verify_solver.py --equation burgers --config configs/burgers_stationary_shock.json
  python verify_solver.py --equation burgers --config configs/burgers_traveling_wave.json
  python verify_solver.py --equation burgers --config configs/burgers_direct_grw_diagnostic.json
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
    "burgers": "configs/burgers_stationary_shock.json",
    "fhn":     "configs/fhn_grw_steady.json",
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


def exact_burgers_traveling_wave(x, t, nu, x_center=0.0):
    """
    Exact traveling wave solution for Burgers' equation  u_t + u*u_x = nu*u_xx.

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
    Exact stationary-shock solution for Burgers' equation  u_t + u*u_x = nu*u_xx.

    u(x, t) = -A * tanh(A * (x - x_center) / (2 * nu))   for all t >= 0

    where A = amplitude.  This is an exact STATIONARY solution: the nonlinear
    advection term u*u_x and the diffusion term nu*u_xx cancel exactly for any
    amplitude A and viscosity nu.  Verification:

      u_x   = -A^2 / (2*nu) * sech^2(A*(x-xc)/(2*nu))
      u*u_x = A^3 * tanh(...) * sech^2(...) / (2*nu)
      u_xx  = A^3 * tanh(...) * sech^2(...) / (2*nu^2)
      nu*u_xx = A^3 * tanh(...) * sech^2(...) / (2*nu) = u*u_x  QED

    The Cole-Hopf phi0 for this IC satisfies phi0(0) = phi0(L) whenever xc = L/2
    and the domain is symmetric about xc, making the cumulative reconstruction in
    the GRW return exactly to its starting value -- a well-conditioned property for
    the Cole-Hopf GRW benchmark.

    :param x:         array of spatial positions
    :param nu:        kinematic viscosity
    :param x_center:  shock centre; if None, defaults to domain midpoint
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
    Run the active Burgers GRW solver (mode dispatched via config.burgers_mode).

    Returns (x_sorted, u_sorted) on a uniform or near-uniform grid.
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

    Increases num_points by ref_factor and reduces time_step by the same factor.
    The IC is re-interpolated onto the finer grid.
    This config is passed to _run_burgers_fd (not the GRW solver).
    CFL stability conditions apply only to the FD solver.
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
    mode    = getattr(cfg, 'burgers_mode', 'cole_hopf_grw') or 'cole_hopf_grw'
    ic_type = getattr(cfg, 'burgers_ic_type', '') or ''
    use_exact = (mode == 'cole_hopf_grw' and ic_type in ('traveling_wave', 'stationary_shock'))

    # Header
    print("\n" + "=" * 68)
    mode_labels = {
        'cole_hopf_grw': 'Cole-Hopf GRW  (thesis-faithful: Burgers -> heat via Cole-Hopf)',
        'direct_grw':    'Direct GRW  (diagnostic: expected to be noisy, thesis Section 5)',
        'lagrangian_grw':'Lagrangian GRW  (experimental operator splitting)',
    }
    if ic_type == 'stationary_shock':
        ref_label_str = "exact stationary shock"
    elif ic_type == 'traveling_wave' and use_exact:
        ref_label_str = "exact traveling wave"
    else:
        ref_label_str = "high-resolution FD"
    print(f"  Burgers -- {mode_labels.get(mode, mode)}")
    print(f"  Reference : {ref_label_str}")
    print("=" * 68)

    N  = cfg.num_points
    dt = cfg.time_step
    nu = cfg.diff_constant
    T  = cfg.total_time
    L  = cfg.domain_size

    print(f"\n  Parameters")
    print(f"    nu = {nu},  T = {T},  dt = {dt}  ({int(T / dt)} steps)")
    print(f"    N = {N} globs/particles,  domain [0, {L}]")
    if mode == 'cole_hopf_grw':
        print(f"    Cole-Hopf: u -> phi=exp(-Psi/(2*nu)), GRW on phi_x, "
              f"u_out = -2*nu*phi_x/phi")
    elif mode == 'direct_grw':
        print(f"    Direct GRW: v=u_x globs, R=-(u*u_xx/u_x + u_x) reaction statistic")
        print(f"    NOTE: noise in u_x and u_xx is expected and intentional.")
    if use_exact and ic_type == 'stationary_shock':
        x_center = L / 2.0
        print(f"    IC: stationary shock centered at x={x_center}, exact solution available")
        print(f"    u(x,t) = -A*tanh(A*(x-xc)/(2*nu))  [independent of t]")
    elif use_exact and ic_type == 'traveling_wave':
        x_center = L / 2.0
        print(f"    IC: traveling wave centered at x={x_center}, exact solution available")

    print(f"\n  Running Burgers GRW ({mode}) ...", flush=True)
    x_num, u_num = _run_burgers_grw(cfg)
    print("  Done.")

    # Build comparison grid and reference solution.
    x_grid = np.linspace(0.0, L, N)
    dx_grid = L / (N - 1)

    if use_exact and ic_type == 'stationary_shock':
        # Exact stationary-shock solution: u(x,t) = u0(x) for all t.
        # Use the IC values directly as the reference: u_exact = u0 for all t.
        x_center = L / 2.0
        ic_x    = np.array([pos for pos, _ in cfg.initial_conditions])
        ic_vals = np.array([val for _, val in cfg.initial_conditions])
        u_ref   = np.interp(x_grid, ic_x, ic_vals)
        # Amplitude A from config; fall back to max|u0| on finite domain if not set.
        amplitude = (float(cfg.burgers_ic_amplitude)
                     if getattr(cfg, 'burgers_ic_amplitude', None) is not None
                     else float(np.max(np.abs(ic_vals))))
        x_ref = x_grid
        ref_description = (f"exact stationary shock  "
                           f"(A={amplitude:.4g}, width=2*nu/A={2*nu/amplitude:.4g})")
        ref_short = "exact"
        metrics_label = "Error metrics  (Cole-Hopf GRW vs exact stationary shock)"
    elif use_exact and ic_type == 'traveling_wave':
        x_center = L / 2.0
        u_ref  = exact_burgers_traveling_wave(x_grid, T, nu, x_center=x_center)
        x_ref  = x_grid
        ref_description = f"exact traveling wave  (c=1, width=sqrt(nu)={np.sqrt(nu):.4g})"
        ref_short = "exact"
        metrics_label = "Error metrics  (Cole-Hopf GRW vs exact traveling wave)"
    else:
        ref_cfg = _burgers_ref_config(cfg, ref_factor)
        print(
            f"\n  Running Burgers FD reference  "
            f"(N={ref_cfg.num_points}, dt={ref_cfg.time_step:.5g}) ...",
            flush=True,
        )
        x_ref_raw, u_ref_raw = _run_burgers_fd(ref_cfg)
        x_ref  = x_grid
        u_ref  = np.interp(x_grid, x_ref_raw, u_ref_raw)
        ref_description = (f"high-resolution FD  "
                           f"(N={ref_cfg.num_points}, dt={ref_cfg.time_step:.5g})")
        ref_short = "FD reference"
        metrics_label = f"Error metrics  ({mode} GRW vs FD reference)"
        print("  Done.")

    # Interpolate numerical result onto comparison grid.
    u_num_on_grid = np.interp(x_grid, x_num, u_num)
    metrics = compute_metrics(u_num_on_grid, u_ref, dx_grid)

    # Wave / shock location diagnostics.
    u_mid = float(np.mean(u_ref))
    wave_loc_num = float(x_grid[np.argmin(np.abs(u_num_on_grid - u_mid))]) \
        if len(x_grid) else None
    wave_loc_ref = float(x_grid[np.argmin(np.abs(u_ref - u_mid))]) \
        if len(x_grid) else None

    metrics["wave_location_grw"]     = wave_loc_num
    metrics["wave_location_ref"]     = wave_loc_ref
    metrics["wave_location_diff"]    = (
        float(abs(wave_loc_num - wave_loc_ref))
        if (wave_loc_num is not None and wave_loc_ref is not None) else None
    )
    metrics["u_min"]   = float(np.min(u_num_on_grid))
    metrics["u_max"]   = float(np.max(u_num_on_grid))
    metrics["burgers_mode"]    = mode
    metrics["burgers_ic_type"] = ic_type
    if not use_exact:
        metrics["ref_factor"]     = ref_factor
        metrics["ref_num_points"] = ref_cfg.num_points
        metrics["ref_time_step"]  = ref_cfg.time_step

    _print_metrics(metrics, metrics_label)
    if wave_loc_num is not None and wave_loc_ref is not None:
        print(f"\n  Wave / shock location diagnostics")
        print(f"    Wave center (GRW)    : {wave_loc_num:.4f}")
        print(f"    Wave center (ref)    : {wave_loc_ref:.4f}")
        print(f"    Location diff        : {abs(wave_loc_num - wave_loc_ref):.6f}")
    print(f"\n  Solution range")
    print(f"    u_min = {_fmt_val(metrics['u_min'])},  u_max = {_fmt_val(metrics['u_max'])}")

    if mode == 'direct_grw':
        print(f"\n  [direct_grw] Diagnostic interpretation:")
        print(f"    The reaction statistic R = -(u*u_xx/u_x + u_x) amplifies noise")
        print(f"    in u_x and u_xx computed from the particle field.  Large errors")
        print(f"    relative to the FD reference are expected and confirm the thesis")
        print(f"    conclusion that the direct GRW path for Burgers is impractical.")

    # Figure.
    if use_exact and ic_type == 'stationary_shock':
        method_str = "Cole-Hopf GRW"
        num_label  = f"Cole-Hopf GRW  (N={N}, dt={dt})"
        ref_note   = (f"Reference = exact stationary shock  "
                      f"u(x,t) = -A*tanh(A*(x-xc)/(2*nu))  [all t]")
    elif use_exact:
        method_str = "Cole-Hopf GRW"
        num_label  = f"Cole-Hopf GRW  (N={N}, dt={dt})"
        ref_note   = (f"Reference = exact traveling wave  u(x,t) = "
                      f"1 - 2*sqrt(nu)*tanh((x-x0-t)/sqrt(nu))")
    elif mode == 'direct_grw':
        method_str = "Direct GRW (diagnostic)"
        num_label  = f"Direct GRW  (N={N}, dt={dt})  [noisy by design]"
        ref_note   = ("Reference = high-resolution FD solution  "
                      "(simulate_burgers_fd).  Noise is expected for direct GRW.")
    else:
        method_str = f"{mode} GRW"
        num_label  = f"{mode} GRW  (N={N}, dt={dt})"
        ref_note   = "Reference = high-resolution FD solution  (simulate_burgers_fd)"

    title = f"Burgers: {method_str} vs {ref_short}  (T={T}, nu={nu}, N={N})"
    plot_comparison(
        x_grid, u_num_on_grid, u_ref,
        title=title,
        ylabel="u(x, T)",
        num_label=num_label,
        ref_label=ref_description,
        metrics=metrics,
        output_path=os.path.join(output_dir, "comparison_plot.png"),
        ref_note=ref_note,
    )
    save_metrics(metrics, os.path.join(output_dir, "metrics.json"))
    if do_save_data:
        save_npz(x_grid, u_num_on_grid, u_ref,
                 os.path.join(output_dir, "comparison_data"))


# ---------------------------------------------------------------------------
# FitzHugh-Nagumo verification
# Primary solver: GRW-inspired particle method
# ---------------------------------------------------------------------------
# FHN helper: exact traveling wave and GRW runners
# ---------------------------------------------------------------------------

def exact_fhn_traveling_wave(x, t, a, x_center=0.0):
    """
    Exact traveling-wave solution for the thesis scalar FHN equation.

    u(x, t) = 1 / (1 + exp(-(x + theta*t - x_center) / 2))
    theta    = sqrt(2) * (0.5 - a)

    The wave front (u = 0.5) starts at x = x_center at t = 0 and moves at
    speed -theta in the x-direction (leftward for a < 0.5, rightward for a > 0.5).

    :param x:        array-like, spatial coordinates
    :param t:        float, time
    :param a:        float, FHN threshold parameter
    :param x_center: float, initial position of the wave center (u = 0.5)
    :return:         ndarray, exact u values at positions x, time t
    """
    theta = np.sqrt(2.0) * (0.5 - a)
    xi    = np.asarray(x, dtype=float) + theta * float(t) - x_center
    return 1.0 / (1.0 + np.exp(-xi / 2.0))


def _run_fhn_grw_scalar(cfg, total_time_override=None):
    """
    Run the thesis scalar GRW FHN solver.

    Returns (x_sorted, u_reconstructed) on a uniform output grid.
    u is reconstructed from the sorted glob list via cumulative summation.

    :param cfg:                 SimulationConfig
    :param total_time_override: if not None, use this total time instead of cfg.total_time
    :return:                    (x_grid, u_grid) arrays on a uniform output grid
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

    result = simulate_fitzhugh_nagumo(globs, run_cfg)

    x_pos = np.array([g['position'] for g in result])
    w_val = np.array([float(g['value']) for g in result])
    order = np.argsort(x_pos)
    x_s   = x_pos[order]
    w_s   = w_val[order]

    # Reconstruct u on a uniform grid by binning weights.
    L      = cfg.domain_size
    n_grid = max(200, cfg.num_points)
    x_grid = np.linspace(0.0, L, n_grid)
    u_grid = np.zeros(n_grid)

    # For each output x, u = sum of weights at positions <= x.
    for j, xj in enumerate(x_grid):
        u_grid[j] = float(np.sum(w_s[x_s <= xj]))

    return x_grid, u_grid


def _run_fhn_grw_legacy(cfg):
    """
    Run the legacy two-component GRW particle solver for FHN.

    Returns (x_sorted, u_sorted, v_sorted) on a scattered grid.
    """
    globs = [{'position': float(pos), 'value': list(val)}
             for pos, val in cfg.initial_conditions]
    result = simulate_fitzhugh_nagumo(globs, cfg)
    x  = np.array([g['position'] for g in result])
    uv = np.array([g['value']    for g in result])
    order = np.argsort(x)
    return x[order], uv[order, 0], uv[order, 1]


def _run_fhn_fd(cfg):
    """
    Run the FD reference solver for FHN on a fixed uniform grid.

    Returns (x_sorted, u_sorted, v_sorted) where x is the uniform grid.
    """
    globs = [{'position': float(pos), 'value': list(val)}
             for pos, val in cfg.initial_conditions]
    result = simulate_fitzhugh_nagumo_fd(globs, cfg)
    x  = np.array([g['position'] for g in result])
    uv = np.array([g['value']    for g in result])
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


def plot_fhn_scalar_grw(
    snap_times, snap_exact, snap_num,
    x_grid, title, ic_label,
    output_path,
    metrics_final,
):
    """
    Multi-time verification figure for the thesis scalar FHN GRW.

    Layout: 2 rows x 2 columns (up to 4 time snapshots).
      Left column:  numerical GRW vs exact traveling wave
      Right column: pointwise error

    :param snap_times:    list of floats, snapshot times
    :param snap_exact:    list of 1d arrays, exact solution at each snap time
    :param snap_num:      list of 1d arrays, GRW reconstruction at each snap time
    :param x_grid:        1d array, common output grid
    :param title:         str, figure suptitle
    :param ic_label:      str, IC type label for annotations
    :param output_path:   str, file path to save the PNG
    :param metrics_final: dict, error metrics at the final snapshot time
    """
    n_snap = len(snap_times)
    n_rows = (n_snap + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows + 0.8))
    if n_rows == 1:
        axes = np.array([axes])
    fig.suptitle(title, fontsize=10, y=0.99)

    colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
    col_num = colors[1] if len(colors) > 1 else 'crimson'
    col_exa = 'black'

    for idx, (t_snap, u_ex, u_gr) in enumerate(
            zip(snap_times, snap_exact, snap_num)):
        row = idx // 2
        col = idx % 2
        ax  = axes[row, col]

        err = u_gr - u_ex
        m   = compute_metrics(u_gr, u_ex, float(x_grid[1] - x_grid[0]))

        ax.plot(x_grid, u_ex, color=col_exa, linewidth=2,   linestyle='-',  label='exact')
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


def plot_fhn_comparison(
    x, u_num, u_ref, v_num, v_ref,
    title, num_label, ref_label, m_u, m_v,
    output_path,
    diag_u=None,
    diag_v=None,
):
    """
    Four-panel figure for legacy FHN two-component verification.

    Layout (2 rows x 2 columns):
      [0,0] u: numerical vs reference overlay
      [0,1] u: pointwise error
      [1,0] v: numerical vs reference overlay
      [1,1] v: pointwise error
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
    plt.close(fig)


def run_fhn(cfg, output_dir, do_save_data, ref_factor):
    """
    FHN verification dispatcher.

    Scalar GRW ICs (steady_solution, nonsmooth, discontinuous):
      Compare the thesis scalar GRW result to the exact traveling-wave solution
      at multiple time snapshots (t = 0, T/3, 2T/3, T).

    Legacy two-component IC:
      Compare the two-component GRW particle solver to a high-resolution FD
      reference (same spatial grid, dt reduced by ref_factor).
    """
    ic_type = getattr(cfg, 'fhn_ic_type', '') or ''
    scalar_path = ic_type in ('steady_solution', 'nonsmooth', 'discontinuous', 'scalar_grw')

    if scalar_path:
        _run_fhn_scalar(cfg, output_dir, do_save_data)
    else:
        _run_fhn_legacy(cfg, output_dir, do_save_data, ref_factor)


def _run_fhn_scalar(cfg, output_dir, do_save_data):
    """
    Thesis scalar GRW FHN verification.

    Runs the GRW solver at multiple time snapshots and compares each to the
    exact traveling-wave solution:
      u(x, t) = 1 / (1 + exp(-(x + theta*t - x_center) / 2))
      theta   = sqrt(2) * (0.5 - a)
    """
    print("\n" + "=" * 62)
    print("  FitzHugh-Nagumo -- thesis scalar GRW vs exact solution")
    print("  Primary : GRW gradient-side method (thesis Chapter 4)")
    print("  Reference: exact traveling-wave solution (analytic)")
    print("=" * 62)

    N      = cfg.num_points
    dt     = cfg.time_step
    T      = cfg.total_time
    L      = cfg.domain_size
    D      = cfg.diff_constant
    a_     = cfg.a if cfg.a is not None else 0.25
    ic_type = getattr(cfg, 'fhn_ic_type', '') or ''
    theta  = np.sqrt(2.0) * (0.5 - a_)

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
    snap_num   = []
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
            print(f"\n  Running GRW to t = {t_snap:.3g} ...", flush=True)
            x_g, u_g = _run_fhn_grw_scalar(cfg, total_time_override=t_snap)
            print("  Done.")

        if x_grid_out is None:
            x_grid_out = x_g
        snap_num.append(u_g)

    if x_grid_out is None:
        x_grid_out = np.linspace(0.0, L, max(200, N))

    # All snapshots on the same x_grid; re-interpolate if needed.
    for i in range(len(snap_num)):
        if len(snap_num[i]) != len(x_grid_out):
            snap_num[i]   = np.interp(x_grid_out, np.linspace(0.0, L, len(snap_num[i])),  snap_num[i])
        if len(snap_exact[i]) != len(x_grid_out):
            snap_exact[i] = np.interp(x_grid_out, np.linspace(0.0, L, len(snap_exact[i])), snap_exact[i])

    dx_g  = float(x_grid_out[1] - x_grid_out[0])
    m_fin = compute_metrics(snap_num[-1], snap_exact[-1], dx_g)

    # Front location: where u = 0.5 (zero-crossing of u - 0.5).
    def _front_loc(u, x):
        diff = u - 0.5
        idx  = np.where(np.diff(np.sign(diff)))[0]
        if len(idx) == 0:
            return float(x[np.argmin(np.abs(diff))])
        i = idx[0]
        if u[i+1] == u[i]:
            return float(x[i])
        return float(x[i] + (0.5 - u[i]) / (u[i+1] - u[i]) * (x[i+1] - x[i]))

    front_grw  = _front_loc(snap_num[-1],   x_grid_out)
    front_ex   = _front_loc(snap_exact[-1], x_grid_out)
    front_diff = abs(front_grw - front_ex)

    combined_metrics = {
        "ic_type":          ic_type,
        "a":                float(a_),
        "theta":            float(theta),
        "x_center":         float(x_center),
        "T":                float(T),
        "N_globs":          int(N),
        "D":                float(D),
        "dt":               float(dt),
        "u_final": m_fin,
        "front_location_grw":   front_grw,
        "front_location_exact": front_ex,
        "front_location_diff":  front_diff,
    }

    _print_metrics(m_fin, f"u error metrics  (GRW vs exact, t = {T})")
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


def _run_fhn_legacy(cfg, output_dir, do_save_data, ref_factor):
    """
    Legacy two-component FHN verification: GRW particle vs FD reference.
    """
    print("\n" + "=" * 62)
    print("  FitzHugh-Nagumo -- legacy two-component GRW vs FD reference")
    print("  Primary : two-component GRW particle method (legacy)")
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
    print(f"    N = {N} particles,  dx = {dx:.5g},  domain [0, {L}]")

    print("\n  Running FHN two-component GRW ...", flush=True)
    x_num, u_num, v_num = _run_fhn_grw_legacy(cfg)
    print("  Done.")

    ref_cfg = _fhn_ref_config(cfg, ref_factor)
    print(
        f"\n  Running FHN FD reference  (dt={ref_cfg.time_step:.5g}) ...",
        flush=True,
    )
    x_ref, u_ref, v_ref = _run_fhn_fd(ref_cfg)
    print("  Done.")

    u_num_i = np.interp(x_ref, x_num, u_num)
    v_num_i = np.interp(x_ref, x_num, v_num)

    dx_grid = float(L) / (N - 1)
    m_u = compute_metrics(u_num_i, u_ref, dx_grid)
    m_v = compute_metrics(v_num_i, v_ref, dx_grid)

    u_peak_num     = float(np.max(u_num))
    u_peak_ref     = float(np.max(u_ref))
    u_peak_loc_num = float(x_num[np.argmax(u_num)])
    u_peak_loc_ref = float(x_ref[np.argmax(u_ref)])

    v_peak_num     = float(np.min(v_num))
    v_peak_ref     = float(np.min(v_ref))
    v_peak_loc_num = float(x_num[np.argmin(v_num)])
    v_peak_loc_ref = float(x_ref[np.argmin(v_ref)])

    combined_metrics = {
        "u": m_u,
        "v": m_v,
        "u_peak_grw":           u_peak_num,
        "u_peak_fd_ref":        u_peak_ref,
        "u_peak_error":         float(abs(u_peak_num - u_peak_ref)),
        "u_peak_error_rel":     float(abs(u_peak_num - u_peak_ref) / u_peak_ref)
                                if abs(u_peak_ref) > 1e-12 else None,
        "u_peak_location_grw":  u_peak_loc_num,
        "u_peak_location_ref":  u_peak_loc_ref,
        "u_peak_location_diff": float(abs(u_peak_loc_num - u_peak_loc_ref)),
        "v_peak_grw":           v_peak_num,
        "v_peak_fd_ref":        v_peak_ref,
        "v_peak_error":         float(abs(v_peak_num - v_peak_ref)),
        "v_peak_location_grw":  v_peak_loc_num,
        "v_peak_location_ref":  v_peak_loc_ref,
        "v_peak_location_diff": float(abs(v_peak_loc_num - v_peak_loc_ref)),
        "ref_factor":           ref_factor,
        "ref_time_step":        ref_cfg.time_step,
    }

    _print_metrics(m_u, "u error metrics  (GRW particle vs FD reference)")
    _print_metrics(m_v, "v error metrics  (GRW particle vs FD reference)")
    print(f"\n  FHN peak diagnostics")
    print(f"    u peak (GRW) : {_fmt_val(u_peak_num)},  loc = {u_peak_loc_num:.4f}")
    print(f"    u peak (FD)  : {_fmt_val(u_peak_ref)},  loc = {u_peak_loc_ref:.4f}")
    print(f"    u peak diff  : {_fmt_val(abs(u_peak_num - u_peak_ref))}")
    print(f"    v peak (GRW) : {_fmt_val(v_peak_num)},  loc = {v_peak_loc_num:.4f}")
    print(f"    v peak (FD)  : {_fmt_val(v_peak_ref)},  loc = {v_peak_loc_ref:.4f}")
    print(f"    v peak diff  : {_fmt_val(abs(v_peak_num - v_peak_ref))}")

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
