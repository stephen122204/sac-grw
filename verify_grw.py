"""Mathematical verification of the heat GRW solver.

Runs a heat step IC (Dirichlet–Dirichlet BCs) and checks the reconstruction
against the exact infinite-domain solution
    u(x, T) = uL + (uR - uL) * 0.5 * (1 + erf((x - x0) / (2 * sqrt(alpha * T)))),
valid here because [0, L] is many diffusion lengths wide
(L/2 / sqrt(2*alpha*T) >> 1).  Reports three checks: glob statistics,
weight conservation (sum of glob values = jump_height), and L2/max
reconstruction error.  Optional argv[1] = config path.
"""

import os
import sys
from math import erf as _erf

import numpy as np

import matplotlib
if sys.platform.startswith("darwin"):
    matplotlib.use("MacOSX")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as config_module
from simulation import simulate_heat_equation


def _erf_vec(x):
    return np.array([_erf(float(v)) for v in np.asarray(x).ravel()]).reshape(np.asarray(x).shape)


def exact_heat_step(x, T, x0, uL, uR, alpha):
    """Exact infinite-domain heat solution for a Heaviside step IC:
    u(x, T) = uL + (uR - uL) * 0.5 * (1 + erf((x - x0) / (2 * sqrt(alpha * T)))).
    """
    return uL + (uR - uL) * 0.5 * (1.0 + _erf_vec((x - x0) / (2.0 * np.sqrt(alpha * T))))


def run_verification(config_path, nbins=300):
    print(f"\n{'='*60}")
    print(f"  Heat GRW — Mathematical Verification")
    print(f"  Config: {config_path}")
    print(f"{'='*60}\n")

    cfg = config_module.load_config_from_json(config_path)

    alpha = cfg.diff_constant
    T = cfg.total_time
    L = cfg.domain_size
    N = cfg.num_points
    dt = cfg.time_step
    uL = float(cfg.boundary_conditions['LEFT']['value'])
    uR = float(cfg.boundary_conditions['RIGHT']['value'])

    heat_ic = getattr(cfg, '_raw_heat_ic', None)
    x0 = L / 2.0
    jump_height = uR - uL

    print(f"  Parameters")
    print(f"    alpha (diff_constant) = {alpha}")
    print(f"    T (total_time) = {T}")
    print(f"    dt (time_step) = {dt}  ({int(T/dt)} steps)")
    print(f"    N (num_points) = {N}")
    print(f"    Domain = [0, {L}]")
    print(f"    BCs = {cfg.boundary_conditions['LEFT']['type']}"
          f"–{cfg.boundary_conditions['RIGHT']['type']}")
    print(f"    Step at x0 = {x0},  jump_height = {jump_height}")

    expected_std = np.sqrt(2.0 * alpha * T)
    print(f"\n  Theoretical glob std after T: sqrt(2*alpha*T) = {expected_std:.6f}")

    print(f"\n  Running simulation...", flush=True)
    initial_globs = [{'position': pos, 'value': val} for pos, val in cfg.initial_conditions]
    results = simulate_heat_equation(initial_globs, cfg)
    print(f"  Done.\n")

    positions = np.array([g['position'] for g in results])
    values = np.array([g['value']    for g in results])

    # Check 1 — Glob statistics
    mean_pos = float(np.mean(positions))
    std_pos = float(np.std(positions))
    total_wt = float(np.sum(values))

    print(f"  [1] Glob statistics")
    print(f"      Mean position : {mean_pos:.6f}   (expected ≈ {x0:.4f})")
    print(f"      Std  position : {std_pos:.6f}   (expected ≈ {expected_std:.6f})")
    mean_err = abs(mean_pos - x0)
    std_err = abs(std_pos - expected_std)
    std_tol = 3.0 * expected_std / np.sqrt(N)   # 3-sigma Monte Carlo tolerance on std
    print(f"      |mean - x0|   : {mean_err:.6f}  {'OK' if mean_err < 3*expected_std/np.sqrt(N) else 'WARN'}")
    print(f"      |std - theory|: {std_err:.6f}  {'OK' if std_err < std_tol else 'WARN'}  (3σ MC tol = {std_tol:.6f})")

    # Check 2 — Weight conservation
    print(f"\n  [2] Weight conservation")
    print(f"      sum(values) = {total_wt:.8f}   (expected = {jump_height:.8f})")
    print(f"      Relative error: {abs(total_wt - jump_height) / max(abs(jump_height), 1e-12):.2e}"
          f"  {'OK' if abs(total_wt - jump_height) < 1e-10 else 'WARN (Neumann reflections changed total weight)'}")

    # Check 3 — Reconstruction vs. exact solution
    edges = np.linspace(0.0, L, nbins + 1)
    bin_weights, _ = np.histogram(positions, bins=edges, weights=values)
    u_grw = uL + np.cumsum(bin_weights)
    centers = 0.5 * (edges[:-1] + edges[1:])

    u_exact = exact_heat_step(centers, T, x0, uL, uR, alpha)

    l2_err = float(np.sqrt(np.mean((u_grw - u_exact) ** 2)))
    max_err = float(np.max(np.abs(u_grw - u_exact)))

    print(f"\n  [3] Reconstruction vs. exact error function")
    print(f"      L2  error : {l2_err:.6f}")
    print(f"      Max error : {max_err:.6f}")
    mc_noise = jump_height / np.sqrt(N)
    print(f"      Expected MC noise O(1/sqrt(N)) ≈ {mc_noise:.6f}")
    print(f"      Ratio L2/noise    : {l2_err/mc_noise:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(centers, u_exact, 'k-', linewidth=2, label="Exact (error function)")
    ax.plot(centers, u_grw, 'r--', linewidth=1.5, label=f"GRW  (N={N})")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, T)")
    ax.set_title(f"Heat GRW vs. Exact  (T={T}, α={alpha})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

    ax = axes[1]
    ax.plot(centers, u_grw - u_exact, 'b-', linewidth=1.5)
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel("x")
    ax.set_ylabel("u_GRW − u_exact")
    ax.set_title(f"Pointwise error  (L2={l2_err:.4f}, max={max_err:.4f})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = "output/verify_grw_vs_exact.png"
    plt.savefig(out, dpi=150)
    print(f"\n  Saved verification plot -> {out}")
    plt.show(block=True)

    print(f"\n{'='*60}\n")
    return {"l2_error": l2_err, "max_error": max_err,
            "mean_pos": mean_pos, "std_pos": std_pos, "total_weight": total_wt}


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) >= 2 else "configs/heat_step_dirichlet.json"
    run_verification(cfg_path)
