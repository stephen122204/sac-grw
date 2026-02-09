# utils.py
# ------------------------------------------------------------
# GUI backend BEFORE importing pyplot
import sys
import os
import matplotlib

if sys.platform.startswith("darwin"):
    matplotlib.use("MacOSX")   # macOS native backend
else:
    matplotlib.use("TkAgg")    # cross-platform fallback

import matplotlib.pyplot as plt
import numpy as np

# Import module-level config as a fallback (we prefer the runtime cfg passed in)
try:
    import config as config_module
except Exception:
    config_module = None
    print("[plot] Warning: could not import module-level config; will rely on cfg param.")

os.makedirs("output", exist_ok=True)
# ------------------------------------------------------------


def _safe_str(x):
    try:
        return (x or "").strip().lower()
    except Exception:
        return ""


def _get_attr(obj, keys, default=None):
    """Return first existing attribute among possible names on obj."""
    if obj is None:
        return default
    for k in keys:
        if hasattr(obj, k):
            return getattr(obj, k)
    return default


def _bc_info(cfg=None):
    """
    Read BC info from cfg.boundary_conditions (preferred).
    Fall back to legacy attribute names if needed.
    """
    source = cfg if cfg is not None else config_module

    L = float(_get_attr(source, ["domain_size", "L", "length"], 1.0))

    # NEW: handle dict-based boundary_conditions (your JSON path)
    bc = _get_attr(source, ["boundary_conditions"], None)
    if isinstance(bc, dict) and "LEFT" in bc and "RIGHT" in bc:
        ltype = _safe_str(bc["LEFT"].get("type", ""))
        rtype = _safe_str(bc["RIGHT"].get("type", ""))
        lval  = float(bc["LEFT"].get("value", 0.0))
        rval  = float(bc["RIGHT"].get("value", 0.0))
        print(f"[plot] BCs detected -> left: {ltype} ({lval}), right: {rtype} ({rval}), L={L}")
        return L, ltype, lval, rtype, rval

    # FALLBACK: legacy attribute-style configs (kept for compatibility)
    ltype = _safe_str(_get_attr(
        source,
        ["left_boundary_condition_type", "left_boundary_type", "left_bc_type", "left_bc"],
        ""
    ))
    rtype = _safe_str(_get_attr(
        source,
        ["right_boundary_condition_type", "right_boundary_type", "right_bc_type", "right_bc"],
        ""
    ))
    lval = float(_get_attr(source, ["left_boundary_value", "left_bc_value", "left_value"], 0.0))
    rval = float(_get_attr(source, ["right_boundary_value", "right_bc_value", "right_value"], 0.0))

    print(f"[plot] BCs detected -> left: {ltype} ({lval}), right: {rtype} ({rval}), L={L}")
    return L, ltype, lval, rtype, rval

def _both_dirichlet(cfg=None):
    _, ltype, _, rtype, _ = _bc_info(cfg)
    return ltype.startswith("dirichlet") and rtype.startswith("dirichlet")


def _both_neumann0(cfg=None):
    _, ltype, lval, rtype, rval = _bc_info(cfg)
    return (
        ltype.startswith("neumann")
        and rtype.startswith("neumann")
        and abs(float(lval)) == 0.0
        and abs(float(rval)) == 0.0
    )


def _extract_positions(results):
    n = len(results)
    print(f"[plot] number of globs: {n}")
    if n == 0:
        print("[plot] No results to plot.")
        return None

    try:
        positions = np.array([g["position"] for g in results], dtype=float)
    except Exception as e:
        print(f"[plot] Failed to read positions from results: {e}")
        return None

    if positions.size == 0 or not np.isfinite(positions).any():
        print("[plot] Positions are empty or non-finite.")
        return None

    return positions


# ---------------------- HEAT: density (default) ---------------------- #
def _plot_heat_density(results, title_extra=""):
    """Plot the spatial density (histogram) of particle positions (≈ u_x)."""
    positions = _extract_positions(results)
    if positions is None:
        return False

    xmin, xmax = float(np.nanmin(positions)), float(np.nanmax(positions))
    print(f"[plot] x-range: {xmin:.6g} .. {xmax:.6g}")
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        print("[plot] x-range not finite; aborting plot.")
        return False
    if xmin == xmax:
        xmin -= 1e-6
        xmax += 1e-6

    n = positions.size
    nbins = int(max(25, min(200, n // 20)))

    counts, edges = np.histogram(
        positions, bins=nbins, range=(xmin, xmax), density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure()
    plt.plot(centers, counts, linewidth=2)
    plt.xlabel("Position x")
    plt.ylabel("Density (≈ temperature)")
    title = "Heat via random-walk density"
    if title_extra:
        title += f" {title_extra}"
    plt.title(title)
    plt.grid(True, alpha=0.3)

    ymax = (np.nanmax(counts) if np.isfinite(counts).any() else 1.0) * 1.1
    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)

    plt.tight_layout()
    out = "output/heat_density.png"
    plt.savefig(out, dpi=150)
    print(f"[plot] Saved PNG -> {out}")
    plt.show(block=True)
    return True


# ---------------------- HEAT: Dirichlet field ----------------------- #
def _plot_heat_dirichlet_as_field(results, cfg=None):
    """
    For Dirichlet–Dirichlet BCs, integrate the (empirical) gradient density to get u(x,t),
    then enforce boundary values on [0, L].
    """
    positions = _extract_positions(results)
    if positions is None:
        return False

    L, _, uL, _, uR = _bc_info(cfg)
    L = float(L)
    uL = float(uL)
    uR = float(uR)
    du = uR - uL

    # fixed domain bins so integration is on [0, L]
    n = positions.size
    nbins = int(max(100, min(400, n // 10)))
    edges = np.linspace(0.0, L, nbins + 1)

    counts, _ = np.histogram(positions, bins=edges)   # counts per bin
    dx = edges[1] - edges[0]

    # Convert counts to a density on [0, L]
    density = counts / (np.sum(counts) * dx) if np.sum(counts) > 0 else np.zeros_like(counts, dtype=float)

    # Integrate density to get a monotone field, then scale to match Dirichlet endpoints
    F = np.cumsum(density) * dx  # in [0,1] approximately
    if F.size == 0:
        return False

    # Normalize cumulative to [0, 1]
    Fmin, Fmax = float(np.min(F)), float(np.max(F))
    if Fmax - Fmin > 1e-12:
        Fnorm = (F - Fmin) / (Fmax - Fmin)
    else:
        Fnorm = np.linspace(0.0, 1.0, F.size)

    u = uL + du * Fnorm

    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure()
    plt.plot(centers, u, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Heat: reconstructed field u(x,t) (Dirichlet–Dirichlet)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, L)

    plt.tight_layout()
    out = "output/heat_field_dirichlet.png"
    plt.savefig(out, dpi=150)
    print(f"[plot] Saved PNG -> {out}")
    plt.show(block=True)
    return True


# ---------------------- BURGERS: u(x,t) ---------------------- #
def _plot_burgers_field(results):
    positions = _extract_positions(results)
    if positions is None:
        return False

    try:
        u = np.array([g["value"][0] for g in results], dtype=float)
    except Exception as e:
        print(f"[plot] Failed to read Burgers u from results: {e}")
        return False

    order = np.argsort(positions)
    xs = positions[order]
    us = u[order]

    plt.figure()
    plt.plot(xs, us, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Burgers: u(x,t)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "output/burgers_u.png"
    plt.savefig(out, dpi=150)
    print(f"[plot] Saved PNG -> {out}")
    plt.show(block=True)
    return True


# ---------------------- FHN: (u,v)(x,t) ---------------------- #
def _plot_fhn_fields(results):
    positions = _extract_positions(results)
    if positions is None:
        return False

    try:
        uv = np.array([g["value"] for g in results], dtype=float)
        u = uv[:, 0]
        v = uv[:, 1]
    except Exception as e:
        print(f"[plot] Failed to read FHN (u,v) from results: {e}")
        return False

    order = np.argsort(positions)
    xs = positions[order]
    u = u[order]
    v = v[order]

    plt.figure()
    plt.plot(xs, u, linewidth=2, label="u")
    plt.plot(xs, v, linewidth=2, label="v")
    plt.xlabel("x")
    plt.ylabel("state")
    plt.title("FitzHugh–Nagumo: u(x,t), v(x,t)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = "output/fhn_uv.png"
    plt.savefig(out, dpi=150)
    print(f"[plot] Saved PNG -> {out}")
    plt.show(block=True)
    return True


def plot_results(results, equation_type: str, cfg=None):
    """
    Plotting entry point used by main.py.
    - Heat + Dirichlet-Dirichlet  → plot field u(x,t) via cumulative integral
    - Heat + (Neumann or mixed)   → plot density (≈ u_x)
    - Burgers                     → plot u(x,t)
    - FitzHugh–Nagumo             → plot u(x,t) and v(x,t)
    `cfg` is the runtime config object from get_user_input(); we use it to read BCs.
    """
    eq = (equation_type or "").strip().lower()

    if eq == "heat":
        if _both_dirichlet(cfg):
            print("[plot] Detected Dirichlet–Dirichlet BCs → plotting FIELD (u).")
            ok = _plot_heat_dirichlet_as_field(results, cfg)
        else:
            extra = "(Neumann-0 reflecting)" if _both_neumann0(cfg) else ""
            print("[plot] Non-Dirichlet or mixed BCs → plotting DENSITY (u_x).")
            ok = _plot_heat_density(results, title_extra=extra)

        if not ok:
            print("[plot] Heat plot failed or had no data.")
        return

    if eq in {"fitzhugh-nagumo", "fitzhugh–nagumo", "fitzhugh", "nagumo"}:
        print("[plot] Plotting FitzHugh–Nagumo fields...")
        ok = _plot_fhn_fields(results)
        if not ok:
            print("[plot] FHN plot failed or had no data.")
        return

    if eq in {"burgers", "burger", "burgers'"}:
        print("[plot] Plotting Burgers field...")
        ok = _plot_burgers_field(results)
        if not ok:
            print("[plot] Burgers plot failed or had no data.")
        return

    print(f"[plot] Unknown equation type for plotting: {equation_type!r}")