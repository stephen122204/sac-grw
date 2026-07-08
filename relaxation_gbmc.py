"""BPC Relaxation + Gradient Brownian Monte Carlo (GBMC) for viscous Burgers.

PDE:  u_t + u * u_x = nu * u_xx

Domain mode: WHOLE LINE only.  Particles move freely on the real line.
No boundary reflection is applied.  An output window [0, L] is used for
reporting; particles outside that window contribute correctly to the raw
cumulative-sum reconstruction and are counted in the outside-window report.

Gradient particles represent w = u_x via the signed empirical measure:

    w^N(x, t) = sum_i  m_i * delta(x - X_i(t))

The field u is reconstructed from sorted particles by the RAW cumulative sum:

    u(x, t) = u_{-inf} + sum_{i : X_i <= x}  m_i

Supported initial condition: STATIONARY SHOCK only.

    u(x) = -A * tanh(A * (x - x_c) / (2 * nu))

Quantile particle initialisation (recovers exact IC in the particle limit):

    r_i = (i - 1/2) / N   for i = 1 .. N
    X_i = x_c + (2*nu/A) * arctanh(2*r_i - 1)
    m_i = -2*A / N          (all equal; sum(m) = -2A = u(+inf) - u(-inf))
    u_{-inf} = A

Each particle carries a persistent velocity label  V_i in {-a, +a}.

The relaxation speed a is a FIXED positive scalar read from
config.relaxation_speed_a.  ValueError is raised if the field is absent or
None.  The strict subcharacteristic condition  a > A  (hence a > max|u_i|)
must hold at every step; RuntimeError is raised immediately on violation.

Per-step Lie splitting (A then B):

  A — BPC instantaneous-equilibrium relaxation transport:
    1. Transport:  X_i <- X_i + V_i * dt   (no reflection: whole-line mode)
    2. Sort (X, m, V) together by X.
    3. Reconstruct:  u_i = u_{-inf} + cumsum(m)[i]
    4. Subcharacteristic check:  raise RuntimeError if max|u_i| >= a.
    5. Validate p_plus: raise RuntimeError if any p_i^+ outside (0, 1).
    6. Stochastic instantaneous-equilibrium switching:
         p_i^+ = (a + u_i) / (2*a)
         V_i   = +a  if U < p_i^+,  else -a,   U ~ Uniform(0,1)
       E[V_i | u_i] = u_i (recovers Burgers' characteristic in expectation).

  B — Brownian diffusion:
    7. Displace: X_i <- X_i + Normal(0, sqrt(2*nu*dt))  (no reflection)

Velocity labels are initialised once before the loop by steps 2-6 applied
to the initial sorted configuration (transport step 1 is skipped at t=0).

Output: u on a uniform grid [0, L] via the RAW cumulative-sum reconstruction.
No smoothing is applied to the primary output.  A smoothed profile is
available via _reconstruct_u_on_grid() for diagnostic plots only.

RNG: a private numpy.random.Generator is used throughout.  If config.seed is
not None it is seeded deterministically; otherwise a fresh non-seeded
generator is used (independent of the global numpy random state).

References:
  Bouchut & Perthame (1993) — kinetic relaxation for conservation laws.
  Bossy & Talay (1997) — GBMC for McKean SDEs / Burgers via McKean–Vlasov.
  Méléard (1996) — propagation of chaos, particle approximations.
"""

import os

import numpy as np


def _reflect(x, m, bc_left, bc_right, L, n_passes=4):
    """
    Reflect particles at x = 0 (left) and x = L (right).

    Dirichlet: mirror position, keep mass.
    Neumann:   mirror position, negate mass.

    Applied n_passes times to handle particles that bounce more than once
    (large sigma relative to domain size).
    """
    for _ in range(n_passes):
        ml = x < 0.0
        if np.any(ml):
            x[ml] = -x[ml]
            if bc_left == 'neumann':
                m[ml] = -m[ml]
        mr = x > L
        if np.any(mr):
            x[mr] = 2.0 * L - x[mr]
            if bc_right == 'neumann':
                m[mr] = -m[mr]
    return x, m


def _reconstruct_u_on_grid(x_p, m_p, u_left, x_out, sigma_bins=4):
    """
    Reconstruct u on a uniform output grid from sorted gradient particles.

    Algorithm:
      1. Bin particle masses onto the output grid (floor assignment).
      2. Apply boundary-corrected Gaussian smoothing to suppress shot noise.
         The kernel is divided by its effective support at each bin so that
         boundary truncation does not bias the gradient near x=0 and x=L.
      3. Enforce the correct total: for near-zero-total (symmetric IC) subtract
         the mean; otherwise proportional rescale.
      4. u_out[j] = u_left + cumsum(smoothed_bins)[j]

    :param x_p: 1d array, sorted particle positions
    :param m_p: 1d array, particle masses (same order as x_p)
    :param u_left: float, left-boundary anchor value u(0)
    :param x_out: 1d array, uniform output grid
    :param sigma_bins: float, Gaussian smoothing width in units of bins (0 = off)
    :return: 1d array, reconstructed u values on x_out
    """
    N = len(x_out)
    if N < 2:
        return np.full(N, float(u_left))

    x0_grid = float(x_out[0])
    dx = float(x_out[1] - x_out[0])

    # Bin masses
    bin_sums = np.zeros(N)
    idx = np.clip(np.floor((x_p - x0_grid) / dx).astype(int), 0, N - 1)
    np.add.at(bin_sums, idx, m_p)

    total_raw = float(bin_sums.sum())

    # Gaussian smoothing with boundary correction
    if sigma_bins > 0:
        kw = int(4.0 * sigma_bins) + 1
        kx = np.arange(-kw, kw + 1, dtype=float)
        kernel = np.exp(-0.5 * (kx / sigma_bins) ** 2)
        kernel /= kernel.sum()
        raw_conv = np.convolve(bin_sums, kernel, mode='same')
        kernel_norm = np.convolve(np.ones(N, dtype=float), kernel, mode='same')
        bin_sums_s = raw_conv / np.maximum(kernel_norm, 1e-12)
    else:
        bin_sums_s = bin_sums.copy()

    # Restore correct total
    near_zero_tol = 1e-6 * max(float(np.abs(bin_sums).max()), 1e-30)
    if abs(total_raw) < near_zero_tol:
        # Symmetric IC: enforce sum = 0 exactly
        bin_sums_s -= bin_sums_s.mean()
    else:
        total_smooth = float(bin_sums_s.sum())
        if abs(total_smooth) > 1e-30:
            bin_sums_s *= total_raw / total_smooth

    # Cumulative sum -> u field
    u_out = u_left + np.cumsum(bin_sums_s)
    return u_out


def _save_rbmc_diagnostics(diag_dir, mass_t, u_min_t, u_max_t, snaps,
                            x_out, u_final, u_left, dt, n_steps):
    """Save a 2x2 diagnostic figure for the relaxation GBMC run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(diag_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Relaxation GBMC diagnostics", fontsize=11)
    ax = axes.flat

    t_arr = np.arange(1, len(mass_t) + 1) * dt

    # [0] Mass (total weight) over time
    ax[0].plot(t_arr, mass_t, lw=0.8)
    ax[0].axhline(mass_t[0], color='r', lw=1.0, ls='--', label=f'initial = {mass_t[0]:.4g}')
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('sum(m_i)')
    ax[0].set_title('Total mass over time (conservation check)')
    ax[0].legend(fontsize=8)

    # [1] u range over time
    ax[1].plot(t_arr, u_min_t, lw=0.8, label='min u')
    ax[1].plot(t_arr, u_max_t, lw=0.8, label='max u')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('u range')
    ax[1].set_title('min / max of reconstructed u')
    ax[1].legend(fontsize=8)

    # [2] Particle positions at snapshots (x histograms)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for k, (step, (xs, ms)) in enumerate(sorted(snaps.items())):
        t_s = (step + 1) * dt
        ax[2].hist(xs, bins=60, weights=np.abs(ms), alpha=0.4,
                   color=colors[k % len(colors)], label=f't≈{t_s:.2g}')
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('|mass| per bin')
    ax[2].set_title('Particle distribution at snapshots')
    ax[2].legend(fontsize=7)

    # [3] Reconstructed u at final time
    ax[3].plot(x_out, u_final, lw=1.2, label='RBMC u(T)')
    ax[3].axhline(float(u_final.min()), color='gray', lw=0.5, ls=':')
    ax[3].axhline(float(u_final.max()), color='gray', lw=0.5, ls=':')
    ax[3].set_xlabel('x')
    ax[3].set_ylabel('u(x, T)')
    ax[3].set_title('Final reconstructed u(x, T)')
    ax[3].legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(diag_dir, 'rbmc_diagnostics.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [RBMC] Saved diagnostics -> {out_path}")


def simulate_burgers_relaxation_gbmc(globs, config, _diag_dir=None):
    """
    Viscous Burgers solver: BPC two-speed relaxation + Gradient Brownian MC.
    Whole-line mode.  Stationary-shock IC only.  No boundary reflection.
    Raw cumulative-sum primary output.  Private seeded RNG.

    :param globs: list of dicts with 'position' (float) and 'value' ([u_i])
    :param config: SimulationConfig; required fields documented in module header.
    :param _diag_dir: optional path; if set, saves a 4-panel diagnostic figure.
    :return: updated globs, positions on uniform [0,L] grid, values = [u_i].
             Always raises (never returns original globs) on any failure.
    """
    # 1. Domain mode
    domain_mode = (
        getattr(config, 'relaxation_domain_mode', None) or 'whole_line'
    ).strip().lower()
    if domain_mode != 'whole_line':
        raise NotImplementedError(
            f"relaxation_gbmc supports only relaxation_domain_mode='whole_line'. "
            f"Got {domain_mode!r}."
        )

    # 2. Initial condition type
    ic_type = (getattr(config, 'burgers_ic_type', '') or '').strip().lower()
    if ic_type != 'stationary_shock':
        raise NotImplementedError(
            "relaxation_gbmc currently supports only "
            "stationary_shock quantile initialisation. "
            f"Got burgers_ic_type={ic_type!r}."
        )

    # 3. Relaxation speed -- must be explicit, no silent fallback
    a_raw = getattr(config, 'relaxation_speed_a', None)
    if a_raw is None:
        raise ValueError(
            "relaxation_speed_a must be set explicitly in the config for "
            "relaxation_gbmc.  Add  \"relaxation_speed_a\": <float>  to the JSON "
            "(e.g. 2.0 for a stationary shock with amplitude A=1)."
        )
    a = float(a_raw)
    if a <= 0.0:
        raise ValueError(f"relaxation_speed_a must be > 0, got {a}.")

    # 4. Shock amplitude
    A_raw = getattr(config, 'burgers_ic_amplitude', None)
    if A_raw is None:
        raise ValueError(
            "burgers_ic_amplitude must be set for stationary_shock IC. "
            "Add  \"amplitude\": <float>  in burgers_initial_condition."
        )
    A = float(A_raw)
    if A <= 0.0:
        raise ValueError(f"burgers_ic_amplitude (A) must be > 0, got {A}.")
    if a <= A:
        raise ValueError(
            f"relaxation_speed_a (a={a:.6g}) must be strictly greater than "
            f"burgers_ic_amplitude (A={A:.6g}) to satisfy  a > A > max|u_i|."
        )

    # 5. Physical parameters
    nu = float(config.diff_constant)
    dt = float(config.time_step)
    T  = float(config.total_time)
    L  = float(config.domain_size)
    N  = int(config.num_points)

    xc_raw = getattr(config, 'burgers_ic_center', None)
    xc = float(xc_raw) if xc_raw is not None else L / 2.0

    # 6. Parameter validation
    if N < 2:
        raise ValueError(f"num_points must be >= 2, got {N}.")
    if nu <= 0.0:
        raise ValueError(
            f"diff_constant (nu) must be > 0 for relaxation_gbmc, got {nu}."
        )
    if dt <= 0.0:
        raise ValueError(f"time_step must be > 0, got {dt}.")
    if T < 0.0:
        raise ValueError(f"total_time must be >= 0, got {T}.")

    n_steps_exact = T / dt
    n_steps = int(round(n_steps_exact))
    if T > 0.0 and abs(n_steps_exact - n_steps) > 1e-8 * max(n_steps_exact, 1.0):
        raise ValueError(
            f"total_time / time_step = {n_steps_exact:.10g} is not an integer "
            f"(total_time={T}, time_step={dt}).  "
            "Adjust either so their ratio is exactly integral."
        )

    # 7. Private RNG
    seed = getattr(config, 'seed', None)
    rng = np.random.default_rng(int(seed) if seed is not None else None)

    # 8. Quantile initialisation for stationary shock
    #
    #   r_i = (i - 1/2) / N   for i = 1..N
    #   X_i = x_c + (2*nu/A) * arctanh(2*r_i - 1)
    #   m_i = -2*A/N  (all equal; sum = -2A = u(+inf) - u(-inf))
    #   u_{-inf} = A
    i_arr = np.arange(1, N + 1, dtype=float)
    r     = (i_arr - 0.5) / float(N)
    x_p   = xc + (2.0 * nu / A) * np.arctanh(2.0 * r - 1.0)
    m_p   = np.full(N, -2.0 * A / float(N))
    u_inf = float(A)
    n_p   = N

    if not np.all(np.isfinite(x_p)):
        raise RuntimeError(
            "Non-finite particle positions from quantile initialisation.  "
            "Check nu, A, xc, and N."
        )

    sigma = np.sqrt(2.0 * nu * dt)

    print(f"  [RBMC] domain_mode=whole_line  IC=stationary_shock")
    print(f"  [RBMC] A={A:.4g}  a={a:.4g}  nu={nu:.4g}  "
          f"x_c={xc:.4g}  L={L:.4g}")
    print(f"  [RBMC] N={N}  dt={dt:.4g}  T={T:.4g}  n_steps={n_steps}")
    print(f"  [RBMC] particle x range at t=0: "
          f"[{float(x_p.min()):.4g}, {float(x_p.max()):.4g}]")

    # 9. Initialise velocity labels (steps 2-6; transport step 1 is skipped)
    order = np.argsort(x_p, kind='stable')
    x_p = x_p[order]
    m_p = m_p[order]

    u = u_inf + np.cumsum(m_p)

    if not np.all(np.isfinite(u)):
        raise RuntimeError("Non-finite reconstructed u at initialisation.")

    max_u_init = float(np.max(np.abs(u)))
    if max_u_init >= a:
        raise RuntimeError(
            f"Subcharacteristic violation at initialisation: "
            f"max|u|={max_u_init:.8g} >= a={a:.8g}."
        )

    p_plus = (a + u) / (2.0 * a)
    if np.any(p_plus < 0.0) or np.any(p_plus > 1.0):
        raise RuntimeError(
            "Invalid BPC equilibrium probability despite subcharacteristic check."
        )

    v = np.where(rng.random(n_p) < p_plus, +a, -a)

    print(f"  [RBMC] Subcharacteristic OK at t=0: "
          f"max|u|={max_u_init:.6g} < a={a:.6g}")

    if _diag_dir is not None:
        _snap_at = {0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1}
        _snaps: dict = {}
        _mass_t: list = []
        _u_min_t: list = []
        _u_max_t: list = []

    # 10. Time loop
    for step in range(n_steps):

        # A: BPC relaxation transport (NO reflection)
        # H1. Transport: X_i <- X_i + V_i * dt
        x_p = x_p + v * dt

        # H2. Sort (X, m, V) together by position.
        ord_s = np.argsort(x_p, kind='stable')
        x_p = x_p[ord_s]
        m_p = m_p[ord_s]
        v   = v[ord_s]

        # H3. Reconstruct u_i = u_{-inf} + cumsum(m_i).
        u = u_inf + np.cumsum(m_p)

        # H4. Finite check.
        if not np.all(np.isfinite(u)):
            raise RuntimeError(
                f"Non-finite reconstructed u at step {step + 1}."
            )

        # H5. Strict subcharacteristic check.
        max_u_step = float(np.max(np.abs(u)))
        if max_u_step >= a:
            raise RuntimeError(
                f"Subcharacteristic violation at step {step + 1}: "
                f"max|u|={max_u_step:.8g} >= a={a:.8g}.  "
                "Reduce dt, increase N, or increase relaxation_speed_a."
            )

        # H6. Validate equilibrium probabilities.
        p_plus = (a + u) / (2.0 * a)
        if np.any(p_plus < 0.0) or np.any(p_plus > 1.0):
            raise RuntimeError(
                f"Invalid switching probability at step {step + 1}: "
                f"p_plus range "
                f"[{float(p_plus.min()):.6g}, {float(p_plus.max()):.6g}]."
            )

        # H7. Stochastic instantaneous-equilibrium switching.
        #     E[V_i | u_i] = a*p_i^+ - a*(1-p_i^+) = u_i.
        #     Labels resampled exactly ONCE here; NOT resampled after diffusion.
        v = np.where(rng.random(n_p) < p_plus, +a, -a)

        # B: Brownian diffusion (NO reflection)
        # H8. Diffuse.
        x_p = x_p + rng.normal(0.0, sigma, size=n_p)

        if _diag_dir is not None:
            _mass_t.append(float(m_p.sum()))
            _u_min_t.append(float(u.min()))
            _u_max_t.append(float(u.max()))
            if step in _snap_at:
                _snaps[step] = (x_p.copy(), m_p.copy())

    # 11. Outside-window tracking
    x_lo = 0.0
    x_hi = float(L)
    outside = (x_p < x_lo) | (x_p > x_hi)
    n_outside          = int(np.sum(outside))
    mass_outside_signed = float(np.sum(m_p[outside]))
    mass_outside_abs    = float(np.sum(np.abs(m_p[outside])))

    total_mass_final = float(m_p.sum())
    print(f"  [RBMC] Final: n_outside={n_outside}  "
          f"mass_outside_abs={mass_outside_abs:.4e}  "
          f"total_particle_mass={total_mass_final:.6e}  "
          f"(expected {-2.0 * A:.6e})")

    # 12. Raw cumulative-sum reconstruction on output grid [0, L]
    #     No smoothing applied.  Particles outside [0, L] contribute correctly:
    #     those at X < 0 are counted for all output points;
    #     those at X > L are counted for none.
    N_out = len(globs)
    x_out = np.linspace(0.0, L, N_out)

    ord_f   = np.argsort(x_p, kind='stable')
    x_p_s   = x_p[ord_f]
    m_p_s   = m_p[ord_f]
    cumm    = np.concatenate([[0.0], np.cumsum(m_p_s)])
    idx     = np.searchsorted(x_p_s, x_out, side='right')
    u_out   = u_inf + cumm[idx]

    print(f"  [RBMC] u_out range: [{float(u_out.min()):.6g}, {float(u_out.max()):.6g}]")

    if _diag_dir is not None:
        _save_rbmc_diagnostics(
            _diag_dir, _mass_t, _u_min_t, _u_max_t, _snaps,
            x_out, u_out, u_inf, dt, n_steps,
        )

    for i in range(N_out):
        globs[i]['position'] = float(x_out[i])
        globs[i]['value']    = [float(u_out[i])]

    return globs
