"""Bertaglia--Pareschi--Caflisch (BPC) relaxation + gradient Brownian Monte
Carlo (GBMC) for viscous Burgers' equation.

Manuscript map: this module is the shared particle update of Algorithm 1
(labels `alg:gbmc`, `sec:gbmc-algorithm`); its exact stage properties (mass
conservation, conditional-mean transport, Brownian heat consistency) are
stated in `sec:gbmc-properties`. Every study driver calls this one
implementation; there are no copied evolution loops.

PDE:  u_t + u * u_x = nu * u_xx

Domain mode: WHOLE LINE only.  Particles move freely on the real line.
No boundary reflection is applied.  An output window [0, L] is used for
reporting; particles outside that window contribute correctly to the raw
cumulative-sum reconstruction and are counted in the outside-window report.

Gradient particles represent w = u_x via the signed empirical measure:

    w^N(x, t) = sum_i  m_i * delta(x - X_i(t))

The field u is reconstructed from sorted particles by the RAW cumulative sum:

    u(x, t) = u_{-inf} + sum_{i : X_i <= x}  m_i

The public compatibility wrapper supports the stationary shock used in the
paper.  The shared particle core also advances shifted (traveling) tanh shocks,
so the stationary and traveling studies exercise one implementation of the
relaxation--Brownian update rather than copied time loops.

    u(x) = -A * tanh(A * (x - x_c) / (2 * nu))

Quantile particle initialisation (recovers exact IC in the particle limit):

    r_i = (i - 1/2) / N   for i = 1 .. N
    X_i = x_c + (2*nu/A) * arctanh(2*r_i - 1)
    m_i = -2*A / N          (all equal; sum(m) = -2A = u(+inf) - u(-inf))
    u_{-inf} = A

Each particle carries a velocity label  V_i in {-a, +a}, redrawn once per
time step from the local equilibrium and carried through the intervening
sort and Brownian displacement.

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
The output uses the unsmoothed cumulative sum.

RNG: a private numpy.random.Generator is used throughout.  If config.seed is
not None it is seeded deterministically; otherwise a fresh non-seeded
generator is used (independent of the global numpy random state).

References:
  Bertaglia, Pareschi & Caflisch (2024) — gradient-based Monte Carlo (GBMC)
    two-speed relaxation particles; the "BPC" switching law used here.
  Jin & Xin (1995) — relaxation system and subcharacteristic condition.
  Bouchut & Perthame (1993) — kinetic relaxation for conservation laws.
  Bossy & Talay (1997) — GBMC for McKean SDEs / Burgers via McKean–Vlasov.
  Méléard (1996) — propagation of chaos, particle approximations.
"""

import os

import numpy as np


def exact_stationary_shock(x, nu, amplitude=1.0, center=2.0):
    """Return the stationary viscous Burgers shock on the whole line."""
    x = np.asarray(x, dtype=float)
    return -amplitude * np.tanh(
        amplitude * (x - center) / (2.0 * nu)
    )


def initialize_tanh_shock_particles(N, nu, amplitude, center,
                                    mean_level=0.0):
    """Create the equal-mass quantile representation of a tanh shock
    (manuscript label `eq:quantile-init`).

    ``mean_level=0`` gives the stationary profile used by the production
    wrapper.  A nonzero value gives the traveling profile
    ``mean_level - amplitude*tanh(...)``.  In both cases the represented
    derivative and signed particle masses are identical.
    """
    N = int(N)
    nu = float(nu)
    amplitude = float(amplitude)
    center = float(center)
    mean_level = float(mean_level)
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}.")
    if nu <= 0.0:
        raise ValueError(f"nu must be > 0, got {nu}.")
    if amplitude <= 0.0:
        raise ValueError(f"amplitude must be > 0, got {amplitude}.")

    r = (np.arange(1, N + 1, dtype=float) - 0.5) / float(N)
    x_p = center + (2.0 * nu / amplitude) * np.arctanh(2.0 * r - 1.0)
    m_p = np.full(N, -2.0 * amplitude / float(N))
    u_left = mean_level + amplitude
    if not np.all(np.isfinite(x_p)):
        raise RuntimeError(
            "Non-finite particle positions from quantile initialisation. "
            "Check nu, amplitude, center, and N."
        )
    return x_p, m_p, u_left


def reconstruct_cumulative_field(x_p, m_p, u_left, x_out):
    """Reconstruct the field by the unsmoothed signed cumulative sum
    (manuscript label `eq:background-cumsum`)."""
    x_p = np.asarray(x_p, dtype=float)
    m_p = np.asarray(m_p, dtype=float)
    x_out = np.asarray(x_out, dtype=float)
    if x_p.ndim != 1 or m_p.ndim != 1 or len(x_p) != len(m_p):
        raise ValueError("x_p and m_p must be one-dimensional arrays of equal length.")
    order = np.argsort(x_p, kind='stable')
    x_sorted = x_p[order]
    m_sorted = m_p[order]
    cumulative_mass = np.concatenate([[0.0], np.cumsum(m_sorted)])
    # side='right' implements the manuscript convention: u(x) sums the masses
    # of every particle with X_i <= x.
    indices = np.searchsorted(x_sorted, x_out, side='right')
    return float(u_left) + cumulative_mass[indices]


def advance_rbgbmc_particles(x_p, m_p, u_left, nu, a, dt, n_steps, rng,
                             snapshot_steps=None, record_history=False,
                             collect_label_diagnostics=False,
                             rng_brownian=None,
                             conditional_mean_transport=False,
                             redraw_after_diffusion=False):
    """Advance signed gradient particles with the paper's shared stepper.

    One step is exactly the Lie composition stated in the manuscript:
    two-speed relaxation transport and equilibrium resampling, followed by
    Brownian diffusion. ``snapshot_steps`` uses one-based completed-step
    indices and is intended for study drivers, not for a separate algorithm.

    ``rng_brownian`` and ``conditional_mean_transport`` support an internal
    conditional-mean transport control (an ablation), not the production path.
    When ``rng_brownian`` is given, label draws use ``rng`` and Brownian draws
    use ``rng_brownian``, so two arms sharing the same ``rng_brownian`` receive
    identical Brownian increments regardless of their transport. When
    ``conditional_mean_transport`` is True, the transport velocity is the exact
    conditional mean V_i = u_i (Burgers f'(u)=u) and no label uniforms are
    drawn. Both default off, leaving the single-stream two-speed production path
    bit-for-bit unchanged. This control is not Roberts' method or a competing
    solver.

    ``redraw_after_diffusion`` is an OPT-IN alternative update schedule for the
    ordering pilot only; it is not the production path. When True, one step is
    transport -> diffuse -> sort/reconstruct -> verify -> redraw, so the
    velocity used by the next transport is set from the post-Brownian
    reconstruction instead of the post-transport one. Per step both schedules
    draw the same velocity-resampling uniforms and the same Brownian normal array in the
    same order, so two runs sharing generators are paired through common
    random numbers. The default (False) leaves the production schedule
    bit-for-bit unchanged.

    ``collect_label_diagnostics`` is the legacy internal name of a read-only
    velocity-sampling diagnostic. When True the
    stepper accumulates the mean of ``a**2 - u_i**2`` over the reconstructed
    states at which the sampled velocities *actually used for transport* were
    drawn: the initial reconstruction (whose velocities drive the first transport)
    and every in-loop reconstruction except the final one (whose velocities are
    drawn but never used before the loop ends). It consumes no random numbers
    and does not alter the draw order, so the returned solution arrays are
    identical whether or not it is enabled. The manuscript denotes the resulting
    scale by ``D_vel = (dt/2) * <a**2 - u**2>``. Archived outputs retain the
    legacy key ``D_label``; for the equal-mass shock the quantity equals the
    closed form in `eq:D-label-stationary`.
    """
    x_p = np.asarray(x_p, dtype=float).copy()
    m_p = np.asarray(m_p, dtype=float).copy()
    u_left = float(u_left)
    nu = float(nu)
    a = float(a)
    dt = float(dt)
    n_steps = int(n_steps)
    if x_p.ndim != 1 or m_p.ndim != 1 or len(x_p) != len(m_p):
        raise ValueError("x_p and m_p must be one-dimensional arrays of equal length.")
    if len(x_p) < 2:
        raise ValueError("At least two particles are required.")
    if nu <= 0.0:
        raise ValueError(f"nu must be > 0, got {nu}.")
    if a <= 0.0:
        raise ValueError(f"a must be > 0, got {a}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}.")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}.")
    if not np.all(np.isfinite(x_p)) or not np.all(np.isfinite(m_p)):
        raise RuntimeError("Non-finite initial particle data.")

    requested = set(snapshot_steps or ())
    invalid_steps = {step for step in requested if step < 1 or step > n_steps}
    if invalid_steps:
        raise ValueError(
            f"snapshot_steps must lie in [1, {n_steps}], got {sorted(invalid_steps)}."
        )

    order = np.argsort(x_p, kind='stable')
    x_p = x_p[order]
    m_p = m_p[order]
    u = u_left + np.cumsum(m_p)
    if not np.all(np.isfinite(u)):
        raise RuntimeError("Non-finite reconstructed u at initialisation.")
    max_u_init = float(np.max(np.abs(u)))
    if max_u_init >= a:
        raise RuntimeError(
            "Subcharacteristic violation at initialisation: "
            f"max|u|={max_u_init:.8g} >= a={a:.8g}."
        )
    p_plus = (a + u) / (2.0 * a)
    if np.any(p_plus < 0.0) or np.any(p_plus > 1.0):
        raise RuntimeError(
            "Invalid BPC equilibrium probability despite subcharacteristic check."
        )
    rng_brownian_stream = rng if rng_brownian is None else rng_brownian
    if conditional_mean_transport:
        v = u.copy()
    else:
        v = np.where(rng.random(len(x_p)) < p_plus, +a, -a)

    # Read-only label-variance diagnostic. The initial labels (drawn just above
    # from u^(0)) drive the first transport, so u^(0) is a used label state
    # whenever at least one step runs.
    label_excess_sum = 0.0
    label_excess_count = 0
    if collect_label_diagnostics and n_steps >= 1:
        label_excess_sum += float(np.sum(a * a - u * u))
        label_excess_count += len(u)

    sigma = np.sqrt(2.0 * nu * dt)
    snapshots = {}
    mass_history = []
    u_min_history = []
    u_max_history = []

    for step in range(1, n_steps + 1):
        # Production timing (Algorithm 1, `sec:gbmc-algorithm`): the labels in
        # v were drawn from the PREVIOUS reconstruction, so this transport,
        # the following sort/reconstruct/verify/redraw, and the Brownian
        # displacement must keep this order. Do not reorder these stages.
        x_p = x_p + v * dt
        if redraw_after_diffusion:
            # Ordering-pilot schedule: diffuse BEFORE the reconstruction that
            # sets the next transport velocity. Same Brownian draw per step as
            # the production schedule, so paired runs stay aligned.
            x_p = x_p + rng_brownian_stream.normal(0.0, sigma, size=len(x_p))
        order = np.argsort(x_p, kind='stable')
        x_p = x_p[order]
        m_p = m_p[order]
        v = v[order]
        u = u_left + np.cumsum(m_p)
        if not np.all(np.isfinite(u)):
            raise RuntimeError(f"Non-finite reconstructed u at step {step}.")
        max_u_step = float(np.max(np.abs(u)))
        if max_u_step >= a:
            raise RuntimeError(
                f"Subcharacteristic violation at step {step}: "
                f"max|u|={max_u_step:.8g} >= a={a:.8g}. "
                "Reduce dt, increase N, or increase the relaxation speed."
            )
        # Equilibrium switching probability (`eq:switch-prob`); the strict
        # subcharacteristic check above guarantees p_plus lies in (0, 1), so
        # any violation here is a genuine failure, never clipped or repaired.
        p_plus = (a + u) / (2.0 * a)
        if np.any(p_plus < 0.0) or np.any(p_plus > 1.0):
            raise RuntimeError(
                f"Invalid switching probability at step {step}: p_plus range "
                f"[{float(p_plus.min()):.6g}, {float(p_plus.max()):.6g}]."
            )
        # The labels drawn from this reconstruction drive the *next* transport,
        # so this state is a used label state for every step except the last.
        if collect_label_diagnostics and step < n_steps:
            label_excess_sum += float(np.sum(a * a - u * u))
            label_excess_count += len(u)
        if conditional_mean_transport:
            v = u.copy()
        else:
            v = np.where(rng.random(len(x_p)) < p_plus, +a, -a)
        if not redraw_after_diffusion:
            x_p = x_p + rng_brownian_stream.normal(0.0, sigma, size=len(x_p))

        if record_history:
            mass_history.append(float(m_p.sum()))
            u_min_history.append(float(u.min()))
            u_max_history.append(float(u.max()))
        if step in requested:
            snapshots[step] = (x_p.copy(), m_p.copy())

    label_excess_mean = (
        label_excess_sum / label_excess_count
        if label_excess_count > 0 else float('nan')
    )
    return {
        'x': x_p,
        'm': m_p,
        'v': v,
        'u_last_sorted': u,
        'max_u_init': max_u_init,
        'snapshots': snapshots,
        'mass_history': mass_history,
        'u_min_history': u_min_history,
        'u_max_history': u_max_history,
        'label_excess_mean': label_excess_mean,
    }


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
        t_s = step * dt
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


def simulate_burgers_relaxation_gbmc(globs, config, _diag_dir=None,
                                     diagnostics_out=None):
    """
    Viscous Burgers solver: BPC two-speed relaxation + Gradient Brownian MC.
    Whole-line mode.  Stationary-shock IC only.  No boundary reflection.
    Raw cumulative-sum primary output.  Private seeded RNG.

    :param globs: list of dicts with 'position' (float) and 'value' ([u_i]).
        These are reconstruction-grid samples, not gradient particles.
        Their count sets the output grid; quantile particles are initialized
        separately from config. The legacy name is kept for interface stability.
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

    # 8. Equal-mass quantile initialisation for the stationary shock.
    x_p, m_p, u_inf = initialize_tanh_shock_particles(
        N, nu, A, xc, mean_level=0.0
    )

    print(f"  [RBMC] domain_mode=whole_line  IC=stationary_shock")
    print(f"  [RBMC] A={A:.4g}  a={a:.4g}  nu={nu:.4g}  "
          f"x_c={xc:.4g}  L={L:.4g}")
    print(f"  [RBMC] N={N}  dt={dt:.4g}  T={T:.4g}  n_steps={n_steps}")
    print(f"  [RBMC] particle x range at t=0: "
          f"[{float(x_p.min()):.4g}, {float(x_p.max()):.4g}]")

    # 9. Advance through the same shared core used by the traveling study.
    snapshot_steps = None
    if _diag_dir is not None and n_steps > 0:
        snapshot_steps = {
            1, n_steps // 4 + 1, n_steps // 2 + 1,
            3 * n_steps // 4 + 1, n_steps,
        }
        snapshot_steps = {step for step in snapshot_steps if step <= n_steps}
    run = advance_rbgbmc_particles(
        x_p, m_p, u_inf, nu, a, dt, n_steps, rng,
        snapshot_steps=snapshot_steps,
        record_history=_diag_dir is not None,
        collect_label_diagnostics=diagnostics_out is not None,
    )
    x_p = run['x']
    m_p = run['m']
    max_u_init = run['max_u_init']
    if diagnostics_out is not None:
        label_excess_mean = run['label_excess_mean']
        diagnostics_out['label_excess_mean'] = label_excess_mean
        diagnostics_out['a'] = a
        diagnostics_out['nu'] = nu
        diagnostics_out['dt'] = dt
        diagnostics_out['D_label'] = 0.5 * dt * label_excess_mean
        diagnostics_out['rho_label'] = (0.5 * dt * label_excess_mean) / nu
    print(f"  [RBMC] Subcharacteristic OK at t=0: "
          f"max|u|={max_u_init:.6g} < a={a:.6g}")

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

    # 12. Raw cumulative-sum reconstruction on the reconstruction grid [0, L]
    #     No smoothing applied.  Particles outside [0, L] contribute correctly:
    #     those at X < 0 are counted for all output points;
    #     those at X > L are counted for none.
    N_out = len(globs)
    x_out = np.linspace(0.0, L, N_out)

    u_out = reconstruct_cumulative_field(x_p, m_p, u_inf, x_out)

    print(f"  [RBMC] u_out range: [{float(u_out.min()):.6g}, {float(u_out.max()):.6g}]")

    if _diag_dir is not None and n_steps > 0:
        _save_rbmc_diagnostics(
            _diag_dir, run['mass_history'], run['u_min_history'],
            run['u_max_history'], run['snapshots'],
            x_out, u_out, u_inf, dt, n_steps,
        )

    for i in range(N_out):
        globs[i]['position'] = float(x_out[i])
        globs[i]['value']    = [float(u_out[i])]

    return globs
