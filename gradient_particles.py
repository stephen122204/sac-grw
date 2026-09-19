"""Single-simulation gradient-particle update of Algorithm 1 in the paper (Section 2.2).

Sorted signed particles carry positions, masses, and the velocities used by the next
transport, then receive independent Gaussian diffusion increments. The module also
provides the cumulative field reconstruction and the tanh-shock initialization of
Appendix C, with particles on the whole line and an output window used only for reporting.
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


def advance_particles(x_p, m_p, u_left, nu, a, dt, n_steps, rng,
                             snapshot_steps=None, record_history=False,
                             collect_label_diagnostics=False,
                             rng_brownian=None,
                             conditional_mean_transport=False,
                             redraw_after_diffusion=False,
                             compensate_transport_variance=False,
                             min_variance_three_speed=False,
                             antithetic=False):
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

    ``compensate_transport_variance`` is an OPT-IN experimental variant, not the
    default path and not used by the sign-aware coupling manuscript. When True the Brownian
    standard deviation is reduced per particle from sqrt(2*nu*dt) to
    sqrt(2*nu*dt - (a**2 - u_i**2) * dt**2), using the same reconstructed state
    u_i that set the switching probability for the velocity carried into the
    next transport. The conditional displacement variance between interior velocity-sampling
    stages is then 2*nu*dt; this does not cover the initial transport-first loop. It is admissible
    only while every radicand is positive, i.e. dt < 2*nu/(a**2 - u_i**2) for
    every particle; a violation raises RuntimeError rather than clipping. The
    default (False) leaves the production path bit-for-bit unchanged.

    ``min_variance_three_speed`` is an archived experimental option using
    {-a,0,+a}, with mean u and variance a*abs(u)-u**2. It is not the method
    evaluated in the sign-aware coupling manuscript.

    ``antithetic`` is an OPT-IN pairing flag for variance-reduction pilots. When
    True the drawn uniforms are reflected (xi -> 1-xi) and the drawn normals are
    negated (Z -> -Z). Both consume the identical underlying stream, so a run
    with ``antithetic=True`` and one with ``antithetic=False`` on the same
    generators form an antithetic pair: each retains the correct sampling law
    marginally, while their inputs are maximally negatively coupled. The default
    (False) leaves the production path bit-for-bit unchanged.

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
    if compensate_transport_variance and (conditional_mean_transport or redraw_after_diffusion):
        raise ValueError("compensation requires sampled transport with the carried-velocity schedule")
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
    def _draw_velocity(u_state, p_state, n):
        """Velocity draw. Two-speed is the BPC choice on {-a,+a}, the unique
        distribution there with mean u.  The three-speed option is the
        minimum-variance distribution on {-a,0,+a} with the same mean: move at
        a*sign(u) with probability |u|/a, otherwise stay.  Its conditional
        variance is a|u| - u^2 rather than a^2 - u^2.  Both consume one uniform
        per particle, so paired streams stay aligned."""
        xi = rng.random(n)
        if antithetic:
            xi = 1.0 - xi
        if min_variance_three_speed:
            return np.where(xi < np.abs(u_state) / a, a * np.sign(u_state), 0.0)
        return np.where(xi < p_state, +a, -a)

    if conditional_mean_transport:
        v = u.copy()
    else:
        v = _draw_velocity(u, p_plus, len(x_p))

    # Read-only label-variance diagnostic. The initial labels (drawn just above
    # from u^(0)) drive the first transport, so u^(0) is a used label state
    # whenever at least one step runs.
    label_excess_sum = 0.0
    label_excess_count = 0
    if collect_label_diagnostics and n_steps >= 1:
        label_excess_sum += float(np.sum(a * a - u * u))
        label_excess_count += len(u)

    sigma = np.sqrt(2.0 * nu * dt)

    def _brownian(sd, n):
        """Brownian increment; negated under ``antithetic`` so the reflected run
        consumes the same normals with opposite sign. The generator is called
        with the scale argument, as the interface contract test requires."""
        z = rng_brownian_stream.normal(0.0, sd, size=n)
        return -z if antithetic else z

    def _sigma_for(u_state):
        """Per-particle Brownian sigma. Without compensation this is the scalar
        sqrt(2*nu*dt); with it, the transport variance sampled at ``u_state`` is
        subtracted so the interior sampling-stage-to-sampling-stage conditional
        displacement variance is 2*nu*dt. This does not cover the initial
        transport-first loop or arbitrary output-to-output intervals."""
        if not compensate_transport_variance:
            return sigma
        q = (a * np.abs(u_state) - u_state * u_state) if min_variance_three_speed \
            else (a * a - u_state * u_state)
        var = 2.0 * nu * dt - q * dt * dt
        if np.any(var <= 0.0):
            raise RuntimeError(
                'compensate_transport_variance is inadmissible at this dt: '
                f'min residual Brownian variance {float(var.min()):.6g} <= 0. '
                f'Require q*dt < 2*nu for the selected velocity law; here 2*nu*dt={2.0 * nu * dt:.6g} '
                f'and max transport variance='
                f'{float(np.max(q) * dt * dt):.6g}.')
        return np.sqrt(var)

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
            x_p = x_p + _brownian(sigma, len(x_p))
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
            v = _draw_velocity(u, p_plus, len(x_p))
        if not redraw_after_diffusion:
            x_p = x_p + _brownian(_sigma_for(u), len(x_p))

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
