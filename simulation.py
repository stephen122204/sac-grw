import numpy as np


def random_walk(globs, diff_constant, time_step):
    """
    Vectorized Brownian displacement for all heat globs.

    The GRW method evolves globs representing the gradient-side computational elements.
    Each glob's position is updated by a Gaussian displacement with mean 0 and variance
    2 * alpha * dt, i.e. sigma = sqrt(2 * alpha * dt), matching the thesis random-walk step.

    Glob values (signed gradient weights) are not changed here; they are only modified by
    apply_boundary_conditions when a Neumann wall is crossed.

    :param globs: list of dicts with keys 'position' and 'value'
    :param diff_constant: thermal diffusivity alpha
    :param time_step: dt
    :return: updated list of globs (same list, mutated in place)
    """
    n = len(globs)
    if n == 0:
        return globs

    positions = np.fromiter((g['position'] for g in globs), dtype=float, count=n)

    sigma = np.sqrt(2.0 * diff_constant * time_step)
    positions += np.random.normal(0.0, sigma, size=n)

    for i, g in enumerate(globs):
        g['position'] = float(positions[i])

    return globs


def apply_boundary_conditions(globs, boundary_conditions, domain_size):
    """
    Apply boundary conditions to heat globs after each Brownian step.

    Thesis GRW rules:
    - Dirichlet: symmetric reflection — reflect by the overshoot distance back into the domain.
      The glob value is preserved (the boundary fixes u, not u_x).
    - Neumann: anti-symmetric reflection — reflect by the overshoot distance AND negate the
      glob value. Negating the value is what enforces zero flux (zero u_x) at the wall.

    :param globs: list of dicts with keys 'position' and 'value'
    :param boundary_conditions: dict with 'LEFT' and 'RIGHT' sub-dicts, each having 'type'
    :param domain_size: float, right endpoint of the domain (left endpoint is 0)
    :return: updated list of globs
    """
    for glob in globs:
        # --- left boundary (x = 0) ---
        if glob['position'] < 0:
            bc_type = boundary_conditions['LEFT']['type'].lower()
            if bc_type == 'dirichlet':
                # symmetric reflection: overshoot = -position, reflect back to +|position|
                glob['position'] = -glob['position']
            elif bc_type == 'neumann':
                # anti-symmetric: reflect position, negate value
                glob['position'] = -glob['position']
                glob['value'] = -glob['value']

        # --- right boundary (x = domain_size) ---
        elif glob['position'] > domain_size:
            bc_type = boundary_conditions['RIGHT']['type'].lower()
            if bc_type == 'dirichlet':
                # symmetric reflection: overshoot = position - domain_size, reflect back
                glob['position'] = 2 * domain_size - glob['position']
            elif bc_type == 'neumann':
                # anti-symmetric: reflect position, negate value
                glob['position'] = 2 * domain_size - glob['position']
                glob['value'] = -glob['value']

    return globs


def simulate_heat_equation(globs, config):
    """
    Evolve heat globs forward in time using the GRW method.

    Each glob carries a position and a signed value representing its contribution to the
    gradient field u_x. The time loop applies:
      1. Brownian random walk — displacement ~ Normal(0, 2 * alpha * dt)
      2. Boundary conditions — Dirichlet uses symmetric reflection (value preserved);
         Neumann uses anti-symmetric reflection (value negated).

    The heat solution u(x, t) is NOT stored directly; it is recovered afterward by sorting
    globs by position and cumulatively summing their values (numerical integration of u_x).

    :param globs: list of dicts with keys 'position' (float) and 'value' (float)
    :param config: SimulationConfig with diff_constant, time_step, total_time,
                   boundary_conditions, and domain_size
    :return: final list of globs after all time steps
    """
    for _ in range(int(config.total_time / config.time_step)):
        globs = random_walk(globs, config.diff_constant, config.time_step)
        globs = apply_boundary_conditions(globs, config.boundary_conditions, config.domain_size)
    return globs


def simulate_fitzhugh_nagumo(globs, config):
    """
    Experimental GRW-inspired particle method for the FitzHugh-Nagumo equations.

    PDEs:
      du/dt = D * d2u/dx2  +  tau * (u - u^3/3 + v)
      dv/dt =               - (1/tau) * (u - a + b*v)

    GRW-inspired formulation:

      1. Spatial diffusion of u (GRW step):
           x_i += Normal(0, sqrt(2 * D * dt))
         Each particle's position undergoes a Brownian random walk, identical to
         the heat GRW with alpha = D.  This represents the D * d2u/dx2 term.

      2. Boundary reflection (same rules as the heat GRW):
         Dirichlet -- symmetric reflection (position mirrored, u_i preserved).
         Neumann   -- anti-symmetric reflection (position mirrored, u_i negated).
         v always uses symmetric reflection (v_i is always preserved at walls).

      3. Local reaction (explicit Euler per particle):
           u_i += dt * tau * (u_i - u_i^3/3 + v_i)
           v_i += dt * (-(1/tau) * (u_i - a + b*v_i))
         Each particle integrates its own local ODE independently, without
         spatial coupling to neighboring particles.

    Each glob carries:
      'position' : x_i, evolves by random walk
      'value'    : [u_i, v_i], local state variables at this particle's location

    Reconstruction of the spatial fields u(x, t) and v(x, t) is done in
    utils.py and verify_solver.py by sorting particles by position and
    interpolating their values onto a uniform grid.

    Methodological notes (experimental formulation):
      - The reaction step is evaluated at each particle's carried (u_i, v_i)
        without reference to neighbouring particles.  In the true FHN PDE, the
        diffusion term provides spatial coupling: excited u at one location
        drives u at nearby locations above the excitation threshold.  In this
        particle method that coupling exists only indirectly, through particles
        diffusing from one spatial region into another.
      - As a consequence, traveling-wave propagation in this formulation is
        weaker than in the FD solver.  The wave "appears" to spread primarily
        via diffusion of excited particles into the rest region, rather than
        via the threshold-activation mechanism of the true FHN field PDE.
      - v is NOT diffused in the true FHN, but in this particle formulation v
        is transported with the particle position.  This introduces an effective
        spatial diffusion of v, which is an approximation error.
      - These limitations are quantified by the comparison with the FD reference
        in verify_solver.py and are part of the GRW feasibility study.
      - For the reference FD solve, see simulate_fitzhugh_nagumo_fd below.

    :param globs: list of dicts with 'position' (evolves) and 'value' = [u_i, v_i]
    :param config: SimulationConfig with a, b, tau, diff_constant (D),
                   time_step, total_time, domain_size, boundary_conditions
    :return: updated glob list with evolved positions and state values
    """
    a   = config.a
    b   = config.b
    tau = config.tau
    dt  = config.time_step
    D   = config.diff_constant
    L   = config.domain_size
    bc  = config.boundary_conditions
    n   = len(globs)
    if n == 0:
        return globs

    bc_left_type  = bc['LEFT']['type'].lower()
    bc_right_type = bc['RIGHT']['type'].lower()

    positions = np.array([g['position']   for g in globs], dtype=float)
    u_vals    = np.array([g['value'][0]   for g in globs], dtype=float)
    v_vals    = np.array([g['value'][1]   for g in globs], dtype=float)

    sigma = np.sqrt(2.0 * D * dt) if D > 0.0 else 0.0

    for _ in range(int(config.total_time / dt)):
        # Step 1: GRW diffusion — Brownian walk for u spatial diffusion.
        if sigma > 0.0:
            positions += np.random.normal(0.0, sigma, size=n)

        # Step 2: Boundary reflection on positions.
        mask = positions < 0.0
        if np.any(mask):
            positions[mask] = -positions[mask]
            if bc_left_type == 'neumann':
                u_vals[mask] = -u_vals[mask]
            # v is always preserved at walls (symmetric reflection only).

        mask = positions > L
        if np.any(mask):
            positions[mask] = 2.0 * L - positions[mask]
            if bc_right_type == 'neumann':
                u_vals[mask] = -u_vals[mask]

        # Step 3: Local reaction — each particle integrates its own ODE.
        u_new = u_vals + dt * tau * (u_vals - u_vals**3 / 3.0 + v_vals)
        v_new = v_vals + dt * (-(1.0 / tau) * (u_vals - a + b * v_vals))
        u_vals, v_vals = u_new, v_new

    for i, g in enumerate(globs):
        g['position'] = float(positions[i])
        g['value']    = [float(u_vals[i]), float(v_vals[i])]

    return globs


def simulate_fitzhugh_nagumo_fd(globs, config):
    """
    Standard fixed-grid explicit finite-difference solver for FitzHugh-Nagumo.

    This implementation is kept as an internal reference-quality solver.
    It is used by verify_solver.py to generate high-resolution reference
    solutions for comparison with the GRW particle method above.  It is NOT
    the primary solver on the main (GRW-feasibility) branch.

    PDEs:
      du/dt = D * d2u/dx2  +  tau * (u - u^3/3 + v)
      dv/dt =               - (1/tau) * (u - a + b*v)

    Stability (von Neumann criterion for the diffusion term): dt <= dx^2 / (2*D).

    :param globs: list of dicts with 'position' (fixed) and 'value' = [u_i, v_i]
    :param config: SimulationConfig
    :return: same glob list with updated 'value' fields; positions unchanged
    """
    a   = config.a
    b   = config.b
    tau = config.tau
    dt  = config.time_step
    D   = config.diff_constant

    N  = len(globs)
    dx = config.domain_size / (N - 1)

    bc = config.boundary_conditions
    bc_left_type  = bc['LEFT']['type'].lower()
    bc_right_type = bc['RIGHT']['type'].lower()
    bc_left_val   = float(bc['LEFT'].get('value', 0.0))
    bc_right_val  = float(bc['RIGHT'].get('value', 0.0))

    u = np.array([g['value'][0] for g in globs], dtype=float)
    v = np.array([g['value'][1] for g in globs], dtype=float)

    for _ in range(int(config.total_time / dt)):
        if D > 0.0:
            u_xx       = np.empty(N, dtype=float)
            u_xx[1:-1] = (u[:-2] - 2.0 * u[1:-1] + u[2:]) / dx**2
            u_xx[0]    = 2.0 * (u[1]  - u[0])  / dx**2
            u_xx[-1]   = 2.0 * (u[-2] - u[-1]) / dx**2
        else:
            u_xx = np.zeros(N, dtype=float)

        u_new = u + dt * (D * u_xx + tau * (u - u**3 / 3.0 + v))
        v_new = v + dt * (-(1.0 / tau) * (u - a + b * v))
        u, v = u_new, v_new

        if bc_left_type == 'dirichlet':
            u[0] = bc_left_val
        else:
            u[0] = u[1]
        if bc_right_type == 'dirichlet':
            u[-1] = bc_right_val
        else:
            u[-1] = u[-2]
        v[0]  = v[1]
        v[-1] = v[-2]

    for i, g in enumerate(globs):
        g['value'] = [float(u[i]), float(v[i])]

    return globs


def simulate_burgers(globs, config):
    """
    Experimental GRW-inspired Lagrangian particle method for Burgers' equation.

    Burgers' equation:  u_t + u * u_x = nu * u_xx

    Operator splitting at each time step:

      1. Characteristic advection (Lagrangian):
           x_i += u_i * dt
         Each particle moves at its own carried velocity, following the inviscid
         Burgers characteristic.  The u_i values are fixed (they represent the
         initial data transported along each characteristic).

      2. Viscous diffusion (GRW):
           x_i += Normal(0, sqrt(2 * nu * dt))
         The diffusion term nu * u_xx is modelled by the same Brownian random walk
         used in the heat GRW, with alpha = nu.

      3. Boundary reflection: same overshoot-reflection rules as the heat GRW.
         Dirichlet walls use symmetric reflection (position mirrored, u value
         preserved).  Neumann walls use anti-symmetric reflection (position
         mirrored, u value negated).

    Each glob carries:
      'position' : current particle location x_i  (evolves at every step)
      'value'    : [u_i], the velocity value carried by this particle

    Reconstruction of u(x, t) from the scattered particle list is done in
    utils.py and verify_solver.py by sorting particles and interpolating.

    Methodological notes (experimental formulation):
      - u_i values are held fixed throughout the simulation (inviscid
        characteristics).  This is exact for nu = 0 and an approximation for
        nu > 0 because the diffusion step does not feed back into the carried
        velocity.
      - Near a shock, characteristics converge, causing particle clustering and
        noisy reconstruction in that region.  This is physically meaningful
        (shock formation concentrates characteristics) but degrades pointwise
        accuracy compared to the FD reference.
      - The GRW diffusion step broadens the particle distribution and smooths
        the apparent shock, playing the role of viscosity without requiring
        any spatial derivative calculation.
      - For comparison with the FD baseline, use simulate_burgers_fd below.

    :param globs: list of dicts with 'position' and 'value' = [u_i]
    :param config: SimulationConfig with time_step, total_time, domain_size,
                   diff_constant (viscosity nu), and boundary_conditions
    :return: updated glob list with evolved positions and unchanged u values
    """
    dt  = config.time_step
    nu  = config.diff_constant
    L   = config.domain_size
    bc  = config.boundary_conditions
    n   = len(globs)
    if n == 0:
        return globs

    positions = np.array([g['position'] for g in globs], dtype=float)
    u_vals    = np.array(
        [g['value'][0] if isinstance(g['value'], list) else float(g['value'])
         for g in globs],
        dtype=float,
    )

    sigma = np.sqrt(2.0 * nu * dt)
    bc_left_type  = bc['LEFT']['type'].lower()
    bc_right_type = bc['RIGHT']['type'].lower()

    for _ in range(int(config.total_time / dt)):
        # Step 1: Lagrangian advection along characteristics (u_t + u*u_x = 0 part).
        positions += u_vals * dt

        # Step 2: GRW diffusion (nu * u_xx part) — Brownian walk with alpha = nu.
        positions += np.random.normal(0.0, sigma, size=n)

        # Step 3: Boundary reflection.
        mask = positions < 0.0
        if np.any(mask):
            positions[mask] = -positions[mask]
            if bc_left_type == 'neumann':
                u_vals[mask] = -u_vals[mask]

        mask = positions > L
        if np.any(mask):
            positions[mask] = 2.0 * L - positions[mask]
            if bc_right_type == 'neumann':
                u_vals[mask] = -u_vals[mask]

    for i, g in enumerate(globs):
        g['position'] = float(positions[i])
        g['value']    = [float(u_vals[i])]

    return globs


def simulate_burgers_fd(globs, config):
    """
    Standard explicit finite-difference solver for Burgers' equation.

    This implementation is kept as an internal reference-quality solver.
    It is used by verify_solver.py to generate high-resolution reference
    solutions for comparison with the GRW particle method above.  It is NOT
    the primary solver on the main (GRW-feasibility) branch.

    :param globs: list of dicts with 'position' (fixed) and 'value' = [u_i]
    :param config: SimulationConfig
    :return: updated glob list (positions unchanged, values updated)
    """
    dt = config.time_step
    dx = config.domain_size / config.num_points
    nu = config.diff_constant

    u = np.array([
        float(glob['value'][0]) if isinstance(glob['value'], list) and len(glob['value']) > 0
        else 0.0
        for glob in globs
    ])

    for _ in range(int(config.total_time / dt)):
        u_x  = np.gradient(u, dx)
        u_xx = np.gradient(u_x, dx)
        u   += (-u * u_x + nu * u_xx) * dt
        u[0]  = config.boundary_conditions['LEFT']['value']
        u[-1] = config.boundary_conditions['RIGHT']['value']

    for i, glob in enumerate(globs):
        glob['value'] = [float(u[i])]

    return globs
