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


def simulate_fitzhugh_nagumo_two_component(globs, config):
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


def simulate_fitzhugh_nagumo_grw(globs, config):
    """
    Thesis-faithful scalar GRW for the FitzHugh-Nagumo traveling front.

    Scalar PDE (after v-elimination for the traveling wave):
      u_t = D * u_xx + f(u)
    where f'(u) = R(u) = -3*u^2 + 2*(0.5-a)*u - a.

    Exact traveling-wave solution:
      u(x, t) = 1 / (1 + exp(-(x + theta*t - x_center) / 2))
      theta   = sqrt(2) * (0.5 - a)

    GRW gradient-side algorithm (globs represent pieces of u_x):
      Each glob carries a position x_i and a signed weight w_i.
      The field u(x) is reconstructed by sorting globs and taking a cumulative
      sum of weights: u(x_n) = sum_{i: x_i <= x_n} w_i.  This is the same
      cumulative-integration reconstruction used by the heat GRW.

      Per time step:
        1. Brownian walk:  x_i += Normal(0, sqrt(2 * D * dt))
        2. Boundary reflection: Dirichlet (preserve weight) or
                                Neumann (negate weight on crossing).
        3. Sort globs by position.
        4. Reconstruct: u_i = sum_{k=1}^{i} w_k  (cumsum in sorted order).
        5. React:   w_i += dt * R(u_i) * w_i
           R(u) = -3*u^2 + 2*(0.5-a)*u - a   (derivative of the cubic f(u)).

    Initialization:
      steady_solution IC: globs placed at inverted-logistic positions
        x_i = -2 * log(1/u_i - 1) + x_center,  u_i = (i + 0.5) / N0
        with uniform weights w_i = 1 / N0.
      discontinuous IC: all N0 globs at x=x_center, w_i = 1/N0.
      nonsmooth IC: linear-ramp inverse, w_i = 1/N0.

    Reconstruction at output times uses a Gaussian-smoothed binned estimate
    to reduce particle noise before reporting/plotting.

    :param globs: list of dicts with 'position' and scalar 'value' (= w_i)
    :param config: SimulationConfig; diff_constant = D, a = threshold param,
                   time_step = dt, total_time = T, domain_size = L,
                   boundary_conditions used for position reflection.
    :return: updated globs with final sorted positions and weights
    """
    D   = config.diff_constant
    a_  = config.a if config.a is not None else 0.25
    dt  = config.time_step
    L   = config.domain_size
    bc  = config.boundary_conditions
    n   = len(globs)
    if n == 0:
        return globs

    bc_left  = bc['LEFT']['type'].lower()
    bc_right = bc['RIGHT']['type'].lower()

    # Extract positions and scalar weights.
    x = np.array([g['position'] for g in globs], dtype=float)
    w = np.array([
        (float(g['value'][0]) if isinstance(g['value'], (list, tuple))
         else float(g['value']))
        for g in globs
    ], dtype=float)

    sigma    = np.sqrt(2.0 * D * dt) if D > 0.0 else 0.0
    n_steps  = int(config.total_time / dt)

    for _ in range(n_steps):
        # Step 1: Brownian walk.
        if sigma > 0.0:
            x += np.random.normal(0.0, sigma, size=n)

        # Step 2: Boundary reflection on a finite domain.
        if L > 0.0:
            for _ in range(4):
                ml = x < 0.0
                if np.any(ml):
                    x[ml] = -x[ml]
                    if bc_left == 'neumann':
                        w[ml] = -w[ml]
                mr = x > L
                if np.any(mr):
                    x[mr] = 2.0 * L - x[mr]
                    if bc_right == 'neumann':
                        w[mr] = -w[mr]

        # Step 3: Sort globs by position.
        order = np.argsort(x, kind='stable')
        x = x[order]
        w = w[order]

        # Step 4: Reconstruct u(x_i) via cumulative sum of weights.
        # u_i = sum_{k <= i} w_k  (assumes u(-inf) = 0).
        u_cum = np.cumsum(w)

        # Step 5: React — multiplicative weight update.
        # R(u) = f'(u) = -3*u^2 + 2*(0.5 - a)*u - a
        # Applied as: w_i += dt * R(u_i) * w_i.
        #
        # Note: int_0^1 R(u) du = -1/2 - 2a, which is negative for a > 0, so
        # the total weight sum(w) is not conserved by this reaction statistic.
        # A renormalization step is applied after each react to maintain the
        # total weight at 1.0 (i.e., u(-inf) = 0, u(+inf) = 1 throughout).
        # This corrects only the mass, not the profile shape.
        R = -3.0 * u_cum**2 + 2.0 * (0.5 - a_) * u_cum - a_
        w += dt * R * w
        w_sum = float(np.sum(w))
        if abs(w_sum) > 1e-15:
            w /= w_sum

    # Write back final state (globs are in sorted order).
    for i in range(n):
        globs[i]['position'] = float(x[i])
        globs[i]['value']    = float(w[i])

    return globs


def simulate_fitzhugh_nagumo(globs, config):
    """
    Dispatcher for the FitzHugh-Nagumo solver.

    Routes to the thesis-faithful scalar GRW (simulate_fitzhugh_nagumo_grw)
    when config.fhn_ic_type indicates a thesis-style IC (steady_solution,
    nonsmooth, or discontinuous).  Falls back to the legacy two-component
    GRW-inspired particle method (simulate_fitzhugh_nagumo_two_component)
    for configs that predate the scalar formulation.

    On the GRW-feasibility (main) branch, the thesis scalar GRW is the
    primary FHN implementation.  The two-component method is retained
    for backward compatibility with legacy configs only.
    """
    ic_type = getattr(config, 'fhn_ic_type', '') or ''
    if ic_type in ('steady_solution', 'nonsmooth', 'discontinuous', 'scalar_grw'):
        return simulate_fitzhugh_nagumo_grw(globs, config)
    return simulate_fitzhugh_nagumo_two_component(globs, config)


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


def simulate_burgers_lagrangian(globs, config):
    """
    Experimental GRW-inspired Lagrangian particle method for Burgers' equation.

    Operator splitting per step:
      1. Lagrangian advection:  x_i += u_i * dt
      2. GRW diffusion:          x_i += Normal(0, sqrt(2*nu*dt))
      3. Boundary reflection:    symmetric (Dirichlet) or anti-symmetric (Neumann)

    Each glob carries:
      'position' : current particle location (evolves each step)
      'value'    : [u_i], the velocity carried by this particle (Lagrangian invariant)

    Limitations:
      - u_i is frozen (inviscid characteristic value), so the diffusion step does
        not feed back into the carried velocity.
      - Near shocks, particle clustering degrades reconstruction quality.

    This method was introduced as a feasibility experiment on the GRW branch.
    It is kept for comparison.  The thesis-preferred method is Cole-Hopf GRW.
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
        positions += u_vals * dt
        positions += np.random.normal(0.0, sigma, size=n)

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


def simulate_burgers_cole_hopf_grw(globs, config):
    """
    Thesis-faithful Burgers GRW via the Cole-Hopf transformation.

    The Cole-Hopf transform  u = -2*nu * phi_x / phi  maps Burgers' equation
      u_t + u*u_x = nu*u_xx
    into the heat equation for phi:
      phi_t = nu*phi_xx

    The GRW method therefore reduces to the heat-equation machinery (random walk
    of phi_x globs with Brownian step sigma = sqrt(2*nu*dt)).  This is the key
    advantage over the direct Burgers GRW: the reaction statistic that plagues
    the direct approach disappears entirely.

    Initialization:
      Given u0(x), compute Psi0(x) = integral_0^x u0(s) ds (trapezoidal rule),
      then phi0(x) = exp(-Psi0(x) / (2*nu)).  phi0 is normalized so its maximum
      equals 1 to avoid floating-point overflow.  Each glob i is assigned:
        position w_i = x_i
        weight   w_i = phi0_x(x_i) * dx  where phi0_x = -u0 * phi0 / (2*nu)

    Evolution:
      Apply the GRW heat random walk (random_walk + apply_boundary_conditions)
      with alpha = nu and Dirichlet BCs for phi.  Dirichlet reflection for
      phi_x globs preserves the glob weight when reflecting at the wall.  This
      corresponds to holding phi = constant at the boundary (rather than
      enforcing phi_x = 0 as Neumann would).  The consequence is that u at the
      boundary is not fixed; it evolves based on the local phi_x density.  This
      is the correct choice when u is non-zero at the domain edges (e.g. the
      traveling wave benchmark where u ≈ 2.4 at the left wall at t=0).  Use a
      domain large enough that little GRW diffusion reaches the walls during the
      simulation to minimise boundary influence on the interior reconstruction.

    Reconstruction at final time T:
      1. Bin glob weights onto a uniform output grid => phi_x density
      2. phi(x_j) = phi_left + cumsum(bin_weights)  [phi_left = 1, free const]
      3. u(x_j) = -2*nu * phi_x(x_j) / phi(x_j)

    Numerical conditioning note:
      phi0 varies as exp(-Psi0/(2*nu)).  For small nu or large u values, phi0
      can span many orders of magnitude across the domain, causing numerical
      issues.  A warning is emitted when the log_phi0 range exceeds 50.  The
      traveling-wave benchmark (nu=0.5, domain [0,4]) is designed to keep the
      conditioning manageable.

    :param globs: list of dicts 'position' and 'value' = [u_i] on a uniform grid
    :param config: SimulationConfig; diff_constant = nu, BCs used for phi_x walk
    :return: updated globs with final u(x,T) on a uniform output grid
    """
    nu = config.diff_constant
    dt = config.time_step
    L  = config.domain_size
    N  = len(globs)
    if N == 0:
        return globs

    # Extract u0 sorted by position.
    order = np.argsort([g['position'] for g in globs])
    x0 = np.array([globs[i]['position'] for i in order])
    u0 = np.array([
        (globs[i]['value'][0] if isinstance(globs[i]['value'], list)
         else float(globs[i]['value']))
        for i in order
    ])

    # Psi0(x) = integral_0^x u0(s) ds  (trapezoidal)
    Psi0 = np.zeros(N)
    for i in range(1, N):
        Psi0[i] = Psi0[i - 1] + 0.5 * (u0[i - 1] + u0[i]) * (x0[i] - x0[i - 1])

    # log_phi0 = -Psi0 / (2*nu); normalize so max = 0 => phi0_max = 1.
    log_phi0 = -Psi0 / (2.0 * nu)
    log_range = log_phi0.max() - log_phi0.min()
    if log_range > 50.0:
        print(
            f"  [Cole-Hopf GRW] WARNING: log_phi0 range = {log_range:.1f}  "
            f"(nu={nu}, L={L}).  phi spans exp({log_range:.1f}); numerical "
            f"conditioning may be poor.  Consider larger nu or smaller domain."
        )
    log_phi0 -= log_phi0.max()
    phi0 = np.exp(np.clip(log_phi0, -700.0, 0.0))

    # Initialize phi_x globs using forward differences of phi0.
    #
    # weight_i = phi0(x_{i+1}) - phi0(x_i), placed at midpoint x_{i+1/2}.
    # This is exact: sum(weights) = phi0(x_{N-1}) - phi0(x_0) = phi0(L) - 1.
    # Because phi0 = exp(...) > 0, phi0(L) > 0 and phi0(L) - 1 > -1, so the
    # reconstruction phi_out = 1 + cumsum(bin_sums) stays positive for all x.
    #
    # Using phi0_x * dx (a rectangle rule) instead would introduce a numerical
    # integral error that can make sum(weights) < -1, forcing phi_out negative
    # and producing extreme u values near the right boundary.
    w_diff = np.diff(phi0)               # N-1 values; sum = phi0(L) - phi0(0)
    x_mid  = 0.5 * (x0[:-1] + x0[1:])  # N-1 midpoints
    n_phi  = len(w_diff)                 # = N - 1
    phi_globs = [
        {'position': float(x_mid[i]), 'value': float(w_diff[i])}
        for i in range(n_phi)
    ]

    # GRW heat walk on phi_x globs with PERIODIC wrapping.
    #
    # The traveling wave IC has a non-zero mean (u -> 1 + sqrt(2) as x -> -inf),
    # which causes Psi0(x) to grow linearly and phi0(x) to decay exponentially
    # across [0, L].  Under Dirichlet (symmetric) reflection, large-magnitude
    # negative-weight globs near x=0 pile up at the left wall, causing phi_out
    # to collapse near x=0 and corrupt the reconstruction.  Neumann (weight-
    # negating) reflection at x=0 is also wrong: it enforces phi_x=0 at x=0,
    # which forces u=0 there -- inconsistent with the traveling wave where u!=0
    # at the domain edges.
    #
    # The physical BC for the Cole-Hopf phi on the INFINITE line requires no
    # wall at all.  On a finite domain we approximate this with periodic wrapping:
    # globs that exit at x=0 re-enter at x=L (and vice versa).  This:
    #   (a) eliminates reflective pile-up near the walls,
    #   (b) preserves all glob weights (total integral unchanged),
    #   (c) pushes any remaining "seam" artifact to x=L, which is far from the
    #       wave center throughout the simulation on a sufficiently large domain.
    # The sum(bin_sums) = phi0(L)-phi0(0) is preserved exactly, so
    # phi_out[-1] = phi0(L) remains correct.
    # GRW heat walk on phi_x globs with Dirichlet (weight-preserving) reflection.
    #
    # Choice of boundary rule for phi_x globs:
    #   Dirichlet reflection: glob crosses boundary -> mirror position back,
    #   preserve weight.  This approximates Neumann (zero-flux) BC for phi,
    #   i.e. phi_x = 0 at the walls.  On a domain large relative to sigma_T,
    #   the boundary artifacts are confined to a narrow layer near each wall.
    #
    # Alternative rules (absorbing, periodic, Neumann weight-negation) were
    # tested and produced larger artifacts for ICs with non-zero phi_x at walls.
    # Dirichlet reflection is retained as the simplest thesis-consistent choice.
    n_steps = int(config.total_time / dt)
    x_ph = np.array([g['position'] for g in phi_globs], dtype=float)
    w_ph = np.array([g['value']    for g in phi_globs], dtype=float)
    sigma_step = np.sqrt(2.0 * nu * dt)
    for _ in range(n_steps):
        x_ph += np.random.normal(0.0, sigma_step, size=x_ph.shape)
        # Dirichlet reflection (may need multiple passes for large steps).
        for _ in range(4):
            ml = x_ph < 0.0;   x_ph[ml] = -x_ph[ml]
            mr = x_ph > L;     x_ph[mr] = 2.0 * L - x_ph[mr]

    # Reconstruct phi and u on a uniform N-point output grid.
    x_out  = np.linspace(0.0, L, N)
    dx_out = L / (N - 1)

    # Bin phi_x weights onto the N-point output grid.
    # Use floor-based nearest-left-neighbour assignment: glob at x goes to bin j
    # where x_out[j] <= x < x_out[j+1].  This avoids the paired-up bias of
    # np.round (which uses banker's rounding and causes alternate bins to be empty).
    bin_sums = np.zeros(N)
    idx = np.clip(np.floor(x_ph / dx_out).astype(int), 0, N - 1)
    np.add.at(bin_sums, idx, w_ph)

    # Smooth bin_sums with a Gaussian kernel to suppress GRW particle noise.
    #
    # With ~1 glob per bin, individual bins fluctuate substantially; the
    # smoothing reduces variance in the bin counts by ~ sqrt(n_kernel_bins).
    # The kernel sigma (sigma_bins=8) is chosen to span enough bins to average
    # down the noise while staying much narrower than the phi variation scale
    # (~sqrt(nu)/dx_out = 70 bins for nu=0.5, dx=0.01).  After smoothing,
    # the total is rescaled to preserve the exact integral phi0(L)-phi0(0).
    sigma_bins = 8
    kw         = int(4 * sigma_bins) + 1
    kernel_x   = np.arange(-kw, kw + 1, dtype=float)
    kernel     = np.exp(-0.5 * (kernel_x / sigma_bins) ** 2)
    kernel    /= kernel.sum()
    bin_sums_s = np.convolve(bin_sums, kernel, mode='same')
    exact_sum  = bin_sums.sum()
    # Rescale to preserve the total integral phi0(L)-phi0(0) = exact_sum.
    # Skip rescaling when exact_sum ~= 0 (symmetric ICs where phi0(L)=phi0(0))
    # to avoid dividing by near-zero and zeroing out bin_sums_s.
    if abs(bin_sums_s.sum()) > 1e-30 and abs(exact_sum) > 1e-10 * max(np.abs(bin_sums).max(), 1e-30):
        bin_sums_s *= exact_sum / bin_sums_s.sum()

    # phi_x density on output grid (smoothed).
    phi_x_out = bin_sums_s / dx_out

    # phi = phi_left + integral phi_x (cumulative sum of smoothed bin weights).
    # phi_left = 1 (free normalization; u = -2*nu*phi_x/phi is scale-invariant).
    phi_out = 1.0 + np.cumsum(bin_sums_s)

    # Physical floor: phi cannot drop below phi0_min (minimum of the initial phi)
    # due to the maximum principle for the heat equation.  GRW particle noise can
    # push the reconstructed phi below phi0_min in sparse regions; we clip to
    # phi0_min/2 as a safeguard and zero out u at those bins.
    phi0_min  = float(phi0.min())
    phi_floor = max(phi0_min / 2.0, 1e-10)
    phi_clipped = phi_out < phi_floor
    if np.any(phi_clipped):
        n_bad = int(phi_clipped.sum())
        print(
            f"  [Cole-Hopf GRW] NOTE: {n_bad} output bins have phi < phi_floor "
            f"({phi_floor:.2e}); phi0_min={phi0_min:.2e}.  "
            f"u is set to 0 at those bins (Monte Carlo noise near phi minimum)."
        )
    phi_safe = np.where(phi_clipped, phi_floor, phi_out)

    u_out = -2.0 * nu * phi_x_out / phi_safe
    u_out = np.where(phi_clipped, 0.0, u_out)

    for i in range(N):
        globs[i]['position'] = float(x_out[i])
        globs[i]['value']    = [float(u_out[i])]

    return globs


def simulate_burgers_direct_grw(globs, config):
    """
    Diagnostic direct Burgers GRW — thesis Section 5, gradient-variable approach.

    THIS IS NOT THE PRIMARY SOLVER.  It is included to reproduce the thesis
    observation that the direct GRW method for Burgers is severely noisy and
    impractical as a primary solver.

    Mathematical background:
      Differentiating Burgers  u_t + u*u_x = nu*u_xx  w.r.t. x gives the
      evolution equation for v = u_x:
        v_t = nu*v_xx  -  u*v_x  -  v^2

      Dividing the non-diffusion terms by v yields the per-glob reaction statistic:
        R(u) = -(u * u_xx / u_x  +  u_x)
             = -(u * v_x / v     +  v)

      In GRW, globs representing v = u_x evolve via:
        1. Brownian random walk (for nu*v_xx): x_i += Normal(0, sqrt(2*nu*dt))
        2. Lagrangian advection (for -u*v_x):  x_i += u(x_i) * dt
        3. Reaction (for -v^2):                w_i *= exp(R(x_i) * dt)

    Why the direct method fails (reproducing the thesis conclusion):
      - u_xx = dv/dx requires two numerical differentiations of a noisy particle
        field, amplifying statistical noise at each step.
      - Division by v in R(u) further amplifies errors when v is small.
      - The reaction term rapidly drives glob weights to extreme values.
      - These instabilities are NOT suppressed here; they are the intended output
        of this diagnostic path.

    :param globs: list of dicts 'position' and 'value' = [u_i] on a uniform grid
    :param config: SimulationConfig
    :return: updated globs with reconstructed u(x,T) on uniform output grid
    """
    nu = config.diff_constant
    dt = config.time_step
    L  = config.domain_size
    bc = config.boundary_conditions
    bc_left_type  = bc['LEFT']['type'].lower()
    bc_right_type = bc['RIGHT']['type'].lower()
    N  = len(globs)
    if N == 0:
        return globs

    # Extract u0 sorted by position.
    order = np.argsort([g['position'] for g in globs])
    x0 = np.array([globs[i]['position'] for i in order])
    u0 = np.array([
        (globs[i]['value'][0] if isinstance(globs[i]['value'], list)
         else float(globs[i]['value']))
        for i in order
    ])

    dx0 = L / (N - 1) if N > 1 else 1.0
    uL  = float(bc['LEFT'].get('value', 0.0))  # left Dirichlet value for u reconstruction

    # v0 = du0/dx; v-glob weights = v0 * dx (pieces of the v distribution).
    v0          = np.gradient(u0, dx0)
    v_positions = x0.copy()
    v_weights   = v0 * dx0

    sigma   = np.sqrt(2.0 * nu * dt)
    n_steps = int(config.total_time / dt)
    n_grid  = N
    x_grid  = np.linspace(0.0, L, n_grid)
    dx_grid = L / (n_grid - 1)

    for _ in range(n_steps):
        # Reconstruct v and u on grid from v-globs.
        bin_v = np.zeros(n_grid)
        idx_v = np.clip(
            np.floor(v_positions / dx_grid).astype(int), 0, n_grid - 1
        )
        np.add.at(bin_v, idx_v, v_weights)
        v_grid = bin_v / dx_grid               # v(x) = u_x density
        u_grid = uL + np.cumsum(bin_v)         # u = uL + integral v dx

        # Reaction statistic R = -(u * u_xx / v + v); minimal zero-division guard.
        u_xx_grid = np.gradient(v_grid, dx_grid)
        eps_div = 1.0e-15
        v_safe  = np.where(
            np.abs(v_grid) > eps_div,
            v_grid,
            np.sign(v_grid + 1e-300) * eps_div
        )
        R_grid = -(u_grid * u_xx_grid / v_safe + v_grid)

        # Interpolate R and u to glob positions.
        R_at_globs = np.interp(v_positions, x_grid, R_grid,
                                left=R_grid[0], right=R_grid[-1])
        u_at_globs = np.interp(v_positions, x_grid, u_grid,
                                left=u_grid[0], right=u_grid[-1])

        # Apply reaction; clamp exponent to prevent immediate blow-up.
        v_weights   *= np.exp(np.clip(R_at_globs * dt, -20.0, 20.0))
        # Lagrangian advection (characteristics move at local u).
        v_positions += u_at_globs * dt
        # GRW diffusion.
        v_positions += np.random.normal(0.0, sigma, size=N)

        # Boundary reflection for v-globs.
        mask_l = v_positions < 0.0
        if np.any(mask_l):
            v_positions[mask_l] = -v_positions[mask_l]
            if bc_left_type == 'neumann':
                v_weights[mask_l] = -v_weights[mask_l]

        mask_r = v_positions > L
        if np.any(mask_r):
            v_positions[mask_r] = 2.0 * L - v_positions[mask_r]
            if bc_right_type == 'neumann':
                v_weights[mask_r] = -v_weights[mask_r]

    # Final reconstruction: u on uniform output grid.
    x_out     = x_grid
    bin_final = np.zeros(n_grid)
    idx_f     = np.clip(
        np.floor(v_positions / dx_grid).astype(int), 0, n_grid - 1
    )
    np.add.at(bin_final, idx_f, v_weights)
    u_out = uL + np.cumsum(bin_final)

    for i in range(N):
        globs[i]['position'] = float(x_out[i])
        globs[i]['value']    = [float(u_out[i])]

    return globs


def simulate_burgers(globs, config):
    """
    Dispatcher for Burgers GRW solvers.

    Routes to the appropriate implementation based on config.burgers_mode:

      'cole_hopf_grw'  (default) — thesis-faithful GRW via Cole-Hopf transformation.
                                    Reduces Burgers to a heat equation solved by GRW.
      'direct_grw'               — diagnostic direct gradient-variable GRW.
                                    Noisy by design; reproduces thesis noise discussion.
      'lagrangian_grw'           — experimental Lagrangian particle method (operator
                                    splitting: advection + GRW diffusion).

    :param globs: list of dicts 'position' and 'value' = [u_i]
    :param config: SimulationConfig with burgers_mode attribute
    :return: updated globs
    """
    mode = (getattr(config, 'burgers_mode', None) or 'cole_hopf_grw').strip().lower()
    if mode == 'direct_grw':
        return simulate_burgers_direct_grw(globs, config)
    elif mode in ('lagrangian_grw', 'lagrangian'):
        return simulate_burgers_lagrangian(globs, config)
    else:
        return simulate_burgers_cole_hopf_grw(globs, config)


def simulate_burgers_fd(globs, config):
    """
    Stable explicit finite-difference solver for Burgers' equation.

    This implementation is kept as an internal reference-quality solver.
    It is used by verify_solver.py to generate high-resolution reference
    solutions for comparison with the GRW particle method above.  It is NOT
    the primary solver on the main (GRW-feasibility) branch.

    Stability: the internal time step is sub-cycled to satisfy both the
    diffusion CFL condition  dt_inner <= dx^2 / (2*nu)  and the advection
    CFL condition  dt_inner <= dx / (|u|_max + eps).  The external dt from
    config is used only to set the macro time interval; the inner step is
    chosen automatically, guaranteeing stability for any nu and IC.

    :param globs: list of dicts with 'position' (fixed) and 'value' = [u_i]
    :param config: SimulationConfig
    :return: updated glob list (positions unchanged, values updated)
    """
    dx = config.domain_size / (config.num_points - 1)
    nu = config.diff_constant
    T  = config.total_time
    bc_left  = float(config.boundary_conditions['LEFT']['value'])
    bc_right = float(config.boundary_conditions['RIGHT']['value'])

    u = np.array([
        float(glob['value'][0]) if isinstance(glob['value'], list) and len(glob['value']) > 0
        else 0.0
        for glob in globs
    ], dtype=float)
    u[0]  = bc_left
    u[-1] = bc_right

    t = 0.0
    while t < T:
        # Adaptive inner step: satisfy diffusion and advection CFL simultaneously.
        dt_diff = 0.4 * dx ** 2 / nu if nu > 0.0 else np.inf
        dt_adv  = 0.4 * dx / (np.abs(u).max() + 1e-12)
        dt_inner = min(dt_diff, dt_adv, T - t)
        if dt_inner <= 0.0:
            break
        u_x  = np.gradient(u, dx)
        u_xx = np.gradient(u_x, dx)
        u   += (-u * u_x + nu * u_xx) * dt_inner
        u[0]  = bc_left
        u[-1] = bc_right
        t += dt_inner

    for i, glob in enumerate(globs):
        glob['value'] = [float(u[i])]

    return globs
