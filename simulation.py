import os

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


def simulate_fitzhugh_nagumo_grw(globs, config, _diag_dir=None):
    """
    Thesis-faithful scalar GRW for the FitzHugh-Nagumo traveling front.

    Scalar PDE:
      u_t = D * u_xx + f(u)

    Exact traveling-wave solution:
      u(x, t) = 1 / (1 + exp(-(x + theta*t - x_center) / 2))
      theta   = sqrt(2) * (0.5 - a)

    Reaction statistic R(u) = f'(u), derived by requiring the sigmoid above to
    be an exact solution of the PDE.  Substituting u = 1/(1+exp(-xi/2)) gives:

      f(u) = u*(1-u) * [theta/2 - D*(1-2*u)/4]
      R(u) = f'(u) = -(3D/2)*u^2 + (3D/2 - theta)*u + (theta/2 - D/4)

    Key property: integral_0^1 R(u) du = f(1) - f(0) = 0, so the total glob
    weight is conserved by this reaction.  No per-step renormalization is
    needed or applied.

    GRW gradient-side algorithm (globs represent pieces of u_x):
      Each glob carries a position x_i and a signed weight w_i.
      The field u(x) is reconstructed by sorting globs and taking a cumulative
      sum of weights: u(x_n) = sum_{i: x_i <= x_n} w_i.

      Per time step:
        1. Brownian walk:  x_i += Normal(0, sqrt(2 * D * dt))
           The Brownian step uses variance 2 * D * dt (thesis convention).
        2. Boundary reflection: Dirichlet (preserve weight) or
                                Neumann (negate weight on crossing).
        3. Sort globs by position.
        4. Reconstruct: u_i = sum_{k=1}^{i} w_k  (cumsum in sorted order).
        5. React:   w_i += dt * R(u_i) * w_i
           where R(u) = -(3D/2)*u^2 + (3D/2 - theta)*u + (theta/2 - D/4).

    Initialization:
      steady_solution IC: globs placed at inverted-logistic positions
        x_i = -2 * log(1/u_i - 1) + x_center,  u_i = (i + 0.5) / N0
        with uniform weights w_i = 1 / N0.
      discontinuous IC: all N0 globs at x=x_center, w_i = 1/N0.
      nonsmooth IC: linear-ramp inverse, w_i = 1/N0.

    :param globs:     list of dicts with 'position' and scalar 'value' (= w_i)
    :param config:    SimulationConfig; diff_constant = D, a = threshold param,
                      time_step = dt, total_time = T, domain_size = L,
                      boundary_conditions used for position reflection.
    :param _diag_dir: optional path; if set, saves a diagnostic figure with
                      front-location vs time and per-snapshot weight profiles.
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

    theta    = np.sqrt(2.0) * (0.5 - a_)
    bc_left  = bc['LEFT']['type'].lower()
    bc_right = bc['RIGHT']['type'].lower()

    x = np.array([g['position'] for g in globs], dtype=float)
    w = np.array([
        (float(g['value'][0]) if isinstance(g['value'], (list, tuple))
         else float(g['value']))
        for g in globs
    ], dtype=float)

    sigma   = np.sqrt(2.0 * D * dt) if D > 0.0 else 0.0
    n_steps = int(config.total_time / dt)

    # Reaction coefficients: R(u) = c2*u^2 + c1*u + c0
    # Derived from f(u) = u*(1-u)*[theta/2 - D*(1-2u)/4].
    # integral_0^1 R(u) du = 0 => total weight is conserved.
    c2 = -1.5 * D
    c1 =  1.5 * D - theta
    c0 =  0.5 * theta - 0.25 * D

    # Optional diagnostics setup.
    if _diag_dir is not None:
        _snap_at = {0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1}
        _snaps: dict = {}
        _front_t:   list = []
        _front_loc: list = []
        # Record initial front (t=0) from the un-stepped sorted globs.
        _ord0 = np.argsort(x)
        _uc0  = np.cumsum(w[_ord0])
        _idx0 = int(np.clip(np.searchsorted(_uc0, 0.5), 0, n - 1))
        _x_center_init = float(np.sort(x)[_idx0])

    for step in range(n_steps):
        # Step 1: Brownian walk.  Variance = 2 * D * dt (thesis convention).
        if sigma > 0.0:
            x += np.random.normal(0.0, sigma, size=n)

        # Step 2: Boundary reflection on a finite domain.
        #   Dirichlet: reflect position, preserve weight.
        #   Neumann:   reflect position, negate weight.
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

        # Step 4: Reconstruct u(x_i) via cumulative sum.
        # u_i = sum_{k <= i} w_k  (the GRW integration of the gradient globs).
        u_cum = np.cumsum(w)

        # Step 5: Multiplicative reaction update.
        # R(u) = -(3D/2)*u^2 + (3D/2-theta)*u + (theta/2-D/4)
        # Total weight is conserved to O(dt^2); no renormalization is applied.
        R = c2 * u_cum**2 + c1 * u_cum + c0
        w += dt * R * w

        # Diagnostic: track front location and record snapshots.
        if _diag_dir is not None:
            u_post = np.cumsum(w)
            idx_f  = int(np.clip(np.searchsorted(u_post, 0.5), 0, n - 1))
            _front_t.append((step + 1) * dt)
            _front_loc.append(float(x[idx_f]))
            if step in _snap_at:
                _snaps[step] = (x.copy(), w.copy(), u_post.copy())

    for i in range(n):
        globs[i]['position'] = float(x[i])
        globs[i]['value']    = float(w[i])

    if _diag_dir is not None:
        _save_fhn_grw_diagnostics(
            _diag_dir, _front_t, _front_loc, _snaps,
            _x_center_init, theta, a_, D, config.total_time,
        )

    return globs


def _save_fhn_grw_diagnostics(diag_dir, front_t, front_loc, snaps,
                               x_center_init, theta, a_, D, T):
    """Save a 4-panel diagnostic figure for the FHN scalar GRW run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(diag_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"FHN scalar GRW diagnostics  "
        f"(a={a_:.4g}, D={D:.4g}, theta={theta:.4g})",
        fontsize=12,
    )
    ax = axes.flat

    # --- Panel 0: Front location vs time ---
    t_arr  = np.asarray(front_t)
    fl_arr = np.asarray(front_loc)
    exact_front = x_center_init - theta * t_arr
    ax[0].plot(t_arr, fl_arr,      'b-',  lw=0.7, alpha=0.8, label='GRW front')
    ax[0].plot(t_arr, exact_front, 'r--', lw=1.5, label=f'Exact (speed={theta:.4g})')
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('front x  (u = 0.5 crossing)')
    ax[0].legend(fontsize=9)
    ax[0].set_title('Front location vs time')

    # --- Panel 1: Front location error vs time ---
    ax[1].plot(t_arr, fl_arr - exact_front, 'g-', lw=0.8)
    ax[1].axhline(0, color='k', lw=0.5, ls='--')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('GRW front − exact front')
    ax[1].set_title('Front location error vs time')

    # --- Panel 2: Weight profiles at snapshot times ---
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for k, (step, (xs, ws, uc)) in enumerate(sorted(snaps.items())):
        t_s = (step + 1) * T / max(len(front_t), 1)
        ax[2].plot(xs, ws, '.', ms=2, color=colors[k % len(colors)],
                   alpha=0.6, label=f't≈{t_s:.1f}')
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('glob weight w_i')
    ax[2].legend(fontsize=8)
    ax[2].set_title('Glob weights at snapshot times')

    # --- Panel 3: Reconstructed u at snapshot times ---
    for k, (step, (xs, ws, uc)) in enumerate(sorted(snaps.items())):
        t_s = (step + 1) * T / max(len(front_t), 1)
        ax[3].plot(xs, uc, '-', lw=1.0, color=colors[k % len(colors)],
                   alpha=0.8, label=f't≈{t_s:.1f}')
    ax[3].axhline(0.5, color='k', lw=0.5, ls='--')
    ax[3].set_xlabel('x')
    ax[3].set_ylabel('cumsum(w)  ≈ u(x)')
    ax[3].legend(fontsize=8)
    ax[3].set_title('Reconstructed u(x) at snapshot times')

    fig.tight_layout()
    out_path = os.path.join(diag_dir, 'fhn_grw_diagnostics.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [diag] Saved FHN GRW diagnostics -> {out_path}")


def simulate_fitzhugh_nagumo(globs, config):
    """
    FitzHugh-Nagumo solver for the GRW-feasibility branch.

    Always routes to the thesis-faithful scalar GRW
    (simulate_fitzhugh_nagumo_grw).  The FD reference solver and the legacy
    two-component particle method are available on the mixed-solvers-validation
    branch only.
    """
    return simulate_fitzhugh_nagumo_grw(
        globs, config,
        _diag_dir=getattr(config, '_diag_dir', None),
    )


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


def _save_cole_hopf_diagnostics(diag_dir, x0, u0, phi0, x_out, dx_out,
                                 bin_sums_raw, bin_sums_s, phi_out, u_out):
    """
    Save a multi-panel diagnostic figure for the Cole-Hopf GRW intermediate
    quantities.  Called when simulate_burgers_cole_hopf_grw receives _diag_dir.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(diag_dir, exist_ok=True)

    phi_x_raw      = bin_sums_raw / dx_out
    phi_x_smoothed = bin_sums_s   / dx_out
    phi_x_over_phi = phi_x_smoothed / np.maximum(np.abs(phi_out), 1e-10)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Cole-Hopf GRW: Intermediate Diagnostics", fontsize=10)
    panels = [
        (x0,    u0,              "u0(x)  (initial Burgers field)",      "u0"),
        (x0,    phi0,            "phi0(x) = exp(-Psi0/(2*nu))",          "phi0"),
        (x_out, phi_x_raw,       "phi_x raw  (binned, before smooth)",   "phi_x raw"),
        (x_out, phi_x_smoothed,  "phi_x smoothed + corrected",           "phi_x"),
        (x_out, phi_out,         "phi(x, T)  reconstructed",             "phi"),
        (x_out, phi_x_over_phi,  "phi_x / phi  (before -2*nu factor)",   "phi_x/phi"),
        (x_out, u_out,           "u(x, T)  Cole-Hopf GRW output",        "u"),
        (x_out, -2.0 * (x_out[1] - x_out[0]) / dx_out *
                phi_x_over_phi,  "u_out redundant check",                "check"),
    ]
    colors = ['steelblue', 'darkgreen', 'gray', 'darkorange',
              'darkgreen', 'purple', 'crimson', 'black']
    for ax, (x, y, title, ylabel), c in zip(axes.flat, panels, colors):
        ax.plot(x, y, color=c, linewidth=1.2)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(useOffset=False, axis='y', style='plain')

    plt.tight_layout()
    path = os.path.join(diag_dir, "cole_hopf_diagnostics.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [Cole-Hopf] Saved diagnostics -> {path}")


def simulate_burgers_cole_hopf_grw(globs, config, _diag_dir=None):
    """
    Thesis-faithful Burgers GRW via the Cole-Hopf transformation.

    The Cole-Hopf transform  u = -2*nu * phi_x / phi  maps Burgers' equation
      u_t + u*u_x = nu*u_xx
    into the heat equation for phi:
      phi_t = nu*phi_xx

    The GRW method therefore reduces to the heat-equation machinery: a Brownian
    random walk of phi_x globs with step sigma = sqrt(2*nu*dt).

    Initialization:
      Given u0(x), compute Psi0(x) = integral_0^x u0(s) ds (trapezoidal rule),
      then phi0(x) = exp(-Psi0(x) / (2*nu)), normalized so phi0_max = 1.
      Each phi_x glob is initialized at midpoint x_{i+1/2} with weight
        w_i = phi0(x_{i+1}) - phi0(x_i)
      so sum(weights) = phi0(L) - phi0(0) exactly.

    Evolution:
      Brownian random walk followed by Dirichlet (weight-preserving, position-
      mirroring) boundary reflection.  Dirichlet reflection implements Neumann
      BC for the phi_x density (zero flux at walls), meaning phi_x = 0 at walls
      on average.  The domain should be large relative to the diffusion length
      sqrt(2*nu*T) to minimize wall artifacts on the interior solution.

    Reconstruction at final time T:
      1. Bin glob weights onto a uniform N-point output grid.
      2. Smooth with a Gaussian kernel (sigma_bins=8) to reduce particle noise.
      3. Enforce the correct total for the smoothed bins:
           - Symmetric IC (phi0(L)=phi0(0), exact_integral=0):
             subtract the mean of bin_sums_s to enforce zero total exactly.
             Gaussian convolution with mode='same' truncates the kernel near
             both edges, causing the smoothed sum to drift away from zero.
             Without this correction phi_out[-1] = phi0(0) + (non-zero sum)
             != phi0(L), producing domain-wide systematic drift.
           - Asymmetric IC (phi0(L)!=phi0(0)):
             proportional rescale so sum(bin_sums_s) = phi0(L) - phi0(0).
      4. phi(x_j) = phi0(0) + cumsum(smoothed_bins)
      5. u(x_j)   = -2*nu * phi_x(x_j) / phi(x_j)

    :param globs: list of dicts 'position' and 'value' = [u_i] on a uniform grid
    :param config: SimulationConfig; diff_constant = nu, BCs used for phi_x walk
    :param _diag_dir: optional path; if set, saves intermediate diagnostic plots
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

    # log_phi0 = -Psi0 / (2*nu); normalize so max(log_phi0) = 0 => phi0_max = 1.
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

    phi0_0 = float(phi0[0])   # left-boundary value; = 1.0 after max-normalization
    phi0_L = float(phi0[-1])  # right-boundary value
    print(f"  [Cole-Hopf] phi0(0) = {phi0_0:.6g},  phi0(L) = {phi0_L:.6g}")
    print(f"  [Cole-Hopf] min(phi0) = {float(phi0.min()):.6g},  "
          f"max(phi0) = {float(phi0.max()):.6g}")

    # exact_integral = phi0(L) - phi0(0) = sum of all phi_x glob weights.
    # For the stationary-shock IC (xc = L/2): Psi0(L) = 0, so phi0(L) = phi0(0)
    # and exact_integral = 0.
    exact_integral = phi0_L - phi0_0
    print(f"  [Cole-Hopf] exact_integral phi0(L)-phi0(0) = {exact_integral:.6e}")

    # Phi_x globs: forward finite differences of phi0 at midpoints.
    # sum(w_diff) = phi0(L) - phi0(0) = exact_integral exactly.
    w_diff = np.diff(phi0)               # N-1 weights
    x_mid  = 0.5 * (x0[:-1] + x0[1:])  # N-1 midpoint positions
    print(f"  [Cole-Hopf] sum(w_diff) = {float(w_diff.sum()):.6e}  "
          f"(should equal exact_integral)")

    # GRW heat walk on phi_x globs with Dirichlet (weight-preserving) reflection.
    #
    # When a glob crosses a domain boundary its position is mirrored back into
    # [0, L] and its weight is unchanged.  This implements zero-flux for the
    # phi_x density at the walls (equivalent to Neumann BC for phi, phi_x=0
    # there).  On a domain large relative to sqrt(2*nu*T) the wall influence
    # is negligible in the interior.
    x_ph = x_mid.copy()
    w_ph = w_diff.copy()
    sigma_step = np.sqrt(2.0 * nu * dt)
    n_steps = int(config.total_time / dt)
    for _ in range(n_steps):
        x_ph += np.random.normal(0.0, sigma_step, size=x_ph.shape)
        for _ in range(4):
            ml = x_ph < 0.0;   x_ph[ml] = -x_ph[ml]
            mr = x_ph > L;     x_ph[mr] = 2.0 * L - x_ph[mr]

    # Reconstruct phi and u on a uniform N-point output grid.
    x_out  = np.linspace(0.0, L, N)
    dx_out = x_out[1] - x_out[0]

    # Bin phi_x weights using floor-based nearest-left-neighbour assignment.
    bin_sums = np.zeros(N)
    idx = np.clip(np.floor(x_ph / dx_out).astype(int), 0, N - 1)
    np.add.at(bin_sums, idx, w_ph)

    raw_sum = float(bin_sums.sum())
    print(f"  [Cole-Hopf] raw sum(bin_sums) = {raw_sum:.6e}  "
          f"(should be ~{exact_integral:.6e})")

    # Smooth bin_sums with a Gaussian kernel to suppress GRW particle noise.
    # sigma_bins=8 spans enough bins to average down noise while staying narrower
    # than the phi variation scale (~sqrt(nu)/dx_out bins for smooth profiles).
    sigma_bins = 8
    kw         = int(4 * sigma_bins) + 1
    kernel_x   = np.arange(-kw, kw + 1, dtype=float)
    kernel     = np.exp(-0.5 * (kernel_x / sigma_bins) ** 2)
    kernel    /= kernel.sum()
    bin_sums_s = np.convolve(bin_sums, kernel, mode='same')

    smoothed_before = float(bin_sums_s.sum())
    print(f"  [Cole-Hopf] smoothed sum before correction = {smoothed_before:.6e}")

    # Enforce the correct total for the smoothed bins.
    #
    # Symmetric IC case (exact_integral = 0, e.g. stationary shock):
    #   Gaussian convolution with mode='same' truncates the kernel at both
    #   domain edges, leaking net weight and making bin_sums_s.sum() != 0.
    #   Without correction: phi_out[-1] = phi0(0) + (leaked sum) != phi0(L),
    #   causing systematic domain-wide drift.  Fix: subtract the mean of
    #   bin_sums_s so that sum(bin_sums_s) = 0 exactly.
    #
    # Asymmetric IC case (exact_integral != 0, e.g. traveling wave):
    #   Proportional rescale to restore the correct total.
    near_zero = 1e-6 * max(float(np.abs(bin_sums).max()), 1e-30)
    if abs(exact_integral) < near_zero:
        bin_sums_s -= bin_sums_s.mean()
    elif abs(bin_sums_s.sum()) > 1e-30:
        bin_sums_s *= exact_integral / bin_sums_s.sum()

    smoothed_after = float(bin_sums_s.sum())
    print(f"  [Cole-Hopf] smoothed sum after correction  = {smoothed_after:.6e}")

    # phi_x density on output grid (smoothed, corrected).
    phi_x_out = bin_sums_s / dx_out

    # phi(x_j) = phi0(0) + integral_0^{x_j} phi_x dx
    #           = phi0(0) + cumsum(bin_sums_s up to bin j).
    # phi0(0) = 1.0 after max-normalization (since Psi0(0) = 0).
    phi_out = phi0_0 + np.cumsum(bin_sums_s)

    print(f"  [Cole-Hopf] phi_out[0]  = {float(phi_out[0]):.6g}  "
          f"(expect ~{phi0_0 + float(bin_sums_s[0]):.6g})")
    print(f"  [Cole-Hopf] phi_out[-1] = {float(phi_out[-1]):.6g}  "
          f"(expect ~{phi0_L:.6g})")
    print(f"  [Cole-Hopf] min(phi_out) = {float(phi_out.min()):.6g},  "
          f"max(phi_out) = {float(phi_out.max()):.6g}")
    idx_sparse = np.argsort(phi_out)[:3]
    print(f"  [Cole-Hopf] 3 smallest phi_out at x = "
          f"{[round(float(x_out[i]), 4) for i in idx_sparse]}")

    # Physical floor: clip phi to phi0_min/2 where GRW noise pushes it too low.
    phi0_min    = float(phi0.min())
    phi_floor   = max(phi0_min / 2.0, 1e-10)
    phi_clipped = phi_out < phi_floor
    if np.any(phi_clipped):
        n_bad = int(phi_clipped.sum())
        print(
            f"  [Cole-Hopf GRW] NOTE: {n_bad} bins have phi < phi_floor "
            f"({phi_floor:.2e}); u zeroed there."
        )
    phi_safe = np.where(phi_clipped, phi_floor, phi_out)

    u_out = -2.0 * nu * phi_x_out / phi_safe
    u_out = np.where(phi_clipped, 0.0, u_out)

    print(f"  [Cole-Hopf] max(|phi_x/phi|) = "
          f"{float(np.max(np.abs(phi_x_out / phi_safe))):.6e}")

    if _diag_dir is not None:
        _save_cole_hopf_diagnostics(
            _diag_dir, x0, u0, phi0, x_out, dx_out,
            bin_sums, bin_sums_s, phi_out, u_out,
        )

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
    mode     = (getattr(config, 'burgers_mode', None) or 'cole_hopf_grw').strip().lower()
    diag_dir = getattr(config, '_diag_dir', None)
    if mode == 'direct_grw':
        return simulate_burgers_direct_grw(globs, config)
    elif mode in ('lagrangian_grw', 'lagrangian'):
        return simulate_burgers_lagrangian(globs, config)
    else:
        return simulate_burgers_cole_hopf_grw(globs, config, _diag_dir=diag_dir)


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
