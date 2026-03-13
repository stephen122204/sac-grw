import os

import numpy as np


def random_walk(globs, diff_constant, time_step):
    """
    Vectorized Brownian displacement for all heat globs.

    The GRW method evolves globs representing the gradient-side computational elements.
    Each glob's position is updated by a Gaussian displacement with mean 0 and variance
    2 * alpha * dt, i.e. sigma = sqrt(2 * alpha * dt).

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

    GRW boundary rules:
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
    Scalar GRW for the FitzHugh-Nagumo traveling front.

    Scalar PDE:
      u_t = D * u_xx + f(u)

    Exact traveling-wave solution:
      u(x, t) = 1 / (1 + exp(-(x + theta*t - x_center) / 2))
      theta = sqrt(2) * (0.5 - a)

    Reaction statistic R(u) = f'(u), derived by requiring the sigmoid above to
    be an exact solution of the PDE. Substituting u = 1/(1+exp(-xi/2)) gives:

      f(u) = u*(1-u) * [theta/2 - D*(1-2*u)/4]
      R(u) = f'(u) = -(3D/2)*u^2 + (3D/2 - theta)*u + (theta/2 - D/4)

    Key property: integral_0^1 R(u) du = f(1) - f(0) = 0, so the total glob
    weight is conserved by this reaction. No per-step renormalization is
    needed or applied.

    GRW gradient-side algorithm (globs represent pieces of u_x):
      Each glob carries a position x_i and a signed weight w_i.
      The field u(x) is reconstructed by sorting globs and taking a cumulative
      sum of weights: u(x_n) = sum_{i: x_i <= x_n} w_i.

      Per time step:
        1. Brownian walk: x_i += Normal(0, sqrt(2 * D * dt))
           The Brownian step uses variance 2 * D * dt.
        2. Boundary reflection: Dirichlet (preserve weight) or
                                Neumann (negate weight on crossing).
        3. Sort globs by position.
        4. Reconstruct: u_i = sum_{k=1}^{i} w_k (cumsum in sorted order).
        5. React: w_i += dt * R(u_i) * w_i
           where R(u) = -(3D/2)*u^2 + (3D/2 - theta)*u + (theta/2 - D/4).

    Initialization:
      steady_solution IC: globs placed at inverted-logistic positions
        x_i = -2 * log(1/u_i - 1) + x_center, u_i = (i + 0.5) / N0
        with uniform weights w_i = 1 / N0.
      discontinuous IC: all N0 globs at x=x_center, w_i = 1/N0.
      nonsmooth IC: linear-ramp inverse, w_i = 1/N0.

    :param globs: list of dicts with 'position' and scalar 'value' (= w_i)
    :param config: SimulationConfig; diff_constant = D, a = threshold param,
                      time_step = dt, total_time = T, domain_size = L,
                      boundary_conditions used for position reflection.
    :param _diag_dir: optional path; if set, saves a diagnostic figure with
                      front-location vs time and per-snapshot weight profiles.
    :return: updated globs with final sorted positions and weights
    """
    D = config.diff_constant
    a_ = config.a if config.a is not None else 0.25
    dt = config.time_step
    L = config.domain_size
    bc = config.boundary_conditions
    n = len(globs)
    if n == 0:
        return globs

    theta = np.sqrt(2.0) * (0.5 - a_)
    bc_left = bc['LEFT']['type'].lower()
    bc_right = bc['RIGHT']['type'].lower()

    x = np.array([g['position'] for g in globs], dtype=float)
    w = np.array([
        (float(g['value'][0]) if isinstance(g['value'], (list, tuple))
         else float(g['value']))
        for g in globs
    ], dtype=float)

    sigma = np.sqrt(2.0 * D * dt) if D > 0.0 else 0.0
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
        _front_t: list = []
        _front_loc: list = []
        # Record initial front (t=0) from the un-stepped sorted globs.
        _ord0 = np.argsort(x)
        _uc0 = np.cumsum(w[_ord0])
        _idx0 = int(np.clip(np.searchsorted(_uc0, 0.5), 0, n - 1))
        _x_center_init = float(np.sort(x)[_idx0])

    for step in range(n_steps):
        # Step 1: Brownian walk.  Variance = 2 * D * dt.
        if sigma > 0.0:
            x += np.random.normal(0.0, sigma, size=n)

        # Step 2: Boundary reflection on a finite domain.
        #   Dirichlet: reflect position, preserve weight.
        #   Neumann: reflect position, negate weight.
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
            idx_f = int(np.clip(np.searchsorted(u_post, 0.5), 0, n - 1))
            _front_t.append((step + 1) * dt)
            _front_loc.append(float(x[idx_f]))
            if step in _snap_at:
                _snaps[step] = (x.copy(), w.copy(), u_post.copy())

    for i in range(n):
        globs[i]['position'] = float(x[i])
        globs[i]['value'] = float(w[i])

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
    t_arr = np.asarray(front_t)
    fl_arr = np.asarray(front_loc)
    exact_front = x_center_init - theta * t_arr
    ax[0].plot(t_arr, fl_arr, 'b-', lw=0.7, alpha=0.8, label='GRW front')
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

    Always routes to the scalar GRW
    (simulate_fitzhugh_nagumo_grw). The FD reference solver and the legacy
    two-component particle method are available on the mixed-solvers-validation
    branch only.
    """
    return simulate_fitzhugh_nagumo_grw(
        globs, config,
        _diag_dir=getattr(config, '_diag_dir', None),
    )


def _reference_phi_heat_fd(phi0_interp, x_out, nu, T, phi_left, phi_right):
    """
    Explicit FD solve of phi_t = nu * phi_xx on [0, L] with Dirichlet BCs.

    phi(0, t) = phi_left (= phi0(0), constant for all t)
    phi(L, t) = phi_right (= phi0(L), constant for all t)

    These BCs match the GRW implicit treatment: weight-preserving reflection of
    phi_x globs keeps the cumsum anchor phi(0,t) = phi0_0 and phi(L,t) = phi0_L
    constant. This FD reference is therefore what the GRW converges to in the
    zero-noise limit. The gap between this reference and the infinite-domain
    exact solution quantifies the BC-mismatch (finite-domain truncation) error.
    """
    N = len(x_out)
    dx = float(x_out[1] - x_out[0])
    dt_fd = 0.4 * dx**2 / nu        # r = nu*dt/dx^2 = 0.4 < 0.5 (stable)
    n_steps = int(np.ceil(T / dt_fd))
    dt_fd = T / n_steps
    r = nu * dt_fd / dx**2

    phi = phi0_interp.copy()
    for _ in range(n_steps):
        phi_new = phi.copy()
        phi_new[1:-1] = phi[1:-1] + r * (phi[2:] - 2.0 * phi[1:-1] + phi[:-2])
        phi_new[0] = float(phi_left)
        phi_new[-1] = float(phi_right)
        phi = phi_new
    return phi


def _save_cole_hopf_diagnostics(
    diag_dir, x0, u0, phi0, x_out, phi_out,
    phi_x_out, phi_x_bins, u_out, nu,
    u_ref=None, phi_exact=None, phi_x_over_phi_exact=None,
    phi_ref_fd=None,
):
    """
    Save a 2x3 diagnostic figure that decomposes the Cole-Hopf GRW error.

    Three curves where available:
      exact_shape -- phi0(x) = cosh(...)/cosh(...); equals the infinite-domain
                      exact phi(x,T)/C(T). Shape is preserved on R.
      FD reference -- phi(x,T) from a deterministic heat FD solve with the same
                      Dirichlet BCs as the GRW (phi=phi0_0 at x=0,
                      phi=phi0_L at x=L). Zero-noise limit of the GRW.
      GRW -- stochastic Monte-Carlo reconstruction.

    Error decomposition:
      BC-mismatch = FD_ref - exact_shape (dominant: finite-domain effect)
      GRW-noise = GRW - FD_ref (secondary: particle shot noise)

    Layout:
      [0,0] phi0(x): GRW init vs exact shape
      [0,1] phi(x,T): GRW vs FD_ref vs exact_shape
      [0,2] phi_x/phi: GRW vs FD_ref vs exact
      [1,0] phi_x(T): gradient vs bins vs FD_ref vs exact
      [1,1] u(x,T): GRW vs FD_ref u vs exact
      [1,2] Error decomposition: total / BC-mismatch / GRW-noise
    """
    import matplotlib.pyplot as plt

    os.makedirs(diag_dir, exist_ok=True)

    has_exact = (phi_exact is not None) and (phi_x_over_phi_exact is not None)
    has_fd = (phi_ref_fd is not None)
    dx_out = float(x_out[1] - x_out[0])

    if has_fd:
        phi_x_fd = np.gradient(phi_ref_fd, dx_out)
        phi_safe_fd = np.where(np.abs(phi_ref_fd) < 1e-12, 1e-12, phi_ref_fd)
        ratio_fd = phi_x_fd / phi_safe_fd
        u_fd = -2.0 * nu * phi_x_fd / phi_safe_fd

    phi_safe_grw = np.where(np.abs(phi_out) < 1e-12, 1e-12, phi_out)
    ratio_grw = phi_x_out / phi_safe_grw

    phi_x_exact_arr = (phi_x_over_phi_exact * phi_exact) if has_exact else None

    c_grw = "steelblue"; c_fd = "darkorange"; c_ex = "black"
    lw_grw = 1.4; lw_fd = 1.3; lw_ex = 1.1

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Cole-Hopf GRW  |  blue=GRW   orange=FD ref (same Dirichlet BC)   "
        "black=exact (infinite domain)",
        fontsize=9,
    )

    # [0,0] phi0 initialisation
    ax = axes[0, 0]
    ax.plot(x0, phi0, color=c_grw, linewidth=lw_grw, label="GRW phi0")
    if has_exact:
        ax.plot(x0, np.interp(x0, x_out, phi_exact), color=c_ex,
                linewidth=lw_ex, linestyle="--", label="exact shape")
        ax.legend(fontsize=7)
    ax.set_title("phi0(x)  [GRW init vs exact shape]", fontsize=8)
    ax.set_ylabel("phi0", fontsize=7)

    # [0,1] phi(T) shape
    ax = axes[0, 1]
    ax.plot(x_out, phi_out, color=c_grw, linewidth=lw_grw, label="GRW")
    if has_fd:
        ax.plot(x_out, phi_ref_fd, color=c_fd, linewidth=lw_fd,
                linestyle="--", label="FD ref  (Dirichlet BCs)")
    if has_exact:
        ax.plot(x_out, phi_exact, color=c_ex, linewidth=lw_ex,
                linestyle=":", label="exact shape  (inf. domain)")
    ax.legend(fontsize=6)
    ax.set_title("phi(x,T)  [GRW vs FD ref vs exact shape]", fontsize=8)
    ax.set_ylabel("phi", fontsize=7)

    # [0,2] phi_x/phi ratio
    ax = axes[0, 2]
    ax.plot(x_out, ratio_grw, color=c_grw, linewidth=lw_grw, label="GRW")
    if has_fd:
        ax.plot(x_out, ratio_fd, color=c_fd, linewidth=lw_fd,
                linestyle="--", label="FD ref")
    if has_exact:
        ax.plot(x_out, phi_x_over_phi_exact, color=c_ex, linewidth=lw_ex,
                linestyle=":", label="exact  A/(2nu)*tanh")
    ax.legend(fontsize=6)
    ax.set_title("phi_x/phi  (determines u = -2nu*(phi_x/phi))", fontsize=8)
    ax.set_ylabel("phi_x / phi", fontsize=7)

    # [1,0] phi_x strategies
    ax = axes[1, 0]
    ax.plot(x_out, phi_x_out, color=c_grw, linewidth=lw_grw,
            label="GRW gradient(phi)")
    ax.plot(x_out, phi_x_bins, color="mediumpurple", linewidth=1.1,
            linestyle=":", label="GRW bins/dx")
    if has_fd:
        ax.plot(x_out, phi_x_fd, color=c_fd, linewidth=lw_fd,
                linestyle="--", label="FD ref phi_x")
    if has_exact:
        ax.plot(x_out, phi_x_exact_arr, color=c_ex, linewidth=lw_ex,
                linestyle=":", label="exact phi_x")
    ax.legend(fontsize=6)
    ax.set_title("phi_x(x,T): strategies vs FD ref vs exact", fontsize=8)
    ax.set_ylabel("phi_x", fontsize=7)

    # [1,1] u(T) comparison
    ax = axes[1, 1]
    ax.plot(x_out, u_out, color=c_grw, linewidth=lw_grw, label="GRW u(T)")
    if has_fd:
        ax.plot(x_out, u_fd, color=c_fd, linewidth=lw_fd,
                linestyle="--", label="FD ref u(T)")
    if u_ref is not None:
        ax.plot(x_out, u_ref, color=c_ex, linewidth=lw_ex,
                linestyle=":", label="exact u(T)")
    ax.legend(fontsize=6)
    ax.set_title("u(x,T)  [GRW vs FD ref vs exact]", fontsize=8)
    ax.set_ylabel("u", fontsize=7)

    # [1,2] Error decomposition
    ax = axes[1, 2]
    if u_ref is not None:
        err_total = u_out - u_ref
        rms_total = float(np.sqrt(np.mean(err_total**2)))
        ax.plot(x_out, err_total, color=c_grw, linewidth=lw_grw,
                label=f"GRW - exact  (rms={rms_total:.3f})")
        if has_fd:
            err_bc = u_fd - u_ref
            err_noise = u_out - u_fd
            rms_bc = float(np.sqrt(np.mean(err_bc**2)))
            rms_noise = float(np.sqrt(np.mean(err_noise**2)))
            ax.plot(x_out, err_bc, color=c_fd, linewidth=lw_fd,
                    linestyle="--",
                    label=f"FD ref - exact  BC-mismatch rms={rms_bc:.3f}")
            ax.plot(x_out, err_noise, color="mediumpurple", linewidth=1.1,
                    linestyle=":",
                    label=f"GRW - FD ref  noise rms={rms_noise:.3f}")
            print(f"  [Cole-Hopf] Error decomposition:")
            print(f"    Total   rms = {rms_total:.4f}")
            print(f"    BC-mismatch = {rms_bc:.4f}  ({100*rms_bc/rms_total:.0f}% of total)")
            print(f"    GRW noise = {rms_noise:.4f}  ({100*rms_noise/rms_total:.0f}% of total)")
    ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=6)
    ax.set_title("Error decomp: total / BC-mismatch / GRW-noise", fontsize=8)
    ax.set_ylabel("error in u", fontsize=7)

    for ax in axes.flat:
        ax.set_xlabel("x", fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    path = os.path.join(diag_dir, "cole_hopf_diagnostics.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Cole-Hopf] Saved diagnostics -> {path}")


def simulate_burgers_cole_hopf_grw(globs, config, _diag_dir=None):
    """
    Burgers GRW via the Cole-Hopf transformation.

    The Cole-Hopf transform u = -2*nu * phi_x / phi maps Burgers' equation
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
      mirroring) boundary reflection. Dirichlet reflection implements Neumann
      BC for the phi_x density (zero flux at walls), meaning phi_x = 0 at walls
      on average. The domain should be large relative to the diffusion length
      sqrt(2*nu*T) to minimize wall artifacts on the interior solution.

    Reconstruction at final time T:
      1. Bin glob weights onto a uniform N-point output grid.
      2. Smooth with a boundary-corrected Gaussian kernel (sigma_bins=12).
         The kernel is divided by its effective support at each bin so that
         truncation at x=0 and x=L does not bias the phi_x amplitude near the
         boundaries (previously the standard mode='same' convolution
         underestimated |phi_x| there by up to ~2x for sigma_bins=12).
         sigma_bins=12 spans ~30 output bins, reducing shot noise by ~sqrt(30)
         while remaining narrow relative to the phi variation scale.
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
      5. phi_x(x_j) = d(phi)/dx [np.gradient of the reconstructed phi]
         Using the derivative of phi rather than bin_sums/dx ensures phi and
         phi_x are derived from the same smooth curve, so their ratio is
         self-consistent.
      6. u(x_j) = -2*nu * phi_x(x_j) / phi(x_j) [single inverse-transform]

    :param globs: list of dicts 'position' and 'value' = [u_i] on a uniform grid
    :param config: SimulationConfig; diff_constant = nu, BCs used for phi_x walk
    :param _diag_dir: optional path; if set, saves intermediate diagnostic plots
    :return: updated globs with final u(x,T) on a uniform output grid
    """
    nu = config.diff_constant
    dt = config.time_step
    L = config.domain_size
    N = len(globs)
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
    x_mid = 0.5 * (x0[:-1] + x0[1:])  # N-1 midpoint positions
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
    x_out = np.linspace(0.0, L, N)
    dx_out = float(x_out[1] - x_out[0])

    # Bin phi_x glob weights onto the output grid (floor assignment).
    bin_sums = np.zeros(N)
    idx = np.clip(np.floor(x_ph / dx_out).astype(int), 0, N - 1)
    np.add.at(bin_sums, idx, w_ph)

    raw_sum = float(bin_sums.sum())
    print(f"  [Cole-Hopf] raw sum(bin_sums) = {raw_sum:.6e}  "
          f"(should be ~{exact_integral:.6e})")

    # Gaussian smoothing to suppress GRW particle shot noise.
    # sigma_bins=12 spans ~12*sqrt(2*pi)~30 output bins, reducing shot noise
    # by ~sqrt(30) while staying well below the phi variation scale
    # (shock width / dx_out >> 30 for well-resolved benchmarks).
    #
    # Boundary-corrected smoothing: mode='same' convolution zero-pads outside
    # [0, N-1], which truncates the kernel near both domain edges.  The result
    # underestimates |phi_x| at x=0 and x=L where the stationary shock has its
    # largest values.  Dividing by kernel_norm (the effective kernel weight at
    # each bin) corrects this boundary bias without any tunable parameter.
    sigma_bins = 12
    kw = int(4 * sigma_bins) + 1
    kernel_x = np.arange(-kw, kw + 1, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / sigma_bins) ** 2)
    kernel     /= kernel.sum()
    raw_conv = np.convolve(bin_sums, kernel, mode='same')
    kernel_norm = np.convolve(np.ones(N, dtype=float), kernel, mode='same')
    bin_sums_s = raw_conv / np.maximum(kernel_norm, 1e-12)

    smoothed_before = float(bin_sums_s.sum())
    print(f"  [Cole-Hopf] smoothed sum before correction = {smoothed_before:.6e}")

    # Enforce the correct total integral for the smoothed bins.
    #   Symmetric IC (exact_integral = 0, e.g. stationary shock):
    #     Even after boundary correction the smoothed sum may drift slightly
    #     from zero due to finite particle noise.  Subtract the mean to enforce
    #     sum(bin_sums_s) = 0 exactly, so phi_out[-1] = phi0(L) as required.
    #   Asymmetric IC (exact_integral != 0, e.g. traveling wave):
    #     Proportional rescale to restore the correct total.
    near_zero = 1e-6 * max(float(np.abs(bin_sums).max()), 1e-30)
    if abs(exact_integral) < near_zero:
        bin_sums_s -= bin_sums_s.mean()
    elif abs(bin_sums_s.sum()) > 1e-30:
        bin_sums_s *= exact_integral / bin_sums_s.sum()

    smoothed_after = float(bin_sums_s.sum())
    print(f"  [Cole-Hopf] smoothed sum after correction = {smoothed_after:.6e}")

    # Reconstruct phi by integrating the smoothed phi_x bins.
    # phi(x_j) = phi0(0) + cumsum(bin_sums_s)[j]  (right-endpoint Riemann rule).
    # The right-boundary check phi_out[-1] == phi0(L) serves as a consistency test.
    phi_out = phi0_0 + np.cumsum(bin_sums_s)

    print(f"  [Cole-Hopf] phi_out[0] = {float(phi_out[0]):.6g}  "
          f"(expect ~{float(phi0_0 + bin_sums_s[0]):.6g})")
    print(f"  [Cole-Hopf] phi_out[-1] = {float(phi_out[-1]):.6g}  "
          f"(expect ~{phi0_L:.6g})")
    print(f"  [Cole-Hopf] min(phi_out) = {float(phi_out.min()):.6g},  "
          f"max(phi_out) = {float(phi_out.max()):.6g}")

    # Two phi_x strategies — compared in the diagnostics figure, gradient used
    # for the final u reconstruction.
    #
    # Strategy A: gradient(phi_out, dx_out).
    #   phi and phi_x are derived from the same smooth curve; their ratio
    #   (hence u) is self-consistent.  Central differences at interior points
    #   give a half-bin average of adjacent smoothed bins, which is slightly
    #   smoother than a direct per-bin estimate.
    #
    # Strategy B: bin_sums_s / dx_out.
    #   Direct per-bin estimate of phi_x.  At boundary bins this equals
    #   Strategy A exactly (one-sided difference = per-bin value).  At interior
    #   bins the two strategies agree to O(dx), so they are numerically
    #   equivalent for the smooth profiles produced by sigma_bins=12.
    #
    # With boundary-corrected bin_sums_s both strategies now recover the full
    # phi_x amplitude near x=0 and x=L; previously both were biased there
    # because the kernel was truncated without normalization.
    phi_x_out = np.gradient(phi_out, dx_out)   # Strategy A (used for u)
    phi_x_bins = bin_sums_s / dx_out             # Strategy B (shown in diagnostics)

    # Safety floor for phi in case GRW noise pushes phi below phi0_min/2.
    phi0_min = float(phi0.min())
    phi_floor = max(phi0_min / 2.0, 1e-10)
    phi_clipped = phi_out < phi_floor
    if np.any(phi_clipped):
        n_bad = int(phi_clipped.sum())
        print(f"  [Cole-Hopf] NOTE: {n_bad} bins have phi < {phi_floor:.2e}; "
              f"u zeroed there.")
    phi_safe = np.where(phi_clipped, phi_floor, phi_out)

    # Single inverse-transform: u = -2*nu * phi_x / phi.
    u_out = -2.0 * nu * phi_x_out / phi_safe
    u_out = np.where(phi_clipped, 0.0, u_out)

    print(f"  [Cole-Hopf] max(|phi_x/phi|) = "
          f"{float(np.max(np.abs(phi_x_out / phi_safe))):.6e}")

    # Exact transformed quantities for stationary-shock IC.
    # phi0_exact(x) = cosh(A*(x-xc)/(2*nu)) / cosh(A*xc/(2*nu))
    #   — same normalisation as the code's phi0: max = 1 at x=0 and x=L.
    # phi_x_over_phi_exact = A/(2*nu) * tanh(A*(x-xc)/(2*nu))
    #   — time-independent (the C(T) scaling cancels in the ratio).
    ic_type_ = (getattr(config, 'burgers_ic_type', '') or '').lower()
    amplitude_ = getattr(config, 'burgers_ic_amplitude', None)
    phi_exact = None
    phi_x_over_phi_exact = None
    if ic_type_ == 'stationary_shock' and amplitude_ is not None:
        A_ = float(amplitude_)
        xc_ = L / 2.0
        arg_ = A_ * (x_out - xc_) / (2.0 * nu)
        denom_ = float(np.cosh(A_ * xc_ / (2.0 * nu)))
        phi_exact = np.cosh(arg_) / denom_
        phi_x_over_phi_exact = (A_ / (2.0 * nu)) * np.tanh(arg_)
        u_ref_exact = -2.0 * nu * phi_x_over_phi_exact
    else:
        u_ref_exact = None

    if _diag_dir is not None:
        # Reference FD heat solve with the same Dirichlet BCs as the GRW.
        # phi(0,t) = phi0_0 and phi(L,t) = phi0_L are constant: this is what
        # weight-preserving reflection enforces in the cumsum reconstruction.
        phi0_on_xout = np.interp(x_out, x0, phi0)
        phi_ref_fd = _reference_phi_heat_fd(
            phi0_on_xout, x_out, nu, config.total_time, phi0_0, phi0_L,
        )
        _save_cole_hopf_diagnostics(
            _diag_dir, x0, u0, phi0, x_out, phi_out,
            phi_x_out, phi_x_bins,
            u_out, nu,
            u_ref=u_ref_exact,
            phi_exact=phi_exact,
            phi_x_over_phi_exact=phi_x_over_phi_exact,
            phi_ref_fd=phi_ref_fd,
        )

    for i in range(N):
        globs[i]['position'] = float(x_out[i])
        globs[i]['value'] = [float(u_out[i])]

    return globs


def simulate_burgers(globs, config):
    """
    Cole-Hopf GRW solver for Burgers' equation.

    Reduces Burgers to a heat equation via the Cole-Hopf transformation.
    phi_t = nu*phi_xx is solved by GRW; u = -2*nu*phi_x/phi.

    :param globs: list of dicts 'position' and 'value' = [u_i]
    :param config: SimulationConfig with burgers_mode attribute (cole_hopf_grw)
    :return: updated globs
    """
    mode = (getattr(config, 'burgers_mode', None) or 'cole_hopf_grw').strip().lower()
    diag_dir = getattr(config, '_diag_dir', None)
    if mode not in ('cole_hopf_grw', 'cole_hopf'):
        print(f"  [Burgers] WARNING: unrecognised mode '{mode}'; "
              f"using cole_hopf_grw.")
    return simulate_burgers_cole_hopf_grw(globs, config, _diag_dir=diag_dir)
