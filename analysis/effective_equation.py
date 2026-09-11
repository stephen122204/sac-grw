"""Effective (modified) equation for the RB-GBMC update, and its predictions.

Derivation (see ANALYTICAL_EXTENSION.md).  At the velocity-sampling stage the
two-speed velocity satisfies  V_i = f'(u_i) + eta_i  with  E[eta_i | u_i] = 0
and  Var(eta_i | u_i) = a^2 - f'(u_i)^2.  The two-speed transport displacement
is therefore EXACTLY the conditional-mean (control) displacement plus an
independent, conditionally mean-zero increment eta_i*dt whose conditional
variance is  2 D(u_i) dt  with

    D(u) = (dt/2) ( a^2 - f'(u)^2 ).

Treating that increment in the Ito diffusion limit and passing to the
mean-field limit gives the FORMAL modified equation

    u_t + f(u)_x = d/dx [ ( nu + D(u) ) u_x ].                       (M)

For Burgers f(u)=u^2/2, f'(u)=u.  Equation (M) is the Jin-Xin / BPC Eq. (9)
relaxation modified equation with eps -> dt/2, plus the physical viscosity.

Dimensionless form.  With  X = A(x-x_c)/nu,  tau = A^2 t/nu,  U = u/A:

    U_tau + U U_X = d/dX [ (1 + delta(U)) U_X ],
    delta(U) = (dtt/2)(at^2 - U^2),   dtt = dt A^2/nu,  at = a/A,

so the whole problem depends only on (at, dtt, taut = T A^2/nu, N).  The
initial stationary shock is U(X,0) = -tanh(X/2).

This module provides
  * steady_profile_modified : the EXACT steady solution of (M) (implicit form),
  * solve_modified          : finite-time solution of (M) from the nu-shock,
  * cole_hopf_constant_nu   : exact solution of constant-coefficient Burgers
                              with viscosity nu_e from the nu-shock, used both
                              as a second prediction and to verify the solver.
All routines work in the dimensionless variables above.
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy.integrate import trapezoid


# --------------------------------------------------------------------------
# Exact steady solution of the modified equation (M)
# --------------------------------------------------------------------------
def steady_profile_modified(X, dtt, at, x0=0.0):
    """Steady solution of (M) in dimensionless variables, evaluated at X.

    Integrating (M) once with far-field states U -> +-1 and U_X -> 0 gives
        (1 + delta(U)) U_X = (U^2 - 1)/2,      delta(U) = (dtt/2)(at^2 - U^2),
    which integrates in closed form to the implicit relation
        X - x0 = 2 nu_star z + dtt * tanh(z),      U = -tanh(z),
    with  nu_star = 1 + (dtt/2)(at^2 - 1)  the dimensionless value of
    1 + delta at the far-field states.  ``X`` is inverted for z by bisection;
    the map z -> X is strictly increasing so the inverse is unique.
    """
    X = np.asarray(X, dtype=float)
    nu_star = 1.0 + 0.5 * dtt * (at ** 2 - 1.0)

    def Xof(z):
        return 2.0 * nu_star * z + dtt * np.tanh(z) + x0

    lo = np.full(X.shape, -1.0)
    hi = np.full(X.shape, 1.0)
    while np.any(Xof(lo) > X):
        lo = np.where(Xof(lo) > X, 2.0 * lo, lo)
    while np.any(Xof(hi) < X):
        hi = np.where(Xof(hi) < X, 2.0 * hi, hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = Xof(mid) - X
        lo = np.where(f < 0.0, mid, lo)
        hi = np.where(f < 0.0, hi, mid)
    return -np.tanh(0.5 * (lo + hi))


# --------------------------------------------------------------------------
# Finite-time solution of the modified equation
# --------------------------------------------------------------------------
def _advect_half(U, dX, dtau):
    """Half step of U_tau + (U^2/2)_X = 0 by second-order Richtmyer two-step
    Lax-Wendroff in conservative form.  The solution stays smooth (viscous),
    so a centred second-order flux is adequate; grid convergence is checked."""
    h = 0.5 * dtau
    F = 0.5 * U * U
    # staggered half-step values at faces j+1/2
    Uh = 0.5 * (U[1:] + U[:-1]) - 0.5 * (h / dX) * (F[1:] - F[:-1])
    Fh = 0.5 * Uh * Uh
    out = U.copy()
    out[1:-1] = U[1:-1] - (h / dX) * (Fh[1:] - Fh[:-1])
    return out


def _diffuse_cn(U, coef_face, dX, dtau):
    """Crank-Nicolson step for U_tau = d/dX [ k(X) U_X ] with the face
    coefficients ``coef_face`` (length n-1, k at faces j+1/2) held fixed.
    Dirichlet ends."""
    n = U.size
    r = dtau / (2.0 * dX * dX)
    kl = coef_face[:-1]      # k_{j-1/2}, j = 1 .. n-2
    kr = coef_face[1:]       # k_{j+1/2}
    ab = np.zeros((3, n))
    # implicit operator (I - r*L)
    ab[1, 0] = 1.0
    ab[1, -1] = 1.0
    ab[0, 2:] = -r * kr            # super-diagonal for rows 1..n-2
    ab[1, 1:-1] = 1.0 + r * (kl + kr)
    ab[2, :-2] = -r * kl           # sub-diagonal for rows 1..n-2
    # explicit right-hand side (I + r*L) U
    rhs = U.copy()
    rhs[1:-1] = (U[1:-1] + r * (kr * (U[2:] - U[1:-1])
                                - kl * (U[1:-1] - U[:-2])))
    return solve_banded((1, 1), ab, rhs)


def solve_modified(taut, dtt, at, Xmax=120.0, dX=0.01, dtau=None,
                   delta_fn=None, U0=None, return_grid=True):
    """Integrate the dimensionless modified equation to time ``taut``.

    ``delta_fn(U)`` returns the dimensionless excess diffusivity; the default
    is the derived  delta(U) = (dtt/2)(at^2 - U^2).  Pass ``delta_fn=lambda U:
    const`` for the constant-coefficient comparison.  ``U0`` defaults to the
    exact stationary nu-shock  -tanh(X/2).
    """
    n = int(round(2.0 * Xmax / dX)) + 1
    X = np.linspace(-Xmax, Xmax, n)
    dX = float(X[1] - X[0])
    if delta_fn is None:
        def delta_fn(u):
            return 0.5 * dtt * (at ** 2 - u ** 2)
    U = (-np.tanh(0.5 * X)) if U0 is None else np.asarray(U0, dtype=float).copy()
    if taut <= 0.0:
        return (X, U) if return_grid else U
    if dtau is None:
        dtau = min(0.25 * dX, taut)          # advective CFL 0.25
    n_steps = max(1, int(np.ceil(taut / dtau)))
    dtau = taut / n_steps
    for _ in range(n_steps):
        U = _advect_half(U, dX, dtau)
        Uf = 0.5 * (U[1:] + U[:-1])
        coef_face = 1.0 + delta_fn(Uf)
        U = _diffuse_cn(U, coef_face, dX, dtau)
        U = _advect_half(U, dX, dtau)
        U[0], U[-1] = 1.0, -1.0
    return (X, U) if return_grid else U


# --------------------------------------------------------------------------
# Exact constant-coefficient reference by Cole-Hopf
# --------------------------------------------------------------------------
def cole_hopf_constant_nu(X, taut, ratio):
    """Exact dimensionless solution of  U_tau + U U_X = ratio * U_XX  started
    from the stationary unit-viscosity shock U(X,0) = -tanh(X/2).

    ``ratio`` = nu_e/nu >= 1 is the enlarged viscosity in units of nu.  With
    the Cole-Hopf transform U = -2*ratio*(ln theta)_X and theta_tau =
    ratio*theta_XX, the initial data give
        theta_0(X) = [cosh(X/2)]^{1/ratio}
    (up to an irrelevant constant).  theta is then the heat evolution of
    theta_0, evaluated here by Gauss-Hermite quadrature.
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    if taut <= 0.0:
        return -np.tanh(0.5 * X)
    s = np.sqrt(2.0 * ratio * taut)           # theta = E[theta_0(X + s*Z)]
    zmax, nz = 14.0, 8001                     # truncated standard-normal grid
    z = np.linspace(-zmax, zmax, nz)
    logw = -0.5 * z * z                       # unnormalised N(0,1) log-weights
    Y = X[:, None] + s * z[None, :]
    p = 1.0 / ratio
    # log theta_0(Y) = p * log cosh(Y/2), written stably
    h = 0.5 * np.abs(Y)
    lg = p * (h + np.log1p(np.exp(-2.0 * h)) - np.log(2.0))
    dlg = 0.5 * p * np.tanh(0.5 * Y)          # d/dY log theta_0
    tot = lg + logw[None, :]
    mx = tot.max(axis=1, keepdims=True)
    e = np.exp(tot - mx)
    theta = trapezoid(e, z, axis=1)
    dtheta = trapezoid(e * dlg, z, axis=1)
    return -2.0 * ratio * (dtheta / theta)


# --------------------------------------------------------------------------
# Physical-variable solver, for initial data other than the stationary shock
# --------------------------------------------------------------------------
def solve_modified_physical(u0_fn, nu, T, x_lo, x_hi, dx=0.001, cfl=0.25,
                            D_fn=None, u_left=None, u_right=None):
    """Integrate  u_t + (u^2/2)_x = d/dx[(nu + D(u)) u_x]  in physical
    variables from ``u0_fn`` to time ``T`` with Dirichlet ends.

    ``D_fn(u)`` is the excess diffusivity; pass ``None`` for plain Burgers with
    viscosity ``nu``.  Same Strang splitting and Crank-Nicolson diffusion as
    ``solve_modified``.
    """
    n = int(round((x_hi - x_lo) / dx)) + 1
    x = np.linspace(x_lo, x_hi, n)
    dx = float(x[1] - x[0])
    u = np.asarray(u0_fn(x), dtype=float)
    bl = u[0] if u_left is None else float(u_left)
    br = u[-1] if u_right is None else float(u_right)
    umax = max(np.max(np.abs(u)), 1e-12)
    dtau = cfl * dx / umax
    n_steps = max(1, int(np.ceil(T / dtau)))
    dtau = T / n_steps
    for _ in range(n_steps):
        u = _advect_half(u, dx, dtau)
        uf = 0.5 * (u[1:] + u[:-1])
        coef = np.full(uf.shape, nu) if D_fn is None else nu + D_fn(uf)
        u = _diffuse_cn(u, coef, dx, dtau)
        u = _advect_half(u, dx, dtau)
        u[0], u[-1] = bl, br
    return x, u
