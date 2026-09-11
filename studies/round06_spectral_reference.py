"""Independent deterministic reference for u_t + f(u)_x = nu u_xx.

Written for round 6 because Cole-Hopf does NOT apply to the cubic flux. This
shares no code with the particle solver: it is a Fourier pseudospectral method
with integrating-factor RK4 in time.

DOMAIN AND BOUNDARY CONDITIONS. Periodic on [-L/2, L/2] with L = 32. The
two-pulse initial data are Gaussians of width 0.5 and 0.3 centred at -0.7 and
+0.4, so |u0| < 1e-40 beyond |x| > 8; over T = 1 the fastest characteristic is
max|f'(u)| <= 0.8 (Burgers) or 0.64 (cubic), so nothing reaches the periodic
boundary and the periodic problem agrees with the whole-line problem to far
below the reported errors.

SPACE. Fourier collocation on M points; the flux transform is dealiased with
the 2/3 rule. Convergence in M is verified rather than assumed.

TIME. Integrating-factor RK4: with L_hat = -nu k^2 and N(u) = -i k F[f(u)],

    a = dt N(u)                  b = dt N(E2 (u + a/2))
    c = dt N(E2 u + b/2)         d = dt N(E u + E2 c)
    u <- E u + (E a + 2 E2 (b + c) + d) / 6,     E = e^{L dt}, E2 = e^{L dt/2}.

The diffusion is integrated exactly, so the time error is RK4 on the flux.

OUTPUT. Evaluated at arbitrary points by the exact trigonometric sum of the
computed Fourier coefficients, so no interpolation error is added.

VALIDATION. Run with f(u) = u^2/2 the method must reproduce the stabilised
Cole-Hopf solution of the same two-pulse problem. That check is what licenses
its use as the cubic reference; it is performed in the study driver.
"""
import numpy as np

L_DOMAIN = 32.0
FLUXES = {
    'burgers': lambda u: 0.5 * u * u,
    'cubic': lambda u: u ** 3 / 3.0,
}
FPRIME = {
    'burgers': lambda u: u,
    'cubic': lambda u: u * u,
}


def u0_twopulse(x):
    """Same initial data as studies/twopulse_reference.py."""
    return (0.8 * np.exp(-(x + 0.7) ** 2 / (2 * 0.5 ** 2))
            - 0.5 * np.exp(-(x - 0.4) ** 2 / (2 * 0.3 ** 2)))


def solve(flux, T, nu, M=2048, dt=1e-4, L=L_DOMAIN, u0=u0_twopulse):
    """Return (grid, u(T) on grid, uhat(T), k) for the periodic problem."""
    f = FLUXES[flux]
    xs = -L / 2 + L * np.arange(M) / M
    k = 2.0 * np.pi * np.fft.fftfreq(M, d=L / M)
    dealias = np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))
    Lh = -nu * k ** 2
    n_steps = int(round(T / dt))
    dt = T / n_steps
    E = np.exp(Lh * dt)
    E2 = np.exp(Lh * dt / 2.0)

    def NL(uh):
        u = np.fft.ifft(uh).real
        return -1j * k * np.fft.fft(f(u)) * dealias

    uh = np.fft.fft(u0(xs))
    for _ in range(n_steps):
        a = dt * NL(uh)
        b = dt * NL(E2 * (uh + a / 2.0))
        c = dt * NL(E2 * uh + b / 2.0)
        d = dt * NL(E * uh + E2 * c)
        uh = E * uh + (E * a + 2.0 * E2 * (b + c) + d) / 6.0
    return xs, np.fft.ifft(uh).real, uh, k


def evaluate(uh, k, x, M, L=L_DOMAIN):
    """Exact trigonometric evaluation of the spectral solution at points x."""
    x = np.asarray(x, float)
    shift = np.exp(1j * np.outer(x + L / 2.0, k))
    return (shift @ uh).real / M
