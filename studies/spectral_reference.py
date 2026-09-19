"""Deterministic spectral reference solution for the viscous conservation law (Appendix C)."""
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
