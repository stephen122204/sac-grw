"""Behavioral checks for RB--GBMC initialization, updates and reconstruction.

Archive and study-resume checks live in test_reproducibility_hardening.py.
"""

import numpy as np
import pytest

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxation_gbmc import (
    initialize_tanh_shock_particles,
    reconstruct_cumulative_field,
    advance_rbgbmc_particles,
    simulate_burgers_relaxation_gbmc,
)
from config import SimulationConfig


def _make_config(N=200, nu=0.5, dt=0.005, T=0.5, L=4.0,
                 ic_type='stationary_shock', amplitude=1.0,
                 relaxation_speed_a=2.0,
                 relaxation_domain_mode='whole_line',
                 seed=None,
                 burgers_ic_center=None):
    """Build stationary samples; an unsupported ic_type tests wrapper rejection."""
    xc = burgers_ic_center if burgers_ic_center is not None else L / 2.0
    x = np.linspace(0.0, L, N)
    u0 = -amplitude * np.tanh(amplitude * (x - xc) / (2.0 * nu))
    ic = list(zip(x.tolist(), u0.tolist()))
    return SimulationConfig(
        equation_type='burgers',
        domain_type='Finite',
        domain_size=L,
        boundary_conditions={
            'LEFT': {'type': 'Dirichlet', 'value': 0.0},
            'RIGHT': {'type': 'Dirichlet', 'value': 0.0},
        },
        diff_constant=nu,
        time_step=dt,
        total_time=T,
        num_points=N,
        initial_conditions=ic,
        reaction_term=False,
        burgers_mode='relaxation_gbmc',
        burgers_ic_type=ic_type,
        burgers_ic_amplitude=amplitude,
        burgers_ic_center=xc,
        relaxation_speed_a=relaxation_speed_a,
        relaxation_domain_mode=relaxation_domain_mode,
        seed=seed,
    )


def _run(cfg):
    """Build reconstruction-grid samples, run the solver, return (x, u)."""
    globs = [{'position': float(p), 'value': [float(v)]}
             for p, v in cfg.initial_conditions]
    result = simulate_burgers_relaxation_gbmc(globs, cfg)
    x = np.array([g['position'] for g in result])
    u = np.array([g['value'][0] for g in result])
    return x, u


def _exact_stationary_shock(x, nu, xc, A):
    return -A * np.tanh(A * (x - xc) / (2.0 * nu))


class TestOutputFormat:
    def test_output_length(self):
        """Output must have exactly N globs."""
        cfg = _make_config(N=100, seed=1)
        x, u = _run(cfg)
        assert len(x) == 100
        assert len(u) == 100

    def test_output_grid_uniform(self):
        """Positions must form a uniform grid on [0, L]."""
        cfg = _make_config(N=100, L=4.0, seed=2)
        x, u = _run(cfg)
        dx = np.diff(x)
        assert np.allclose(dx, dx[0], rtol=1e-6), "Reconstruction grid is not uniform."

    def test_output_covers_domain(self):
        """Reconstruction grid must span [0, L]."""
        cfg = _make_config(N=100, L=4.0, seed=3)
        x, u = _run(cfg)
        assert abs(x[0]) < 1e-12, f"x[0] = {x[0]} != 0"
        assert abs(x[-1] - 4.0) < 1e-12, f"x[-1] = {x[-1]} != L"

    def test_value_is_list(self):
        """Each output glob must have value stored as a single-element list."""
        cfg = _make_config(N=50, seed=4)
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        result = simulate_burgers_relaxation_gbmc(globs, cfg)
        for g in result:
            assert isinstance(g['value'], list), "value should be a list"
            assert len(g['value']) == 1, "value should be length-1 list"


class TestQuantileInit:
    """Verify that the quantile IC correctly represents the stationary shock."""

    def test_initial_reconstruction_matches_stationary_profile(self):
        """
        At T=0, cumulative reconstruction on the output grid approximates
        the analytic stationary profile.
        """
        nu, A, L, N = 0.5, 1.0, 4.0, 800
        xc = L / 2.0
        cfg = _make_config(N=N, nu=nu, T=0.0, dt=0.005, L=L,
                           amplitude=A, seed=7)
        # T=0: no time steps, just initialise and output
        x, u = _run(cfg)
        u_exact = _exact_stationary_shock(x, nu, xc, A)
        dx = float(x[1] - x[0])
        rel_l2 = (float(np.sqrt(np.sum((u - u_exact)**2) * dx))
                  / float(np.sqrt(np.sum(u_exact**2) * dx)))
        assert rel_l2 < 0.05, (
            f"IC reconstruction rel L2 = {rel_l2:.4f} >= 0.05 for T=0 run."
        )


class TestStationaryShock:
    """Stationary shock is an exact steady state; RB-GBMC should reproduce it
    to within Monte Carlo noise."""

    def test_stationary_shock_low_noise(self):
        """L2 error vs exact stationary shock < 20% for N=400, T=0.5."""
        cfg = _make_config(N=400, nu=0.5, T=0.5, amplitude=1.0, seed=7)
        x, u = _run(cfg)
        u_exact = _exact_stationary_shock(x, 0.5, 2.0, 1.0)
        dx = float(x[1] - x[0])
        l2_err = float(np.sqrt(np.sum((u - u_exact)**2) * dx))
        l2_ref = float(np.sqrt(np.sum(u_exact**2) * dx))
        rel_l2 = l2_err / l2_ref
        assert rel_l2 < 0.20, f"Relative L2 = {rel_l2:.4f} >= 0.20."

    def test_u_bounded_by_amplitude(self):
        """u_out is always in [-A, A] (follows from raw cumsum + all-negative masses)."""
        A = 1.0
        cfg = _make_config(N=300, nu=0.5, T=0.5, amplitude=A, seed=99)
        x, u = _run(cfg)
        # With quantile init and all m_i = -2A/N < 0, u_out is in [-A, A] exactly.
        assert u.min() >= -A - 1e-9, f"u.min()={u.min():.6f} < -A"
        assert u.max() <= +A + 1e-9, f"u.max()={u.max():.6f} > +A"


class TestParameterValidation:
    """Descriptive errors for invalid parameters."""

    def test_a_none_raises_valueerror(self):
        """Omitting relaxation_speed_a (None) must raise ValueError."""
        cfg = _make_config(relaxation_speed_a=None)
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(ValueError, match="relaxation_speed_a"):
            simulate_burgers_relaxation_gbmc(globs, cfg)

    def test_a_not_greater_than_A_raises(self):
        """a <= A must raise ValueError."""
        # Stationary shock with A=1 requires a > 1.
        cfg = _make_config(amplitude=1.0, relaxation_speed_a=1.0)
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(ValueError, match="strictly greater"):
            simulate_burgers_relaxation_gbmc(globs, cfg)

    def test_nu_zero_raises_valueerror(self):
        """nu=0 must raise ValueError."""
        # Build with nu=0.5 but override the config attribute directly.
        cfg = _make_config(nu=0.5)
        cfg.diff_constant = 0.0
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(ValueError, match="nu"):
            simulate_burgers_relaxation_gbmc(globs, cfg)

    def test_n_less_than_2_raises(self):
        """N < 2 must raise ValueError."""
        cfg = _make_config(N=50)  # valid config
        cfg.num_points = 1
        globs = [{'position': 2.0, 'value': [0.0]}]
        with pytest.raises(ValueError, match="num_points"):
            simulate_burgers_relaxation_gbmc(globs, cfg)

    def test_nonintegral_T_over_dt_raises(self):
        """total_time / time_step non-integral must raise ValueError."""
        cfg = _make_config(T=0.501, dt=0.005)
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(ValueError, match="not an integer"):
            simulate_burgers_relaxation_gbmc(globs, cfg)


class TestUnsupportedIC:
    """Only stationary_shock is supported by the wrapper."""

    def test_traveling_wave_raises(self):
        """traveling_wave IC must raise NotImplementedError."""
        cfg = _make_config(ic_type='traveling_wave',
                           relaxation_speed_a=3.0, nu=0.5)
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(NotImplementedError, match="stationary_shock"):
            simulate_burgers_relaxation_gbmc(globs, cfg)

    def test_linear_ic_raises(self):
        """linear IC must raise NotImplementedError."""
        cfg = _make_config(ic_type='linear', relaxation_speed_a=3.0)
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(NotImplementedError, match="stationary_shock"):
            simulate_burgers_relaxation_gbmc(globs, cfg)

    def test_uniform_ic_raises(self):
        """uniform IC must raise NotImplementedError."""
        cfg = _make_config(ic_type='uniform', relaxation_speed_a=3.0)
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(NotImplementedError, match="stationary_shock"):
            simulate_burgers_relaxation_gbmc(globs, cfg)


class TestUnsupportedDomainMode:
    """Only whole_line domain mode is supported."""

    def test_finite_domain_mode_raises(self):
        """relaxation_domain_mode='finite' must raise NotImplementedError."""
        cfg = _make_config(relaxation_domain_mode='finite')
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(NotImplementedError, match="whole_line"):
            simulate_burgers_relaxation_gbmc(globs, cfg)

    def test_periodic_mode_raises(self):
        """relaxation_domain_mode='periodic' must raise NotImplementedError."""
        cfg = _make_config(relaxation_domain_mode='periodic')
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(NotImplementedError, match="whole_line"):
            simulate_burgers_relaxation_gbmc(globs, cfg)


class TestReproducibility:
    """Verify that seed propagation through SimulationConfig makes runs reproducible."""

    def test_identical_seeds_produce_identical_results(self):
        """Two runs with identical parameters and seed must produce bit-identical output."""
        kwargs = dict(N=50, nu=0.5, T=0.01, dt=0.005, L=4.0,
                      amplitude=1.0, seed=77)
        x1, u1 = _run(_make_config(**kwargs))
        x2, u2 = _run(_make_config(**kwargs))
        np.testing.assert_array_equal(x1, x2, err_msg="Output positions differ for identical seed")
        np.testing.assert_array_equal(u1, u2, err_msg="Output profiles differ for identical seed")

    def test_different_seeds_differ_stochastically(self):
        """Different seeds produce different profiles."""
        _, u1 = _run(_make_config(N=50, nu=0.5, T=0.01, dt=0.005,
                                   L=4.0, amplitude=1.0, seed=77))
        _, u2 = _run(_make_config(N=50, nu=0.5, T=0.01, dt=0.005,
                                   L=4.0, amplitude=1.0, seed=99))
        assert not np.array_equal(u1, u2), (
            "Different seeds produced bit-identical profiles (astronomically unlikely)"
        )

    def test_seed_none_runs_without_error(self):
        """seed=None must still complete with finite output."""
        _, u = _run(_make_config(N=50, nu=0.5, T=0.01, dt=0.005,
                                  L=4.0, amplitude=1.0, seed=None))
        assert np.all(np.isfinite(u))


def test_label_diagnostic_is_non_invasive():
    """The label-variance diagnostic must not perturb the solver's outputs.

    Diagnostics disabled versus enabled must produce identical
    solution arrays, because the diagnostic only reads the already-computed
    reconstruction and consumes no random numbers.
    """
    from relaxation_gbmc import (
        advance_rbgbmc_particles,
        initialize_tanh_shock_particles,
    )
    x0, m0, u_left = initialize_tanh_shock_particles(
        1000, nu=0.5, amplitude=1.0, center=2.0
    )
    a = 2.0

    def run(collect):
        rng = np.random.default_rng(2024)
        return advance_rbgbmc_particles(
            x0.copy(), m0.copy(), u_left, 0.5, a, 0.0025, 100, rng,
            collect_label_diagnostics=collect,
        )

    off = run(False)
    on = run(True)
    for key in ('x', 'm', 'v', 'u_last_sorted'):
        np.testing.assert_array_equal(
            off[key], on[key],
            err_msg=f"diagnostic changed solution array '{key}'",
        )
    # Disabled -> no accumulation (NaN); enabled -> finite mean in (a^2-1, a^2].
    assert np.isnan(off['label_excess_mean'])
    mean_on = on['label_excess_mean']
    assert np.isfinite(mean_on)
    assert (a * a - 1.0) < mean_on <= a * a


def test_label_diagnostic_excludes_unused_final_draw():
    """With n_steps=1 only the initial label state is used for transport.

    The label drawn from the single in-loop reconstruction is never used, so
    the accumulated mean must equal a^2 - <u^2> at the initial configuration.
    """
    from relaxation_gbmc import (
        advance_rbgbmc_particles,
        initialize_tanh_shock_particles,
    )
    x0, m0, u_left = initialize_tanh_shock_particles(
        500, nu=0.5, amplitude=1.0, center=2.0
    )
    a = 2.0
    u_init = u_left + np.cumsum(np.sort(m0))  # sorted-in-place equals m0 order
    expected = float(np.mean(a * a - u_init ** 2))
    rng = np.random.default_rng(5)
    run = advance_rbgbmc_particles(
        x0.copy(), m0.copy(), u_left, 0.5, a, 0.0025, 1, rng,
        collect_label_diagnostics=True,
    )
    assert abs(run['label_excess_mean'] - expected) < 1e-12


class _BrownianSpy:
    """Wrap a Generator, recording every normal() array (Brownian increments)."""
    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.normals = []

    def normal(self, loc, scale, size):
        out = self._rng.normal(loc, scale, size)
        self.normals.append(np.asarray(out).copy())
        return out

    def random(self, *a, **k):
        return self._rng.random(*a, **k)


def _ablation_init():
    from relaxation_gbmc import initialize_tanh_shock_particles
    return initialize_tanh_shock_particles(200, nu=0.5, amplitude=1.0, center=2.0)


def test_split_stream_brownian_generator_is_used():
    """With split streams, changing only the Brownian generator changes output."""
    from relaxation_gbmc import advance_rbgbmc_particles
    x0, m0, uL = _ablation_init()

    def run(bseed):
        return advance_rbgbmc_particles(
            x0.copy(), m0.copy(), uL, 0.5, 2.0, 0.005, 20,
            np.random.default_rng(7), rng_brownian=np.random.default_rng(bseed))['x']
    assert not np.array_equal(run(1), run(2))


def test_identical_brownian_arrays_across_arms():
    """Two-speed and conditional-mean arms with the same Brownian generator
    must consume identical Brownian increment arrays (paired diffusion)."""
    from relaxation_gbmc import advance_rbgbmc_particles
    x0, m0, uL = _ablation_init()
    spy_two = _BrownianSpy(999)
    advance_rbgbmc_particles(x0.copy(), m0.copy(), uL, 0.5, 2.0, 0.005, 20,
                             np.random.default_rng(1), rng_brownian=spy_two)
    spy_ctl = _BrownianSpy(999)
    advance_rbgbmc_particles(x0.copy(), m0.copy(), uL, 0.5, 2.0, 0.005, 20,
                             np.random.default_rng(2), rng_brownian=spy_ctl,
                             conditional_mean_transport=True)
    assert len(spy_two.normals) == len(spy_ctl.normals) == 20
    for b_two, b_ctl in zip(spy_two.normals, spy_ctl.normals):
        np.testing.assert_array_equal(b_two, b_ctl)


def test_control_uses_conditional_mean_and_no_label_uniforms():
    """The control must be invariant to the label generator (V=u^draw, no label
    uniforms consumed), while the two-speed arm must depend on it."""
    from relaxation_gbmc import advance_rbgbmc_particles
    x0, m0, uL = _ablation_init()

    def run(lseed, cm):
        return advance_rbgbmc_particles(
            x0.copy(), m0.copy(), uL, 0.5, 2.0, 0.005, 20,
            np.random.default_rng(lseed),
            rng_brownian=np.random.default_rng(555),
            conditional_mean_transport=cm)['x']
    # control: identical under different label seeds -> no label RNG used
    np.testing.assert_array_equal(run(1, True), run(2, True))
    # two-speed: differs under different label seeds -> label RNG used
    assert not np.array_equal(run(1, False), run(2, False))


@pytest.mark.parametrize("mean_level", [0.0, 0.5])
def test_tanh_initializer_matches_midpoint_quantiles(mean_level):
    """Check the initializer through the analytic CDF and signed jump."""
    x, m, u_left = initialize_tanh_shock_particles(
        200, nu=0.5, amplitude=1.0, center=2.0, mean_level=mean_level)
    assert np.all(np.isfinite(x))
    assert np.all(np.diff(x) > 0)
    cdf = (1.0 + np.tanh(x - 2.0)) / 2.0
    np.testing.assert_allclose(cdf, (np.arange(200) + 0.5) / 200,
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(m, -0.01, rtol=0, atol=1e-15)
    assert abs(m.sum() + 2.0) < 1e-12
    assert u_left == mean_level + 1.0


def test_cumulative_reconstruction_includes_left_particles_and_ties():
    """Use unsorted signed masses, particles outside the window, and X_i=x."""
    u = reconstruct_cumulative_field(
        [2.0, -1.0, 0.5, 0.5], [0.25, -0.5, 0.125, -0.25], 1.0,
        [-2.0, 0.0, 0.5, 1.0, 3.0])
    np.testing.assert_array_equal(u, [1.0, 0.5, 0.375, 0.375, 0.625])


@pytest.mark.parametrize("a, draws, expected", [
    (2.0, [0.60, 0.40, 0.50], [2.0, -2.0, -2.0]),
    (2.0, [0.63, 0.37, 0.49], [-2.0, 2.0, 2.0]),
    (3.0, [0.60, 0.40, 0.50], [-3.0, 3.0, -3.0]),
])
def test_stepper_samples_velocities_at_the_equilibrium_threshold(a, draws, expected):
    """The initial reconstructed states are 1/2, -1/2, 0."""
    class FixedUniforms:
        def random(self, size):
            assert size == 3
            return np.array(draws)

    run = advance_rbgbmc_particles(
        [0.0, 1.0, 2.0], [0.5, -1.0, 0.5], 0.0,
        0.5, a, 0.1, 0, FixedUniforms())
    np.testing.assert_array_equal(run['v'], expected)


def test_one_step_preserves_mass_associations_and_whole_line_motion():
    """A crossing before diffusion must permute positions and masses together."""
    class FixedDraws:
        def __init__(self):
            self.events = []

        def random(self, size):
            assert size == 3
            self.events.append('velocity')
            return np.array([0.2, 0.9, 0.2])

        def normal(self, loc, scale, size):
            assert loc == 0.0 and size == 3
            assert np.isclose(scale, np.sqrt(0.1))
            self.events.append('diffusion')
            return np.array([0.03, -0.04, 0.02])

    draws = FixedDraws()
    run = advance_rbgbmc_particles(
        [4.1, -0.1, 0.05], [0.125, -0.25, 0.125], 0.0,
        0.5, 2.0, 0.1, 1, draws)
    assert draws.events == ['velocity', 'velocity', 'diffusion']
    np.testing.assert_allclose(run['x'], [-0.12, 0.06, 4.32], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(run['m'], [0.125, -0.25, 0.125])
    np.testing.assert_array_equal(run['u_last_sorted'], [0.125, -0.125, 0.0])
    np.testing.assert_array_equal(run['v'], [2.0, -2.0, 2.0])


def test_stepper_rejects_a_subcharacteristic_violation():
    with pytest.raises(RuntimeError, match='Subcharacteristic violation'):
        advance_rbgbmc_particles(
            [0.0, 1.0], [0.5, -0.5], 0.0, 0.5, 0.5, 0.1, 1,
            np.random.default_rng(1))
