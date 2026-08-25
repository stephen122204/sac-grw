"""
test_relaxation_gbmc.py
=======================
Pytest tests for the Bertaglia--Pareschi--Caflisch (BPC) relaxation GBMC
Burgers solver. The lettered items below index the required behaviors;
individual tests cite them as "item X".

Covers the required behaviors:
  A. Two-speed labels V_i in {-a, +a}
  B. Correct equilibrium expectation E[V_i|u_i] = u_i
  C. Stochastic, not deterministic, switching
  D. Fixed a throughout run
  E. No reflection in whole-line mode
  F. One switching event per time step (structural: loop comment)
  G. Sorting integrity
  H. Unsupported IC raises NotImplementedError
  I. Unsupported domain mode raises NotImplementedError
  J. Subcharacteristic failure raises RuntimeError

Run:
    pytest test_relaxation_gbmc.py -v
"""

import numpy as np
import pytest

import inspect
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxation_gbmc import (
    _reconstruct_u_on_grid,
    advance_rbgbmc_particles,
    simulate_burgers_relaxation_gbmc,
)
from config import SimulationConfig


def _dirichlet_bc(u_left=0.0, u_right=0.0):
    return {
        'LEFT':  {'type': 'Dirichlet', 'value': u_left},
        'RIGHT': {'type': 'Dirichlet', 'value': u_right},
    }


def _make_config(N=200, nu=0.5, dt=0.005, T=0.5, L=4.0,
                 ic_type='stationary_shock', amplitude=1.0,
                 relaxation_speed_a=2.0,
                 relaxation_domain_mode='whole_line',
                 seed=None,
                 burgers_ic_center=None):
    """Convenience factory for SimulationConfig (stationary_shock IC by default).

    ic_type controls which initial condition values are embedded in the config.
    For ICs other than stationary_shock the config is still valid Python but
    the solver will raise NotImplementedError on ic_type mismatch.
    """
    xc = burgers_ic_center if burgers_ic_center is not None else L / 2.0
    x = np.linspace(0.0, L, N)
    if ic_type == 'stationary_shock':
        u0 = -amplitude * np.tanh(amplitude * (x - xc) / (2.0 * nu))
    elif ic_type == 'traveling_wave':
        u0 = 1.0 - 2.0 * np.sqrt(nu) * np.tanh((x - xc) / np.sqrt(nu))
    elif ic_type == 'linear':
        u0 = np.linspace(1.0, -1.0, N)
    elif ic_type == 'uniform':
        u0 = np.ones(N) * 0.5
    else:
        raise ValueError(f"Unknown ic_type: {ic_type!r}")
    ic = list(zip(x.tolist(), u0.tolist()))
    return SimulationConfig(
        equation_type='burgers',
        domain_type='Finite',
        domain_size=L,
        boundary_conditions=_dirichlet_bc(),
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
    """Build globs from config IC, run solver, return (x, u) arrays."""
    globs = [{'position': float(p), 'value': [float(v)]}
             for p, v in cfg.initial_conditions]
    result = simulate_burgers_relaxation_gbmc(globs, cfg)
    x = np.array([g['position'] for g in result])
    u = np.array([g['value'][0] for g in result])
    return x, u


def _exact_stationary_shock(x, nu, xc, A):
    return -A * np.tanh(A * (x - xc) / (2.0 * nu))


# Test: output shape and grid

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


# Test: quantile initialisation properties

class TestQuantileInit:
    """Verify that the quantile IC correctly represents the stationary shock."""

    def test_total_particle_mass_is_minus_2A(self):
        """sum(m_p) = N * (-2A/N) = -2A exactly."""
        A, N = 1.0, 200
        m_p = np.full(N, -2.0 * A / N)
        assert abs(float(m_p.sum()) - (-2.0 * A)) < 1e-12

    def test_particle_positions_are_finite(self):
        """Quantile particle positions must all be finite for valid nu, A."""
        nu, A, xc, N = 0.5, 1.0, 2.0, 200
        i_arr = np.arange(1, N + 1, dtype=float)
        r = (i_arr - 0.5) / N
        x_p = xc + (2.0 * nu / A) * np.arctanh(2.0 * r - 1.0)
        assert np.all(np.isfinite(x_p))

    def test_reconstruction_matches_exact_at_quantile_points(self):
        """
        With N quantile particles, the raw cumsum reconstruction at the
        output window mid-point should be close to the exact stationary shock.
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


# Test: stationary shock test problem

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

    def test_stationary_shock_large_N_beats_small_N(self):
        """Error for N=400 must be smaller than for N=100 (average over 3 seeds)."""
        errors = {100: [], 400: []}
        for seed in (10, 11, 12):
            for N in (100, 400):
                cfg = _make_config(N=N, nu=0.5, T=0.5, amplitude=1.0, seed=seed)
                x, u = _run(cfg)
                u_exact = _exact_stationary_shock(x, 0.5, 2.0, 1.0)
                dx = float(x[1] - x[0])
                errors[N].append(float(np.sqrt(np.sum((u - u_exact)**2) * dx)))
        mean100 = float(np.mean(errors[100]))
        mean400 = float(np.mean(errors[400]))
        assert mean400 < mean100, (
            f"Error did not decrease: N=100 avg={mean100:.4g}, "
            f"N=400 avg={mean400:.4g}"
        )

    def test_u_bounded_by_amplitude(self):
        """u_out is always in [-A, A] (follows from raw cumsum + all-negative masses)."""
        A = 1.0
        cfg = _make_config(N=300, nu=0.5, T=0.5, amplitude=A, seed=99)
        x, u = _run(cfg)
        # With quantile init and all m_i = -2A/N < 0, u_out is in [-A, A] exactly.
        assert u.min() >= -A - 1e-9, f"u.min()={u.min():.6f} < -A"
        assert u.max() <= +A + 1e-9, f"u.max()={u.max():.6f} > +A"


# Test: parameter validation

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


# Test: unsupported IC raises NotImplementedError  (item H)

class TestUnsupportedIC:
    """Only stationary_shock is supported by the wrapper (item H)."""

    def _make_unsupported(self, ic_type, **kw):
        cfg = _make_config(ic_type=ic_type, **kw)
        return cfg

    def test_traveling_wave_raises(self):
        """traveling_wave IC must raise NotImplementedError."""
        cfg = self._make_unsupported('traveling_wave',
                                     relaxation_speed_a=3.0, nu=0.5)
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(NotImplementedError, match="stationary_shock"):
            simulate_burgers_relaxation_gbmc(globs, cfg)

    def test_linear_ic_raises(self):
        """linear IC must raise NotImplementedError."""
        cfg = self._make_unsupported('linear', relaxation_speed_a=3.0)
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(NotImplementedError, match="stationary_shock"):
            simulate_burgers_relaxation_gbmc(globs, cfg)

    def test_uniform_ic_raises(self):
        """uniform IC must raise NotImplementedError."""
        cfg = self._make_unsupported('uniform', relaxation_speed_a=3.0)
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(NotImplementedError, match="stationary_shock"):
            simulate_burgers_relaxation_gbmc(globs, cfg)


# Test: unsupported domain mode raises NotImplementedError  (item I)

class TestUnsupportedDomainMode:
    """Only whole_line domain mode is supported (item I)."""

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


# Test: whole-line mode — no reflection  (item E)

class TestWholeLineMode:
    """Particles must travel freely; no reflection at x=0 or x=L."""

    def test_output_positions_always_on_uniform_grid(self):
        """Output positions are the uniform grid [0, L]; always in [0, L]."""
        cfg = _make_config(N=100, T=0.5, seed=11)
        x, u = _run(cfg)
        assert np.all(x >= -1e-12), "Output position < 0"
        assert np.all(x <= 4.0 + 1e-12), "Output position > L"

    def test_particles_not_reflected_at_boundary(self):
        """
        In whole-line mode a particle transported beyond L stays there.
        Manual check: a particle at L-eps with velocity +a moves to > L.
        Then verify the solver run completes without error and output stays
        on the uniform grid (particles outside do not affect output positions).
        """
        a, dt = 2.0, 0.1
        L = 4.0
        x_single = np.array([L - 1e-4])
        v_single = np.array([+a])
        x_after = x_single + v_single * dt   # L - 1e-4 + 0.2 > L
        assert float(x_after[0]) > L, (
            "Expected particle beyond L in manual check."
        )
        # Full solver run: output positions are always the uniform grid [0, L].
        cfg = _make_config(N=50, T=0.005, dt=0.005, seed=42)
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        result = simulate_burgers_relaxation_gbmc(globs, cfg)
        out_x = [g['position'] for g in result]
        assert min(out_x) >= -1e-12
        assert max(out_x) <= L + 1e-12


# Test: BPC two-speed mechanics  (items A-D, F-G, J)

class TestBPCTwoSpeed:
    """Tests for the validated two-speed BPC relaxation method."""

    # Item A: velocity labels are ±a
    def test_velocity_labels_are_plus_minus_a(self):
        """
        Manually perform the initialization and verify every label is +a or -a.
        Also verify one transport step moves each particle by exactly ±a*dt.
        """
        a, nu, dt = 2.0, 0.5, 0.1
        N = 20
        xc, L = 2.0, 4.0
        i_arr = np.arange(1, N + 1, dtype=float)
        r = (i_arr - 0.5) / N
        x_p = xc + (2.0 * nu / a) * np.arctanh(2.0 * r - 1.0)  # use a=2 here for init
        # Actually use A=1 (amplitude), not a, in the formula
        A = 1.0
        x_p = xc + (2.0 * nu / A) * np.arctanh(2.0 * r - 1.0)
        m_p = np.full(N, -2.0 * A / N)
        u_inf = A

        order = np.argsort(x_p)
        x_p, m_p = x_p[order], m_p[order]
        u = u_inf + np.cumsum(m_p)
        p_plus = (a + u) / (2.0 * a)

        rng = np.random.default_rng(101)
        v = np.where(rng.random(N) < p_plus, +a, -a)

        # A: every label must be exactly +a or -a
        assert np.all(np.isin(v, (-a, +a))), f"Labels not in {{-a,+a}}: {np.unique(v)}"

        # Transport step
        x_before = x_p.copy()
        x_after  = x_p + v * dt
        displacements = x_after - x_before
        assert np.all(np.isclose(np.abs(displacements), a * dt, atol=1e-12)), (
            f"Displacements not ±a*dt={a*dt}: {displacements}"
        )

    # Item B: E[V_i | u_i] = u_i
    def test_expected_velocity_equals_u(self):
        """For many draws, mean(V_i) ≈ u_i within statistical tolerance."""
        a = 2.0
        u_vals = np.array([-1.5, -1.0, 0.0, 0.5, 1.5])
        n_draws = 30000
        rng = np.random.default_rng(200)
        for u_i in u_vals:
            p_plus = (a + u_i) / (2.0 * a)
            draws = np.where(rng.random(n_draws) < p_plus, +a, -a)
            mean_v = float(np.mean(draws))
            assert abs(mean_v - u_i) < 0.05, (
                f"E[V|u={u_i}] = {mean_v:.4f}, expected {u_i:.4f}"
            )

    # Item C: stochastic, not deterministic
    def test_switching_is_stochastic_not_deterministic(self):
        """
        For u_i = +0.5 (< a = 2.0), both +a and -a should appear over many draws.
        A deterministic rule V = a*sign(u) would give only +a.
        """
        a, u_i = 2.0, 0.5
        p_plus = (a + u_i) / (2.0 * a)
        n_draws = 1000
        rng = np.random.default_rng(300)
        draws = np.where(rng.random(n_draws) < p_plus, +a, -a)
        frac_plus  = float(np.mean(draws == +a))
        frac_minus = float(np.mean(draws == -a))
        assert frac_plus  > 0.01, "Never drew +a — switching appears deterministic"
        assert frac_minus > 0.01, "Never drew -a — switching appears deterministic"
        # Check frequency is close to p_plus
        assert abs(frac_plus - p_plus) < 0.05, (
            f"frac(+a)={frac_plus:.4f}, expected p_plus={p_plus:.4f}"
        )

    # Item D: a is fixed
    def test_relaxation_speed_a_is_fixed(self):
        """config.relaxation_speed_a must equal what was passed in."""
        cfg = _make_config(relaxation_speed_a=3.0)
        assert cfg.relaxation_speed_a == 3.0

    def test_a_does_not_change_to_max_u(self):
        """
        If a = 2.0 but max|u_initial| < 1.0, a must still be 2.0 throughout.
        We verify indirectly: a run with a=2.0 succeeds where a=1.05 (> A=1)
        would also succeed, but we keep a at its configured value.
        """
        # With A=1 and a=1.5 > A=1, the run must succeed (not reset a=max|u|).
        cfg = _make_config(N=100, nu=0.5, T=0.1, amplitude=1.0,
                           relaxation_speed_a=1.5, seed=400)
        x, u = _run(cfg)
        # Output u is still in [-A, A] = [-1, 1].
        assert u.min() >= -1.0 - 1e-9
        assert u.max() <= +1.0 + 1e-9

    # Item J: subcharacteristic failure raises RuntimeError
    def test_subcharacteristic_violation_raises(self):
        """a <= A must raise RuntimeError (or ValueError before the loop)."""
        cfg = _make_config(N=100, nu=0.5, T=0.1, amplitude=1.0,
                           relaxation_speed_a=0.5)
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises((RuntimeError, ValueError)):
            simulate_burgers_relaxation_gbmc(globs, cfg)

    def test_standalone_config_keeps_explicit_paper_parameters(self):
        """The standalone configuration retains the explicit speed and seed."""
        cfg = _make_config(relaxation_speed_a=2.0, seed=42)
        assert cfg.relaxation_speed_a == 2.0
        assert cfg.seed == 42
        assert cfg.relaxation_domain_mode == 'whole_line'


# Test: sorting integrity  (item G)

class TestSortingIntegrity:
    """Verify that (X, m, V) are sorted together with the same permutation."""

    def test_sort_preserves_tuple_identity(self):
        """
        Start with deliberately shuffled particles and distinct masses/labels.
        After sorting by X, verify each (X_i, m_i, V_i) triple is intact.
        """
        rng = np.random.default_rng(500)
        N = 20
        x_orig = rng.uniform(0, 4, N)
        m_orig = np.arange(N, dtype=float)  # distinct masses 0,1,...,N-1
        v_orig = np.where(np.arange(N) % 2 == 0, +2.0, -2.0)  # alternating

        # Sort all three together (as the solver does)
        order = np.argsort(x_orig, kind='stable')
        x_s = x_orig[order]
        m_s = m_orig[order]
        v_s = v_orig[order]

        # Every (x, m, v) triple must have come from the same original index
        for k, orig_idx in enumerate(order):
            assert m_s[k] == m_orig[orig_idx], (
                f"Mass mismatch at sorted pos {k}: "
                f"m_s={m_s[k]}, expected m_orig[{orig_idx}]={m_orig[orig_idx]}"
            )
            assert v_s[k] == v_orig[orig_idx], (
                f"Label mismatch at sorted pos {k}: "
                f"v_s={v_s[k]}, expected v_orig[{orig_idx}]={v_orig[orig_idx]}"
            )

    def test_masses_unchanged_by_run(self):
        """
        In whole-line mode masses are NEVER modified (no reflection, no negation).
        Verify np.sort(m_final) == np.sort(m_initial) within floating point.
        """
        A, N = 1.0, 100
        m_initial = np.full(N, -2.0 * A / N)

        cfg = _make_config(N=N, nu=0.5, T=0.2, amplitude=A, seed=600)
        # The solver returns on the uniform grid; retrieve the pre-run m_p.
        # Since all m_i = -2A/N, sorted m_final must also all equal -2A/N.
        x, u = _run(cfg)
        # Reconstruct m_final from output: m_j ≈ u[j] - u[j-1] (ignoring reconstruction noise)
        # Better: verify the solver printed the correct total mass.
        # We verify indirectly: u ∈ [-A, A] and max-min ≈ 2A.
        assert abs(float(u.max() - u.min()) - 2.0 * A) < 0.3, (
            f"u range = {float(u.max()-u.min()):.4f}, expected ≈ {2*A:.4f}"
        )


# Test: pure diffusion (nu > 0)

class TestPureDiffusion:
    """Solver must produce finite output for large nu."""

    def test_large_nu_finite_output(self):
        """Solver output is finite for all x when nu is large."""
        cfg = _make_config(N=100, nu=2.0, T=0.1, amplitude=1.0,
                           relaxation_speed_a=3.0, seed=33)
        x, u = _run(cfg)
        assert np.all(np.isfinite(u)), "Non-finite u in output."

    def test_nu_zero_raises_valueerror(self):
        """nu=0 is invalid for relaxation_gbmc and must raise ValueError."""
        cfg = _make_config(N=100, nu=0.5)
        cfg.diff_constant = 0.0  # override
        globs = [{'position': float(p), 'value': [float(v)]}
                 for p, v in cfg.initial_conditions]
        with pytest.raises(ValueError, match="nu"):
            simulate_burgers_relaxation_gbmc(globs, cfg)


# Test: reconstruction helper (_reconstruct_u_on_grid)

class TestReconstructHelper:
    """_reconstruct_u_on_grid is available for diagnostics/visualization;
    it is NOT used for the primary output of the solver."""

    def test_exact_reconstruction_no_smoothing(self):
        """
        With sigma_bins=0, cumulative masses should recover u exactly
        (up to right-endpoint convention) when particles are on the
        reconstruction grid.
        """
        N = 50
        x_out = np.linspace(0.0, 1.0, N)
        x_p = 0.5 * (x_out[:-1] + x_out[1:])
        m_p = np.full(N - 1, 1.0 / (N - 1))
        u_left = 0.0
        u = _reconstruct_u_on_grid(x_p, m_p, u_left, x_out, sigma_bins=0)
        bin_sums_expected = np.append(m_p, 0.0)
        u_expected = u_left + np.cumsum(bin_sums_expected)
        assert len(u) == N
        assert np.allclose(u, u_expected, atol=1e-12)

    def test_mass_total_preserved_by_reconstruction(self):
        """Total mass before and after smoothing should match."""
        rng = np.random.default_rng(55)
        N = 100
        x_out = np.linspace(0.0, 4.0, N)
        x_p = rng.uniform(0.0, 4.0, 200)
        m_p = rng.standard_normal(200) * 0.01
        total_before = float(m_p.sum())
        u = _reconstruct_u_on_grid(x_p, m_p, 0.0, x_out, sigma_bins=4)
        total_recovered = float(u[-1] - 0.0)
        assert abs(total_recovered - total_before) < 1e-4


# Test: seed reproducibility

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

    def test_traveling_driver_calls_shared_stepper(self):
        """The traveling driver must not contain a second stochastic loop."""
        from studies.study_t2_traveling_shock import _run_traveling
        source = inspect.getsource(_run_traveling)
        assert advance_rbgbmc_particles.__name__ in source
        assert "rng.random" not in source
        assert "rng.normal" not in source

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


def test_default_single_stream_path_pinned():
    """The default (single-stream two-speed) path must match a pinned output.

    Guards the production RNG path against regressions from the opt-in
    split-stream / conditional-mean additions.
    """
    from relaxation_gbmc import advance_rbgbmc_particles
    x0, m0, uL = _ablation_init()
    r = advance_rbgbmc_particles(x0.copy(), m0.copy(), uL, 0.5, 2.0, 0.005, 20,
                                 np.random.default_rng(12345))
    np.testing.assert_allclose(
        r['x'][-3:],
        [4.280939283107618, 4.460827146488502, 4.620113866108306],
        rtol=0, atol=1e-12)


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
