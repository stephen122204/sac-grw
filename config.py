"""Minimal configuration objects used by the standalone RB-GBMC studies."""

import numpy as np


class SimulationConfig:
    """Container for one RB-GBMC run.

    The signature retains the fields used by the original study drivers so the
    pinned Paper 2 configurations remain explicit and reviewable.
    """

    def __init__(
        self,
        equation_type,
        domain_type,
        domain_size,
        boundary_conditions,
        diff_constant,
        time_step,
        total_time,
        num_points,
        initial_conditions,
        reaction_term,
        burgers_mode="relaxation_gbmc",
        burgers_ic_type=None,
        burgers_ic_amplitude=None,
        relaxation_speed_a=None,
        relaxation_domain_mode="whole_line",
        seed=None,
        burgers_ic_center=None,
    ):
        self.equation_type = equation_type
        self.domain_type = domain_type
        self.domain_size = float(domain_size)
        self.boundary_conditions = boundary_conditions
        self.diff_constant = float(diff_constant)
        self.time_step = float(time_step)
        self.total_time = float(total_time)
        self.num_points = int(num_points)
        self.initial_conditions = initial_conditions
        self.reaction_term = reaction_term
        self.burgers_mode = str(burgers_mode).strip().lower()
        self.burgers_ic_type = (burgers_ic_type or "").strip().lower()
        self.burgers_ic_amplitude = (
            float(burgers_ic_amplitude)
            if burgers_ic_amplitude is not None
            else None
        )
        self.relaxation_speed_a = (
            float(relaxation_speed_a)
            if relaxation_speed_a is not None
            else None
        )
        self.relaxation_domain_mode = (
            relaxation_domain_mode or "whole_line"
        ).strip().lower()
        self.seed = int(seed) if seed is not None else None
        self.burgers_ic_center = (
            float(burgers_ic_center) if burgers_ic_center is not None else None
        )


def generate_burgers_stationary_shock_ic(
    domain_size, num_points, nu, x_center=None, amplitude=1.0
):
    """Return the exact stationary tanh profile on a uniform reconstruction grid."""
    if x_center is None:
        x_center = 0.5 * float(domain_size)
    positions = np.linspace(0.0, float(domain_size), int(num_points))
    values = -float(amplitude) * np.tanh(
        float(amplitude) * (positions - float(x_center)) / (2.0 * float(nu))
    )
    return list(zip(positions.tolist(), values.tolist()))
