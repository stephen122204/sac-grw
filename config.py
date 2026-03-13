import json
import numpy as np


# ---------------------------
# Allowed-value registries
# ---------------------------
_ALLOWED_EQUATIONS = {"heat", "burgers", "fitzhugh-nagumo"}

_ALLOWED_DOMAIN_TYPES = {"finite", "semi-infinite", "infinite"}

_ALLOWED_BC_TYPES = {"dirichlet", "neumann"}

_ALLOWED_HEAT_IC_TYPES = {
    "step",
    "uniform_gradient", "ramp", "linear", "linear_ramp",
    "gaussian", "gaussian_cloud", "cloud",
}

_ALLOWED_BURGERS_IC_TYPES = {
    "shock", "shockwave", "shock_wave",
    "step", "stepfunction", "step_function",
    "rarefaction",
    "uniform",
    "tanh", "tanh_profile", "hyperbolic_tangent",
    "traveling_wave",
    "stationary_shock",
}

_ALLOWED_BURGERS_MODES = {
    "cole_hopf_grw",   # main thesis-faithful GRW via Cole-Hopf transformation
    "direct_grw",      # diagnostic: direct gradient-variable GRW (noisy by design)
    "lagrangian_grw",  # experimental Lagrangian particle method (previous approach)
}


def validate_config_dict(data: dict) -> None:
    """
    Validate a raw config dict loaded from JSON before building SimulationConfig.

    Checks:
    - All globally required fields are present.
    - equation_type is a recognised PDE.
    - domain_type is a recognised shape.
    - boundary_conditions are present and well-formed for finite domains.
    - All numeric fields are in valid ranges.
    - Equation-specific required fields are present.
    - Initial-condition types are recognised.

    Raises ValueError with a descriptive message on the first violation found.
    """
    # --- required global fields ---
    required_global = [
        "equation_type", "domain_type", "domain_size",
        "diff_constant", "time_step", "total_time", "num_points",
    ]
    for field in required_global:
        if field not in data:
            raise ValueError(
                f"Config is missing required field: '{field}'. "
                f"See config_template.jsonc for documentation."
            )

    eq = str(data["equation_type"]).strip().lower()
    if eq not in _ALLOWED_EQUATIONS:
        raise ValueError(
            f"'equation_type' must be one of {sorted(_ALLOWED_EQUATIONS)}, got {eq!r}."
        )

    dt = str(data["domain_type"]).strip().lower()
    if dt not in _ALLOWED_DOMAIN_TYPES:
        raise ValueError(
            f"'domain_type' must be one of {sorted(_ALLOWED_DOMAIN_TYPES)}, got {dt!r}."
        )

    # --- boundary_conditions (required for finite domains) ---
    if dt == "finite":
        bc = data.get("boundary_conditions")
        if not isinstance(bc, dict) or "LEFT" not in bc or "RIGHT" not in bc:
            raise ValueError(
                "'boundary_conditions' must be an object with 'LEFT' and 'RIGHT' "
                "entries when domain_type is 'Finite'."
            )
        for side in ("LEFT", "RIGHT"):
            bc_entry = bc[side]
            if not isinstance(bc_entry, dict) or "type" not in bc_entry:
                raise ValueError(
                    f"boundary_conditions.{side} must be an object with a 'type' field."
                )
            bc_type = str(bc_entry["type"]).strip().lower()
            if bc_type not in _ALLOWED_BC_TYPES:
                raise ValueError(
                    f"boundary_conditions.{side}.type must be one of "
                    f"{sorted(_ALLOWED_BC_TYPES)}, got {bc_type!r}."
                )

    # --- numeric ranges ---
    domain_size = float(data["domain_size"])
    if domain_size <= 0:
        raise ValueError(f"'domain_size' must be > 0, got {domain_size}.")

    diff_constant = float(data["diff_constant"])
    if diff_constant < 0:
        raise ValueError(f"'diff_constant' must be >= 0, got {diff_constant}.")
    if eq == "heat" and diff_constant <= 0:
        raise ValueError(
            f"'diff_constant' (thermal diffusivity) must be > 0 for the heat equation, "
            f"got {diff_constant}."
        )

    time_step = float(data["time_step"])
    if time_step <= 0:
        raise ValueError(f"'time_step' must be > 0, got {time_step}.")

    total_time = float(data["total_time"])
    if total_time <= 0:
        raise ValueError(f"'total_time' must be > 0, got {total_time}.")

    num_points = int(data["num_points"])
    if num_points < 1:
        raise ValueError(f"'num_points' must be >= 1, got {num_points}.")

    # --- equation-specific validation ---
    if eq == "fitzhugh-nagumo":
        for param in ("a", "b", "tau"):
            if data.get(param) is None:
                raise ValueError(
                    f"FitzHugh–Nagumo requires '{param}' to be set in the config. "
                    f"See config_template.jsonc for documentation."
                )

    if eq == "heat":
        heat_ic = data.get("heat_initial_condition", {})
        ic_type = str(heat_ic.get("type", "step")).strip().lower()
        if ic_type not in _ALLOWED_HEAT_IC_TYPES:
            raise ValueError(
                f"heat_initial_condition.type must be one of "
                f"{sorted(_ALLOWED_HEAT_IC_TYPES)}, got {ic_type!r}."
            )

    if eq == "burgers":
        b_ic = data.get("burgers_initial_condition", {})
        ic_type = str(b_ic.get("type", "shock")).strip().lower()
        if ic_type not in _ALLOWED_BURGERS_IC_TYPES:
            raise ValueError(
                f"burgers_initial_condition.type must be one of "
                f"{sorted(_ALLOWED_BURGERS_IC_TYPES)}, got {ic_type!r}."
            )


class SimulationConfig:
    """
    A class to hold the configuration for the simulation.
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
        a=None,
        b=None,
        tau=None,
        burgers_mode=None,
        burgers_ic_type=None,
        burgers_ic_amplitude=None,
        fhn_ic_type=None,
    ):
        self.equation_type = equation_type
        self.domain_type = domain_type
        self.domain_size = domain_size
        self.boundary_conditions = boundary_conditions
        self.diff_constant = diff_constant
        self.time_step = time_step
        self.total_time = total_time
        self.num_points = num_points
        self.initial_conditions = initial_conditions
        self.reaction_term = reaction_term
        self.a = a
        self.b = b
        self.tau = tau
        # Burgers solver mode: "cole_hopf_grw" | "direct_grw" | "lagrangian_grw"
        self.burgers_mode = (burgers_mode or "cole_hopf_grw").strip().lower()
        # Burgers IC type string, set by load_config_from_json for verify_solver.
        self.burgers_ic_type = (burgers_ic_type or "").strip().lower()
        # For stationary_shock IC: the amplitude parameter A in -A*tanh(A*(x-xc)/(2*nu)).
        self.burgers_ic_amplitude = float(burgers_ic_amplitude) if burgers_ic_amplitude is not None else None
        # FHN IC type: "steady_solution" | "nonsmooth" | "discontinuous" | "" (legacy)
        self.fhn_ic_type = (fhn_ic_type or "").strip().lower()


# ---------------------------
# Initial condition generators
# ---------------------------

def generate_gaussian_particle_cloud(domain_size, num_points, center=None, sigma=None,
                                      jump_height=1.0):
    """
    GRW globs drawn from a Gaussian distribution (exploratory IC).

    Each glob receives value = jump_height / num_points so the total signed weight equals
    jump_height, matching the convention used by the step and uniform-gradient generators.
    This ensures the Dirichlet reconstruction u = uL + cumsum(values) returns values in the
    correct physical range regardless of glob count.
    """
    if center is None:
        center = 0.5 * domain_size
    if sigma is None:
        sigma = 0.1 * domain_size

    pos = np.random.normal(center, sigma, size=num_points)
    pos = np.abs(pos)
    pos = np.where(pos > domain_size, 2 * domain_size - pos, pos)

    weight = jump_height / num_points
    vals = np.full(num_points, weight, dtype=float)
    return list(zip(pos, vals))


def generate_heat_step_initial_conditions(domain_size, num_points, jump_position=None, jump_height=1.0):
    """
    GRW initial conditions for a step (Heaviside) initial temperature profile.

    Differentiating a step function produces a single delta-function in u_x.  The GRW
    method represents that delta-jump as N globs all placed at the jump location, each
    carrying a signed value of jump_height / N so their cumulative sum reconstructs the
    original step.

    :param domain_size: float, right endpoint of the domain
    :param num_points: int, number of globs (N)
    :param jump_position: float, location of the step; defaults to domain midpoint
    :param jump_height: float, magnitude of the jump (u_right - u_left)
    :return: list of (position, value) pairs
    """
    if jump_position is None:
        jump_position = 0.5 * domain_size
    weight = jump_height / num_points
    return [(float(jump_position), float(weight))] * num_points


def generate_heat_uniform_gradient_initial_conditions(domain_size, num_points, gradient_value=1.0):
    """
    GRW initial conditions for a linear initial temperature profile u(x, 0) = gradient_value * x.

    A constant gradient u_x = gradient_value is uniform across [0, L].  The GRW method
    represents this as N globs distributed uniformly over [0, L], each with value
    gradient_value * domain_size / N so their cumulative sum reconstructs the ramp.

    :param domain_size: float, right endpoint of the domain
    :param num_points: int, number of globs
    :param gradient_value: float, constant slope du/dx of the initial profile
    :return: list of (position, value) pairs
    """
    positions = np.linspace(0.0, domain_size, num_points)
    weight = gradient_value * domain_size / num_points
    return [(float(p), float(weight)) for p in positions]


def generate_fitzhugh_nagumo_initial_conditions(
    domain_size, num_points,
    stimulated_region=None,
    stimulus_u=2.0, stimulus_v=None,
    rest_u=0.0, rest_v=0.0,
    # legacy positional alias
    stimulus_magnitude=None,
):
    """
    Build FHN initial conditions on a uniform spatial grid.

    All points outside the stimulated region are set to (rest_u, rest_v).
    Points inside the stimulated region are set to (stimulus_u, stimulus_v).
    stimulus_v defaults to rest_v when not given, which keeps v spatially
    uniform at t=0 and avoids an artificial v-gradient at the stimulus edge.

    For a proper excitable-regime traveling pulse, rest_u and rest_v should be
    the stable fixed point of the ODE, and stimulus_u should be on the opposite
    side of the excitation threshold from rest_u.

    The legacy 'stimulus_magnitude' argument (formerly the u value of the
    stimulus) is accepted for backward compatibility; it takes precedence over
    'stimulus_u' when present.
    """
    if stimulus_magnitude is not None:
        stimulus_u = float(stimulus_magnitude)
    if stimulus_v is None:
        stimulus_v = rest_v

    positions = np.linspace(0, domain_size, num_points)
    values = [(float(rest_u), float(rest_v)) for _ in positions]

    if stimulated_region:
        start_index = int((stimulated_region[0] / domain_size) * num_points)
        end_index = int((stimulated_region[1] / domain_size) * num_points)
        for i in range(start_index, end_index):
            values[i] = (float(stimulus_u), float(stimulus_v))

    return list(zip(positions, values))


def generate_fhn_steady_ic(num_globs, a, x_center=0.0):
    """
    Steady-solution IC for the thesis scalar FHN GRW.

    Globs are initialized so that their cumulative sum reproduces the exact
    traveling-wave profile  u(x, 0) = 1 / (1 + exp(-(x - x_center) / 2))
    at t = 0.

    Initialization strategy (from the thesis):
      - Divide the solution range [0, 1] into N0 equal segments.
        u_i = (i + 0.5) / N0  for  i = 0, ..., N0-1.
      - Invert the logistic:  x_i = x_center - 2 * log(1 / u_i - 1).
      - Assign uniform weights w_i = 1 / N0.

    This places all globs in sorted order with equal weights; the cumulative
    sum reconstructs u from 0 at the left to 1 at the right.

    :param num_globs: int, number of globs N0
    :param a:         float, FHN threshold parameter (used only for annotation)
    :param x_center:  float, position of the wave center (u = 0.5) at t = 0
    :return: list of (position, weight) pairs
    """
    N0 = int(num_globs)
    u_i = (np.arange(N0) + 0.5) / float(N0)
    x_i = x_center - 2.0 * np.log(1.0 / u_i - 1.0)
    w_i = np.full(N0, 1.0 / N0)
    return list(zip(x_i.tolist(), w_i.tolist()))


def generate_fhn_nonsmooth_ic(num_globs, a, x_center=0.0, half_width=3.0):
    """
    Non-smooth IC for the thesis scalar FHN GRW.

    Globs are initialized to reproduce a piecewise-linear ramp profile:
      u_0(x) = 0                          for  x < x_center - half_width
      u_0(x) = (x - (x_center - hw)) / (2*hw)  for  |x - x_center| <= hw
      u_0(x) = 1                          for  x > x_center + half_width

    where hw = half_width.  This is C0 but not C1: the gradient u_x has
    jump discontinuities at the two kink points.  The profile is expected
    to relax toward the smooth logistic traveling wave over time.

    Glob positions are placed via the linear inverse:
      x_i = (x_center - hw) + 2 * hw * u_i

    :param num_globs:   int, number of globs
    :param a:           float, FHN threshold parameter (informational)
    :param x_center:    float, center of the ramp
    :param half_width:  float, half-width of the linear transition zone
    :return: list of (position, weight) pairs
    """
    N0 = int(num_globs)
    hw = float(half_width)
    u_i = (np.arange(N0) + 0.5) / float(N0)
    x_i = (x_center - hw) + 2.0 * hw * u_i
    w_i = np.full(N0, 1.0 / N0)
    return list(zip(x_i.tolist(), w_i.tolist()))


def generate_fhn_discontinuous_ic(num_globs, a, x_center=0.0):
    """
    Discontinuous (Heaviside) IC for the thesis scalar FHN GRW.

    All N0 globs are placed at x = x_center with equal weights w_i = 1/N0.
    This represents the Dirac delta u_x = delta(x - x_center), corresponding
    to the step function u_0(x) = 0 for x < x_center, 1 for x >= x_center.

    Over time the globs diffuse and develop the traveling-wave profile,
    reproducing the thesis observation that even this extreme IC relaxes to
    the exact traveling wave solution.

    :param num_globs: int, number of globs
    :param a:         float, FHN threshold parameter (informational)
    :param x_center:  float, initial position of all globs
    :return: list of (position, weight) pairs
    """
    N0 = int(num_globs)
    w_i = 1.0 / N0
    return [(float(x_center), w_i)] * N0


def generate_burgers_initial_conditions(domain_size, num_points, condition_type='shock',
                                       shock_position=None, left_value=1, right_value=0):
    positions = np.linspace(0, domain_size, num_points)
    values = np.zeros(num_points)

    if condition_type in ['shock', 'step']:
        shock_index = int(shock_position / domain_size * num_points) if shock_position is not None else num_points // 2
        values[:shock_index] = left_value
        values[shock_index:] = right_value
    elif condition_type == 'rarefaction':
        values = np.linspace(left_value, right_value, num_points)
    elif condition_type == 'uniform':
        values.fill(left_value)
    elif condition_type == 'tanh':
        x_center = domain_size / 2
        scale = 0.1 * domain_size
        values = 0.5 * (1 - np.tanh((positions - x_center) / scale))

    return list(zip(positions, values))


def generate_burgers_stationary_shock_ic(domain_size, num_points, nu, x_center=None,
                                         amplitude=1.0):
    """
    Stationary-shock initial condition for Burgers' equation.

    u0(x) = -A * tanh(A * (x - x_center) / (2 * nu))

    where A = amplitude.  This is an EXACT stationary solution to Burgers':
      u_t + u*u_x = nu*u_xx  =>  u_t = 0  (for all t)

    The Cole-Hopf transform phi0 = exp(-Psi0 / (2*nu)) satisfies phi0(0) = phi0(L)
    (since Psi0(L) = integral of u0 over [0,L] = 0 by antisymmetry around x_center).
    This makes the reconstruction cumsum always return to its starting value, giving
    a well-conditioned benchmark for the Cole-Hopf GRW with finite-domain boundaries.

    :param domain_size: float, right endpoint [0, L]
    :param num_points:  int, number of grid points / globs
    :param nu:          float, kinematic viscosity (must match diff_constant in config)
    :param x_center:    float, shock center; defaults to domain midpoint
    :param amplitude:   float, shock amplitude A (sets both wave height and width)
    :return: list of (position, value) pairs
    """
    if x_center is None:
        x_center = 0.5 * domain_size
    positions = np.linspace(0.0, domain_size, num_points)
    values = -amplitude * np.tanh(amplitude * (positions - x_center) / (2.0 * nu))
    return list(zip(positions.tolist(), values.tolist()))


def generate_burgers_traveling_wave_ic(domain_size, num_points, nu, x_center=None):
    """
    Traveling wave initial condition for Burgers' equation.

    u0(x) = 1 - 2*sqrt(nu) * tanh((x - x_center) / sqrt(nu))

    This is the t=0 snapshot of the exact traveling wave solution:
      u(x, t) = 1 - 2*sqrt(nu) * tanh((x - x_center - t) / sqrt(nu))
    which moves at unit speed and satisfies u_t + u*u_x = nu*u_xx exactly.

    Derivation: inserting the ansatz f(x - ct) into Burgers yields c = 1 and
    the tanh width delta = sqrt(nu).  This IC is the canonical benchmark used
    in the thesis to test the Cole-Hopf GRW path.

    :param domain_size: float, right endpoint of the domain [0, L]
    :param num_points: int, number of grid points / globs
    :param nu: float, kinematic viscosity (should match diff_constant in config)
    :param x_center: float, initial wave center; defaults to domain midpoint
    :return: list of (position, value) pairs
    """
    if x_center is None:
        x_center = 0.5 * domain_size
    positions = np.linspace(0.0, domain_size, num_points)
    values = 1.0 - 2.0 * np.sqrt(nu) * np.tanh((positions - x_center) / np.sqrt(nu))
    return list(zip(positions.tolist(), values.tolist()))


# ---------------------------
# Load config from JSON
# ---------------------------

def load_config_from_json(path: str) -> SimulationConfig:
    """
    Load and validate a SimulationConfig from a JSON file.

    Raises ValueError with a descriptive message if the config fails validation.
    See config_template.jsonc for documentation of every supported field.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    validate_config_dict(data)

    equation_type = str(data["equation_type"]).strip().lower()
    domain_type = str(data["domain_type"]).strip()
    domain_size = float(data["domain_size"])

    boundary_conditions = data.get("boundary_conditions", {})
    diff_constant = float(data["diff_constant"])
    time_step = float(data["time_step"])
    total_time = float(data["total_time"])
    num_points = int(data["num_points"])
    reaction_term = bool(data.get("reaction_term", False))

    a = data.get("a", None)
    b = data.get("b", None)
    tau = data.get("tau", None)
    if a is not None:
        a = float(a)
    if b is not None:
        b = float(b)
    if tau is not None:
        tau = float(tau)

    initial_conditions = None

    if equation_type == "heat":
        heat_ic = data.get("heat_initial_condition", {"type": "step"})
        ic_type = str(heat_ic.get("type", "step")).lower()

        if ic_type == "step":
            jump_position = heat_ic.get("jump_position", None)
            jump_height = float(heat_ic.get("jump_height", 1.0))
            initial_conditions = generate_heat_step_initial_conditions(
                domain_size,
                num_points,
                jump_position=float(jump_position) if jump_position is not None else None,
                jump_height=jump_height,
            )
        elif ic_type in {"uniform_gradient", "ramp", "linear", "linear_ramp"}:
            gradient_value = float(heat_ic.get("gradient_value", 1.0))
            initial_conditions = generate_heat_uniform_gradient_initial_conditions(
                domain_size, num_points, gradient_value=gradient_value
            )
        elif ic_type in {"gaussian", "gaussian_cloud", "cloud"}:
            center = heat_ic.get("center", None)
            sigma = heat_ic.get("sigma", None)
            jump_height = float(heat_ic.get("jump_height", 1.0))
            initial_conditions = generate_gaussian_particle_cloud(
                domain_size,
                num_points,
                center=float(center) if center is not None else None,
                sigma=float(sigma) if sigma is not None else None,
                jump_height=jump_height,
            )
        else:
            raise ValueError(f"Unknown heat_initial_condition.type: {ic_type!r}")

    elif equation_type == "fitzhugh-nagumo":
        fhn_ic = data.get("fhn_initial_condition", {})
        ic_type_raw = str(fhn_ic.get("type", "legacy")).strip().lower()

        if ic_type_raw in ("steady_solution", "steady"):
            fhn_ic_type_stored = "steady_solution"
            ic_a = float(fhn_ic.get("a", data.get("a", 0.25)))
            ic_x_center = float(fhn_ic.get("x_center", domain_size / 2.0))
            initial_conditions = generate_fhn_steady_ic(
                num_points, ic_a, x_center=ic_x_center)

        elif ic_type_raw in ("nonsmooth", "non_smooth", "nonsmooth_initial_data"):
            fhn_ic_type_stored = "nonsmooth"
            ic_a = float(fhn_ic.get("a", data.get("a", 0.25)))
            ic_x_center = float(fhn_ic.get("x_center", domain_size / 2.0))
            ic_hw = float(fhn_ic.get("half_width", 3.0))
            initial_conditions = generate_fhn_nonsmooth_ic(
                num_points, ic_a, x_center=ic_x_center, half_width=ic_hw)

        elif ic_type_raw in ("discontinuous", "heaviside", "discontinuous_initial_data"):
            fhn_ic_type_stored = "discontinuous"
            ic_a = float(fhn_ic.get("a", data.get("a", 0.25)))
            ic_x_center = float(fhn_ic.get("x_center", domain_size / 2.0))
            initial_conditions = generate_fhn_discontinuous_ic(
                num_points, ic_a, x_center=ic_x_center)

        else:
            # Legacy two-component stimulus/rest IC.
            fhn_ic_type_stored = "legacy"
            stim = fhn_ic.get("stimulated_region", None)
            stimulated_region = None
            if stim is not None and isinstance(stim, list) and len(stim) == 2:
                stimulated_region = (float(stim[0]), float(stim[1]))
            stim_u = fhn_ic.get("stimulus_u",
                     fhn_ic.get("stimulus_magnitude", 2.0))
            stim_v = fhn_ic.get("stimulus_v", None)
            rest_u = fhn_ic.get("rest_u", 0.0)
            rest_v = fhn_ic.get("rest_v", 0.0)
            initial_conditions = generate_fitzhugh_nagumo_initial_conditions(
                domain_size, num_points,
                stimulated_region=stimulated_region,
                stimulus_u=float(stim_u),
                stimulus_v=float(stim_v) if stim_v is not None else None,
                rest_u=float(rest_u),
                rest_v=float(rest_v),
            )

    elif equation_type == "burgers":
        b_ic = data.get("burgers_initial_condition", {"type": "shock"})
        condition_type = str(b_ic.get("type", "shock")).lower()

        # normalize aliases
        if condition_type in {"shockwave", "shock_wave"}:
            condition_type = "shock"
        if condition_type in {"stepfunction", "step_function"}:
            condition_type = "step"
        if condition_type in {"tanh_profile", "hyperbolic_tangent"}:
            condition_type = "tanh"

        shock_position = b_ic.get("shock_position", None)
        left_value = float(b_ic.get("left_value", 1))
        right_value = float(b_ic.get("right_value", 0))

        if condition_type in {"shock", "step"}:
            if shock_position is None:
                shock_position = domain_size / 2
            initial_conditions = generate_burgers_initial_conditions(
                domain_size, num_points,
                condition_type=condition_type,
                shock_position=float(shock_position),
                left_value=left_value,
                right_value=right_value
            )
        elif condition_type == "rarefaction":
            initial_conditions = generate_burgers_initial_conditions(
                domain_size, num_points,
                condition_type="rarefaction",
                left_value=left_value,
                right_value=right_value
            )
        elif condition_type == "uniform":
            initial_conditions = generate_burgers_initial_conditions(
                domain_size, num_points,
                condition_type="uniform",
                left_value=left_value
            )
        elif condition_type == "tanh":
            initial_conditions = generate_burgers_initial_conditions(
                domain_size, num_points,
                condition_type="tanh"
            )
        elif condition_type == "traveling_wave":
            ic_nu = float(b_ic.get("nu", diff_constant))
            x_center = b_ic.get("x_center", None)
            initial_conditions = generate_burgers_traveling_wave_ic(
                domain_size, num_points,
                nu=ic_nu,
                x_center=float(x_center) if x_center is not None else None,
            )
        elif condition_type == "stationary_shock":
            ic_nu = float(b_ic.get("nu", diff_constant))
            x_center = b_ic.get("x_center", None)
            burgers_shock_amplitude = float(b_ic.get("amplitude", 1.0))
            initial_conditions = generate_burgers_stationary_shock_ic(
                domain_size, num_points,
                nu=ic_nu,
                x_center=float(x_center) if x_center is not None else None,
                amplitude=burgers_shock_amplitude,
            )
        else:
            raise ValueError(f"Unknown burgers_initial_condition.type: {condition_type!r}")
        burgers_ic_type_stored = condition_type

    else:
        raise ValueError(f"Invalid equation type in JSON: {equation_type!r}")

    burgers_mode = str(data.get("burgers_mode", "cole_hopf_grw")).strip().lower()
    # burgers_ic_type_stored and burgers_shock_amplitude are set only for burgers ICs.
    b_ic_type_out = locals().get("burgers_ic_type_stored", "")
    b_ic_amplitude_out = locals().get("burgers_shock_amplitude", None)

    return SimulationConfig(
        equation_type, domain_type, domain_size, boundary_conditions,
        diff_constant, time_step, total_time, num_points,
        initial_conditions, reaction_term, a, b, tau,
        burgers_mode=burgers_mode,
        burgers_ic_type=b_ic_type_out,
        burgers_ic_amplitude=b_ic_amplitude_out,
        fhn_ic_type=locals().get('fhn_ic_type_stored', ''),
    )


# ---------------------------
# Existing interactive input
# ---------------------------

def get_user_input():
    print("Welcome to the Gradient Random Walk Simulation Setup")
    equation_type = input("Choose the equation to simulate (Heat, Fitzhugh-Nagumo, Burgers): ").lower().strip()
    domain_type = input("Enter the domain type (Finite, Semi-Infinite, Infinite): ")
    domain_size = float(input("Enter the domain size: "))

    boundary_conditions = {}
    if domain_type != "Infinite":
        for end in ["LEFT", "RIGHT"]:
            bc_type = input(f"Enter {end} boundary condition type (Dirichlet, Neumann): ")
            bc_value = float(input(f"Enter {end} the boundary value: "))
            boundary_conditions[end] = {'type': bc_type, 'value': bc_value}

    diff_constant = float(input("Enter the diffusivity constant: "))
    time_step = float(input("Enter the time step (delta_t): "))
    total_time = float(input("Enter the total simulation time: "))
    num_points = int(input("Enter the number of globs: "))

    initial_conditions, a, b, tau = None, None, None, None
    stimulated_region, stimulus_magnitude = None, None

    if equation_type == "fitzhugh-nagumo":
        a = float(input("Enter FitzHugh-Nagumo parameter a: "))
        b = float(input("Enter FitzHugh-Nagumo parameter b: "))
        tau = float(input("Enter FitzHugh-Nagumo temporal scaling parameter tau: "))
        if input("Apply an initial electrical stimulus? (yes/no): ").lower() == 'yes':
            start = float(input("Enter the start position of the stimulated region (e.g., 45): "))
            end = float(input("Enter the end position of the stimulated region (e.g., 55): "))
            stimulus_magnitude = float(input("Enter the magnitude of the stimulus: "))
            stimulated_region = (start, end)
        initial_conditions = generate_fitzhugh_nagumo_initial_conditions(
            domain_size, num_points, stimulated_region, stimulus_magnitude
        )

    elif equation_type == "heat":
        print("Select the heat initial condition type:")
        print("  1. Step / Heaviside  — N globs at a jump location (thesis canonical case)")
        print("  2. Uniform gradient  — N globs spread across domain with constant u_x")
        print("  3. Gaussian cloud    — globs drawn from a Gaussian (exploratory)")
        ic_choice = input("Enter your choice (1-3, default 1): ").strip() or "1"

        if ic_choice == "1":
            jump_position = input(f"Jump location (default {domain_size / 2:.4g}): ").strip()
            jump_position = float(jump_position) if jump_position else domain_size / 2
            jump_height = input("Jump height u_R - u_L (default 1.0): ").strip()
            jump_height = float(jump_height) if jump_height else 1.0
            initial_conditions = generate_heat_step_initial_conditions(
                domain_size, num_points, jump_position=jump_position, jump_height=jump_height
            )
        elif ic_choice == "2":
            gradient_value = input("Gradient value du/dx (default 1.0): ").strip()
            gradient_value = float(gradient_value) if gradient_value else 1.0
            initial_conditions = generate_heat_uniform_gradient_initial_conditions(
                domain_size, num_points, gradient_value=gradient_value
            )
        else:
            initial_conditions = generate_gaussian_particle_cloud(domain_size, num_points)

    elif equation_type == "burgers":
        print("Select the type of initial condition:")
        print("1. Shock Wave")
        print("2. Step Function")
        print("3. Rarefaction Wave")
        print("4. Uniform Condition")
        print("5. Hyperbolic Tangent Profile")
        condition_type = input("Enter your choice (1-5): ")

        condition_map = {'1': 'shock', '2': 'step', '3': 'rarefaction', '4': 'uniform', '5': 'tanh'}
        condition_type = condition_map.get(condition_type, 'shock')

        if condition_type in ['shock', 'step']:
            shock_position = float(input("Enter the position for the transition (e.g., 5): "))
            left_value = float(input("Enter the value before the transition: "))
            right_value = float(input("Enter the value after the transition: "))
            initial_conditions = generate_burgers_initial_conditions(
                domain_size, num_points,
                condition_type=condition_type,
                shock_position=shock_position,
                left_value=left_value,
                right_value=right_value
            )
        elif condition_type == 'rarefaction':
            left_value = float(input("Enter the starting value: "))
            right_value = float(input("Enter the ending value: "))
            initial_conditions = generate_burgers_initial_conditions(
                domain_size, num_points, condition_type='rarefaction',
                left_value=left_value, right_value=right_value
            )
        elif condition_type == 'uniform':
            uniform_value = float(input("Enter the uniform value across the domain: "))
            initial_conditions = generate_burgers_initial_conditions(
                domain_size, num_points, condition_type='uniform', left_value=uniform_value
            )
        elif condition_type == 'tanh':
            initial_conditions = generate_burgers_initial_conditions(domain_size, num_points, condition_type='tanh')

    else:
        raise ValueError("Invalid equation type")

    reaction_term = input("Is there a reaction term? (yes/no): ").lower() == 'yes'

    return SimulationConfig(
        equation_type, domain_type, domain_size, boundary_conditions,
        diff_constant, time_step, total_time, num_points,
        initial_conditions, reaction_term, a, b, tau,
        burgers_mode="cole_hopf_grw",
    )
