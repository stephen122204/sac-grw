import json
import numpy as np


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
        tau=None
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


# ---------------------------
# Initial condition generators
# ---------------------------

def generate_gaussian_particle_cloud(domain_size, num_points, center=None, sigma=None):
    if center is None:
        center = 0.5 * domain_size
    if sigma is None:
        sigma = 0.1 * domain_size

    pos = np.random.normal(center, sigma, size=num_points)
    pos = np.abs(pos)
    pos = np.where(pos > domain_size, 2 * domain_size - pos, pos)

    vals = np.ones(num_points, dtype=float)
    return list(zip(pos, vals))


def generate_heat_equation_initial_conditions(domain_size, num_points):
    positions = np.linspace(0, domain_size, num_points)
    values = np.linspace(0, 100, num_points)
    return list(zip(positions, values))


def generate_fitzhugh_nagumo_initial_conditions(domain_size, num_points, stimulated_region=None, stimulus_magnitude=2):
    positions = np.linspace(0, domain_size, num_points)
    values = [(0, 0) for _ in positions]

    if stimulated_region:
        start_index = int((stimulated_region[0] / domain_size) * num_points)
        end_index = int((stimulated_region[1] / domain_size) * num_points)
        for i in range(start_index, end_index):
            values[i] = (stimulus_magnitude, 0)

    return list(zip(positions, values))


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


# ---------------------------
# NEW: load config from JSON
# ---------------------------

def load_config_from_json(path: str) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    equation_type = str(data["equation_type"]).strip().lower()
    domain_type = str(data["domain_type"]).strip()
    domain_size = float(data["domain_size"])

    boundary_conditions = data.get("boundary_conditions", {})
    diff_constant = float(data["diff_constant"])
    time_step = float(data["time_step"])
    total_time = float(data["total_time"])
    num_points = int(data["num_points"])
    reaction_term = bool(data.get("reaction_term", False))

    # FHN parameters (optional)
    a = data.get("a", None)
    b = data.get("b", None)
    tau = data.get("tau", None)
    if a is not None:
        a = float(a)
    if b is not None:
        b = float(b)
    if tau is not None:
        tau = float(tau)

    # Build initial conditions depending on equation
    initial_conditions = None

    if equation_type == "heat":
        # default to gaussian particle cloud (matches your current get_user_input behavior)
        heat_ic = data.get("heat_initial_condition", {"type": "gaussian_cloud"})
        ic_type = str(heat_ic.get("type", "gaussian_cloud")).lower()
        if ic_type in {"gaussian", "gaussian_cloud", "cloud"}:
            center = heat_ic.get("center", None)
            sigma = heat_ic.get("sigma", None)
            initial_conditions = generate_gaussian_particle_cloud(
                domain_size,
                num_points,
                center=float(center) if center is not None else None,
                sigma=float(sigma) if sigma is not None else None
            )
        else:
            # fallback to linear ramp
            initial_conditions = generate_heat_equation_initial_conditions(domain_size, num_points)

    elif equation_type == "fitzhugh-nagumo":
        fhn_ic = data.get("fhn_initial_condition", {})
        stim = fhn_ic.get("stimulated_region", None)
        stim_mag = fhn_ic.get("stimulus_magnitude", 2)
        stimulated_region = None
        if stim is not None and isinstance(stim, list) and len(stim) == 2:
            stimulated_region = (float(stim[0]), float(stim[1]))
        initial_conditions = generate_fitzhugh_nagumo_initial_conditions(
            domain_size, num_points,
            stimulated_region=stimulated_region,
            stimulus_magnitude=float(stim_mag)
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
        else:
            raise ValueError(f"Unknown burgers_initial_condition.type: {condition_type!r}")

    else:
        raise ValueError(f"Invalid equation type in JSON: {equation_type!r}")

    return SimulationConfig(
        equation_type, domain_type, domain_size, boundary_conditions,
        diff_constant, time_step, total_time, num_points,
        initial_conditions, reaction_term, a, b, tau
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
        initial_conditions, reaction_term, a, b, tau
    )
