import numpy as np

def random_walk(globs, diff_constant, time_step):  #edited sa
    """
    vectorized Brownian step for all globs --> runs much faster than looping in python
    -param globs: list of globs with positions
    -param diff_constant: diffusion constant
    -param time_step: time step
    -return: updated list of globs
    keeps external interface (list of dicts) the same
    """
    import numpy as np

    n = len(globs)
    if n == 0:
        return globs

    # pull positions into one contiguous array
    positions = np.fromiter((g['position'] for g in globs), dtype=float, count=n)

    # random walk step
    # <np.random.normal> draws samples from a normal (Gaussian) distribution
    sigma = np.sqrt(2.0 * diff_constant * time_step)
    positions += np.random.normal(0.0, sigma, size=n)

    # put positions back into globs
    # maintain external interface
    # (list of dicts)
    for i, g in enumerate(globs):
        g['position'] = float(positions[i])

    return globs


def apply_boundary_conditions(globs, boundary_conditions, domain_size):
    """
    Apply boundary conditions to the globs based on the specified boundary conditions.
    
    :param globs: List of globs with positions
    :param boundary_conditions: Dictionary containing the boundary conditions
    :param domain_size: Size of the spatial domain
    
    :return: Updated list of globs"""
    for glob in globs:
        # Check the left boundary condition
        if glob['position'] < 0:
            if boundary_conditions['LEFT']['type'] == 'Dirichlet':
                glob['position'] = 0  # Reflect to boundary
            elif boundary_conditions['LEFT']['type'] == 'Neumann':
                glob['position'] = -glob['position']  # Reflect back

        # Check the right boundary condition
        elif glob['position'] > domain_size:
            if boundary_conditions['RIGHT']['type'] == 'Dirichlet':
                glob['position'] = domain_size
            elif boundary_conditions['RIGHT']['type'] == 'Neumann':
                glob['position'] = 2 * domain_size - glob['position']

    return globs


def simulate_heat_equation(globs, config):
    """
    Simulate the heat equation by performing a random walk on the globs.

    :param globs: List of globs with positions
    :param config: SimulationConfig object with the simulation parameters

    :return: Updated list of globs
    """
    for _ in range(int(config.total_time / config.time_step)):
        globs = random_walk(globs, config.diff_constant, config.time_step)
        globs = apply_boundary_conditions(globs, config.boundary_conditions, config.domain_size)
    return globs


def simulate_fitzhugh_nagumo(globs, config):
    a = config.a
    b = config.b
    tau = config.tau
    dt = config.time_step

    for _ in range(int(config.total_time / dt)):
        for glob in globs:
            u, v = glob['value']
            du_dt = tau * (u - u**3 / 3 + v)
            dv_dt = -1/tau * (u - a + b * v)

            # update u & v
            glob['value'] = [u + du_dt * dt, v + dv_dt * dt]
        
        # Apply random walk to U component
        globs = random_walk_u_component(globs, config.diff_constant, dt)

        # Apply boundary conditions
        globs = apply_boundary_conditions(globs, config.boundary_conditions, config.domain_size)
    
    return globs

def random_walk_u_component(globs, diff_constant, time_step):
    """
    Perform a random walk on the globs based on the diffusion constant and time step.

    :param globs: List of globs with positions
    :param diff_constant: Diffusion constant
    :param time_step: Time step

    :return: Updated list of globs
    """
    for glob in globs:
        u, v = glob['value']
        u += np.random.normal(scale=np.sqrt(2 * diff_constant * time_step))
        glob['value'] = [u, v]
    return globs    


def simulate_burgers(globs, config):
    """
    Simulate the Burgers' equation using a finite difference scheme.
    
    :param globs: List of globs with positions and values
    :param config: SimulationConfig object with the simulation parameters
    
    :return: Updated list of globs"""
    print("Globs Received by simulation: ", globs)
    dt = config.time_step
    dx = config.domain_size / config.num_points  # Ensured float conversion is not necessary here
    nu = config.diff_constant  # Viscosity

    # Convert glob values ensuring they are iterable and contain at least one float
    u = np.array([float(glob['value'][0]) if isinstance(glob['value'], list) and len(glob['value']) > 0 else 0.0 for glob in globs])

    print("Array u: ", u)
    print("Data types in u:", [type(x) for x in u])

    for _ in range(int(config.total_time / dt)):
        u_x = np.gradient(u, dx)
        u_xx = np.gradient(u_x, dx)

        # Update u based on Burger's equation including better handling of non-linear term
        du_dt = -(u * u_x) + nu * u_xx
        u += du_dt * dt

        # Enforce Dirichlet boundary conditions explicitly
        u[0] = config.boundary_conditions['LEFT']['value']
        u[-1] = config.boundary_conditions['RIGHT']['value']

        for i, glob in enumerate(globs):
            glob['value'] = [u[i]]  # Ensure it remains a list

    return globs
