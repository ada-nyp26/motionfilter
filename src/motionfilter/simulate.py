import numpy as np
from .motion import motion_matrices
from .motion import generate_acceleration_noise


__all__ = ["simulate_motion"]


def simulate_motion(dt, n_steps, initial_state, acceleration_std,
                    rng_seed=42, accelerations=None):
    """Simulate vehicle movement, keeping track of position and velocities,
    Takes initial state and outputs a matrix which records each state
    including the initial state.

    Parameters
    ----------
    dt: float
        Time step in seconds.

    n_steps: int
        Number of time steps to simulate.

    initial_state: array_like
        Array of the initial state of vehicle, containing initial x and y
        position and velocities. Shape (4,1)

    rng_seed: int
        Random number generator seed.

    accelerations : array_like
        Acceleration vector, stays constant throughout simulation.
        Changes in acceleration are modelled through added acceleration noise.

    accelerations_std: float
        Standard deviation of acceleration noise.

    Returns
    -------
    vehicle_history: numpy.ndarray
        Shape (n_steps + 1, 4), array containing history of each state at
        each step.
    """
    rng = np.random.default_rng(rng_seed)

    next_state = initial_state.copy()
    vehicle_history = np.zeros((n_steps + 1, 4), dtype=float)
    vehicle_history[0] = next_state

    # Generate state_transition and control input matrix for given dt
    F, B = motion_matrices(dt)

    # generate next step from initial step, take for loop as index
    for step in range(1, n_steps + 1):

        # generate acceleration noise
        acceleration_noise = generate_acceleration_noise(
            acceleration_std, rng
        )

        new_accel = accelerations + acceleration_noise
        acceleration_increase = B @ (new_accel)
        velocity_increase = F @ next_state

        next_state = velocity_increase + acceleration_increase
        vehicle_history[step] = next_state

    return vehicle_history
