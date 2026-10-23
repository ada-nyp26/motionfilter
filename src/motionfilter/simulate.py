import numpy as np
from .motion import motion_matrices


__all__ = ["simulate_motion"]


# simulate_motion with no acceleration remove comment when accel added
def simulate_motion(dt, n_steps, initial_state):
    """Simulate vehicle movement, keeping track of position and velocities,
    Takes initial state and outputs a matrix whcih records each state
    including the initial state.

    Parameters
    ----------
    dt : float
        Time step in seconds.

    n_steps : int
        Number of time steps to simulate.

    initial_state : array_like
        Array of the initial state of vehicle, containing initial x and y
        position and velocities. Shape (4,1)

    Returns
    -------
    vehicle_history: numpy.ndarray
        Shape (n_steps + 1, 4), array containing history of each state at
        each step.
    """
    next_state = initial_state.copy()
    vehicle_history = np.zeros((n_steps + 1, 4), dtype=float)
    vehicle_history[0] = next_state

    # Generate state_transition matrix F for given dt
    F, _ = motion_matrices(dt)

    # generate next step from initial step, take for loop as index
    for step in range(1, n_steps + 1):
        next_state = F @ next_state
        vehicle_history[step] = next_state

    return vehicle_history
