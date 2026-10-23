import numpy as np


__all__ = ["motion_matrices"]


def motion_matrices(dt):
    """Generate state transition matrix (F) to set up distance increments via
    timestep and velocity.

    Parameters
    ----------
    dt : float
        Time step.

    Returns
    -------
    F: numpy.ndarray
        Shape (4,4) identity matrix with indices [0][2] and [1][3] replaced
        by dt.

    B: numpy.ndarray
        Shape (4,2) zero matrix with indicies [0][0], [1][1], [2][0], [3][1] to
        be replaced by dt.

    Raises
    ------
    ValueError
        If dt is invalid.
        Non-positive or finite, NaN value or non-scalar (ie a list of values).
    """

    # Checks for dt value and raise ValueError when invalid
    if np.ndim(dt) != 0 or dt <= 0 or not np.isfinite(dt):
        raise ValueError('dt must be a positive integer or finite')

    F = np.identity(4)

    change_rows_F = [0, 1]
    change_cols_F = [2, 3]

    F[change_rows_F, change_cols_F] = dt

    B = np.zeros((4, 2), dtype=float)

    change_rows_B_position = [0, 1]
    change_cols_B_position = [0, 1]

    change_rows_B_velocity = [2, 3]
    change_cols_B_velocity = [0, 1]

    B[change_rows_B_position, change_cols_B_position] = 0.5 * (dt ** 2)
    B[change_rows_B_velocity, change_cols_B_velocity] = dt


    return F, B
