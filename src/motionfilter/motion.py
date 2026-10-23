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
    numpy.ndarray
        Shape (4,4) identity matrix with indices [0][2] and [1][3] replaced
        by dt.
    """

    F = np.identity(4)

    change_rows = [0, 1]
    change_cols = [2, 3]

    F[change_rows, change_cols] = dt

    return F
