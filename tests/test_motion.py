import numpy as np
import pytest

from motionfilter import motion_matrices


class TestMotionMatrices:
    """Tests for motion_matrices() in motion.py."""

    @pytest.mark.parametrize("dt, expected_F, expected_B",
                             [
                                 (
                                     # dt 1
                                     2.0,
                                     # expected_F matrix 1
                                     np.array([
                                         [1, 0, 2, 0],
                                         [0, 1, 0, 2],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]
                                     ]),
                                     # expected_B matrix 1
                                     np.array([
                                         [2, 0],
                                         [0, 2],
                                         [2, 0],
                                         [0, 2]
                                     ])
                                 ),

                                 (
                                     # dt 2
                                     1.0,
                                     # expected_F matrix 2
                                     np.array([
                                         [1, 0, 1, 0],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]
                                     ]),
                                     # expected_B matrix 2
                                     np.array([
                                         [1, 0],
                                         [0, 1],
                                         [1, 0],
                                         [0, 1]
                                     ])
                                 ),
                             ])
    def test_motion_matrices(self, dt, expected_F):
        """Check dt values are correctly used and output matrices are
        correct"""

        F, B = motion_matrices(dt)

        assert np.array_equal(F, expected_F), "F matrix does not match"
        assert np.array_equal(B, expected_B), "B matrix does not match"
