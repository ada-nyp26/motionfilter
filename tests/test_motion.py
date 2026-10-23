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
    def test_matrix_generation(self, dt, expected_F, expected_B):
        """Check dt values are correctly used and output matrices are
        correct"""

        F, B = motion_matrices(dt)

        assert np.array_equal(F, expected_F), "F matrix does not match"
        assert np.array_equal(B, expected_B), "B matrix does not match"

    def test_matrix_shape(self):
        """Check both F and B matrix shapes are correct"""

        F, B = motion_matrices(0.5)

        assert F.shape == (4, 4)
        assert B.shape == (4, 2)

    def test_matrix_type(self):
        """Check both generated matrices are both of dtype float"""
        F, B = motion_matrices(0.5)

        assert F.dtype, B.dtype == float64
