import numpy as np
import pytest

from motionfilter import simulate_motion


class TestSimulateMotion:
    """Tests for simulate_motion() in simulation.py"""

    @pytest.mark.parametrize("dt, n_steps, initial_state, expected_states",
                             [
                                 (
                                     # dt for case 1
                                     2.0,
                                     # n_steps for case 1
                                     10,
                                     # initial_state for case 1
                                     np.array([0, 0, 0, 0]),
                                     # expected_state for case 1
                                     np.zeros(
                                         (11, 4),
                                         dtype=np.float64
                                     )
                                 ),

                                 (
                                     # dt for case 2
                                     0.5,
                                     # n_steps for case 2
                                     3,
                                     # initial_state for case 2
                                     np.array([0, 1, 2, 3]),
                                     # expected state for case 3
                                     np.array(
                                         [
                                             [0, 1, 2, 3],
                                             [1, 2.5, 2, 3],
                                             [2, 4, 2, 3],
                                             [3, 5.5, 2, 3]
                                         ])
                                 ),
                             ])
    def test_simulate_motion(self, dt, n_steps, initial_state,
                             expected_states):
        """Check simulate_motion() function gives correct results"""

        # check if correct without acceleration implemented
        true_states = simulate_motion(dt, n_steps, initial_state)

        assert np.array_equal(true_states, expected_states)
