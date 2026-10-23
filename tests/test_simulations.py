import numpy as np
import pytest

from motionfilter import simulate_motion
import motionfilter.simulate as simulation


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
        vehicle_history = simulate_motion(dt, n_steps, initial_state,
                                          accelerations=np.array([0, 0]),
                                          acceleration_std=0.0)

        assert np.array_equal(vehicle_history, expected_states)

    # monkeypatch used here to skip generate_acceleration_noise and use
    # predeterminted acceleration noise values.:w

    def test_simulate_motion_with_noise(self, monkeypatch):
        """Test simulation with predetermined acceleration noise."""

        noise1 = np.array([0.5, -0.5])
        noise2 = np.array([-0.5, 0.5])

        noises = iter([noise1, noise2])

        def fake_noise_generator(acceleration_std, rng):
            return next(noises)

        monkeypatch.setattr(
            simulation,
            "generate_acceleration_noise",
            fake_noise_generator
        )

        expected_states = np.array([
            [0, 1, 3, 10],
            [11, 24, 8, 13],
            [30, 55, 11, 18]
        ], dtype=float)

        actual_states = simulate_motion(
            dt=2.0,
            n_steps=2,
            initial_state=np.array([0, 1, 3, 10]),
            acceleration_std=0.5,
            accelerations=np.array([2, 2]),
        )

        np.testing.assert_allclose(actual_states, expected_states)
