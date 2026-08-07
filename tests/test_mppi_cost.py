import numpy as np

from uav_sway.mppi.cost import mppi_candidate_cost, mppi_candidate_score


def test_stage_and_terminal_costs_use_frozen_weights():
    states = np.zeros((2, 16)); states[0, 0] = 1.0; states[1, 0] = 2.0
    tips = np.array([0.1, 0.2]); actions = np.array([0.5, 0.25])
    q = np.eye(16); r = np.ones((1, 1))
    value = mppi_candidate_cost(states, tips, actions, q, r, 80.0, 5.0)
    expected = (1.0 + 80*.1**2 + .5**2 + 4.0 + 80*.2**2 + .25**2
                + 5.0*(4.0 - 80*.2**2))
    assert np.isclose(value, expected)


def test_candidate_score_penalties_have_forward_direction():
    base = mppi_candidate_score([1], [1], [1], [1])
    assert mppi_candidate_score([1.1], [1], [1], [1]) > base
    assert mppi_candidate_score([1], [1.1], [1], [1]) > base
    assert mppi_candidate_score([1], [1], [1.1], [1]) > base
    assert mppi_candidate_score([1], [1], [1], [1.1]) > base
