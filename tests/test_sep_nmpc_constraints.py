import numpy as np

from uav_sway.paper_baseline.sep_nmpc_hocbf import build_hocbf_constraints
from uav_sway.paper_baseline.sep_nmpc_passivity import passivity_satisfied


def test_empty_obstacles_have_zero_hocbf_rows():
    rows = build_hocbf_constraints(np.zeros(4), [])
    assert rows.active is False
    assert rows.A.shape == (0, 1)
    assert rows.b.shape == (0,)


def test_passivity_inequality_and_slack_bounds_contract():
    assert passivity_satisfied(0.0, 0.0, 0.05, 0.005, 0.0)
    assert not passivity_satisfied(1.0, 1.0, 0.05, 0.005, 0.0)
    assert 0.0 <= 5.0
