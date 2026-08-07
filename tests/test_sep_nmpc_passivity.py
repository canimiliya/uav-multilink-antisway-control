from uav_sway.paper_baseline.sep_nmpc_hocbf import build_hocbf_constraints
from uav_sway.paper_baseline.sep_nmpc_passivity import passivity_residual, passivity_satisfied


def test_passivity_slack_and_empty_hocbf_contract():
    assert passivity_satisfied(0.0, 0.0, 0.05, 0.005, 0.0)
    assert passivity_residual(1.0, 0.0, 0.05, 0.005, 0.0) > 0
    rows = build_hocbf_constraints([0, 0, 0, 0], [])
    assert rows.active is False
    assert rows.A.shape == (0, 1)
    assert rows.b.shape == (0,)
