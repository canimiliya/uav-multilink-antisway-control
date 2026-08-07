import numpy as np

from uav_sway.paper_baseline.sep_nmpc_model import PlanarParameters
from uav_sway.paper_baseline.sep_nmpc_ocp import SEPTrackingConfig, build_synthetic_casadi_opti


def test_casadi_synthetic_ocp_smoke():
    parameters = PlanarParameters(9.74, 3.5, 2.57)
    config = SEPTrackingConfig()
    opti, z, u, slack = build_synthetic_casadi_opti(parameters, config)
    solution = opti.solve()
    assert solution.stats()["success"]
    assert np.isfinite(np.asarray(solution.value(z))).all()
    assert np.isfinite(np.asarray(solution.value(u))).all()
    assert np.isfinite(np.asarray(solution.value(slack))).all()
    assert solution.value(z).shape == (4, 41)
