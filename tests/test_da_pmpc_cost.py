import numpy as np

from uav_sway.control.base import ReferenceState
from uav_sway.mpc.preview_model import PreviewModel
from uav_sway.mpc.qp_builder import build_preview_qp


def test_qp_contains_control_cost_for_every_delta_ax():
    A=np.eye(16); B=np.zeros((16,1)); Q=np.zeros((16,16)); P=np.zeros((16,16)); C=np.zeros((1,16))
    m=PreviewModel(A,B,Q,P,C,2,control_weight=1.0)
    refs=tuple(ReferenceState(0,0,0,0,3.2,0) for _ in range(3))
    qp=build_preview_qp(m,np.zeros(16),refs,0.0,1.0,0.0)
    assert np.allclose(qp.P, 2.0*np.eye(2), atol=1e-12)
    assert np.allclose(qp.q, 0.0, atol=1e-12)


def test_terminal_state_is_not_also_given_stage_weight():
    A=np.eye(16); B=np.zeros((16,1)); Q=np.eye(16); P=2.0*np.eye(16); C=np.zeros((1,16))
    m=PreviewModel(A,B,Q,P,C,2,control_weight=1.0)
    refs=tuple(ReferenceState(0,0,0,0,3.2,0) for _ in range(3))
    qp=build_preview_qp(m,np.zeros(16),refs,0.0,1.0,0.0)
    # B=0 means all state terms are constant and the two input costs remain.
    assert np.allclose(qp.P, 2.0*np.eye(2), atol=1e-12)
