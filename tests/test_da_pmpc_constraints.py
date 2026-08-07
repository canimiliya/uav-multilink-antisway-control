import numpy as np
from uav_sway.mpc.preview_model import PreviewModel
from uav_sway.mpc.qp_builder import build_preview_qp
from uav_sway.control.base import ReferenceState

def test_qp_has_amplitude_and_slew_rows():
    m=PreviewModel(np.eye(16),np.ones((16,1)),np.eye(16),np.eye(16),np.zeros((1,16)),20)
    r=tuple(ReferenceState(0,0,0,0,3.2,0) for _ in range(21)); qp=build_preview_qp(m,np.zeros(16),r,0,1,20)
    assert qp.P.shape==(20,20) and qp.A.shape==(40,20); assert qp.lower.shape==(40,)
