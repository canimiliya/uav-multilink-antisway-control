import numpy as np
from uav_sway.control.base import ReferenceState
from uav_sway.mpc.preview_model import PreviewModel

def refs(n): return tuple(ReferenceState(float(i),1,0,0,3.2,0) for i in range(n))

def test_preview_has_frozen_dimensions_and_shift():
    A=np.eye(16); B=np.ones((16,1)); Q=np.eye(16); P=np.eye(16); C=np.zeros((1,16)); m=PreviewModel(A,B,Q,P,C,20)
    out=m.rollout(np.zeros(16),np.zeros(20),refs(21)); assert out.states.shape==(21,16); assert np.isfinite(out.states).all()
    assert np.isclose(out.states[1,0],-1.0)


def test_preview_first_order_input_has_explicit_positive_b_sign():
    A=np.eye(16); B=np.arange(1,17,dtype=float).reshape(16,1); Q=np.zeros((16,16)); P=np.zeros((16,16)); C=np.zeros((1,16))
    m=PreviewModel(A,B,Q,P,C,1)
    r=tuple(ReferenceState(0,0,0,0,3.2,0) for _ in range(2))
    out=m.rollout(np.zeros(16),np.asarray([0.5]),r)
    assert np.allclose(out.states[1], B[:,0]*0.5)
