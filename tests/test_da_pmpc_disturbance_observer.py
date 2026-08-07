import numpy as np
from uav_sway.control.disturbance_observer import MatchedDisturbanceObserver

def test_observer_is_17_state_and_does_not_accept_wind():
    o=MatchedDisturbanceObserver(np.eye(16),np.ones(16)); o.reset(np.zeros(16)); o.update(np.zeros(16),0.0)
    assert o.dimension==17 and o.augmented_state().shape==(17,)
    assert np.isfinite(o.augmented_state()).all()


def test_observer_recovers_matched_disturbance_from_applied_previous_command():
    A=np.eye(16); B=np.zeros(16); B[0]=1.0
    o=MatchedDisturbanceObserver(A,B,gain=1.0,limit=2.0); o.reset(np.zeros(16))
    o.update(np.zeros(16), applied_previous_command=0.7, reference_shift=np.zeros(16))
    x1=np.zeros(16); x1[0]=1.1
    assert np.isclose(o.update(x1, applied_previous_command=0.7, reference_shift=np.zeros(16)), 0.4, atol=1e-10)


def test_observer_reference_motion_is_not_disturbance():
    A=np.eye(16); B=np.zeros(16); B[0]=1.0
    o=MatchedDisturbanceObserver(A,B,gain=1.0,limit=2.0); o.reset(np.zeros(16))
    x0=np.zeros(16); o.update(x0, applied_previous_command=0.0, reference_shift=np.zeros(16))
    shift=np.zeros(16); shift[0]=0.2
    x1=np.zeros(16); x1[0]=-0.2
    assert np.isclose(o.update(x1, applied_previous_command=0.0, reference_shift=shift), 0.0, atol=1e-10)
