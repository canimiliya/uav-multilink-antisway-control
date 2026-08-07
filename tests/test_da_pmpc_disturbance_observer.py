import numpy as np
from uav_sway.control.disturbance_observer import MatchedDisturbanceObserver

def test_observer_is_17_state_and_does_not_accept_wind():
    o=MatchedDisturbanceObserver(np.eye(16),np.ones(16)); o.reset(np.zeros(16)); o.update(np.zeros(16),0.0)
    assert o.dimension==17 and o.augmented_state().shape==(17,)
    assert np.isfinite(o.augmented_state()).all()
