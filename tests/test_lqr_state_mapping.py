from pathlib import Path

import mujoco
import numpy as np

from uav_sway.control.base import ReferenceState
from uav_sway.linearization.equilibrium import build_initial_equilibrium
from uav_sway.linearization.reduced_state import ReducedStateLayout


ROOT = Path(__file__).resolve().parents[1]


def test_16_state_inject_extract_roundtrip_100_samples():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/s3/runtime/model_5link_controlled.xml"))
    data, snapshot = build_initial_equilibrium(model)
    layout = ReducedStateLayout(model)
    reference = ReferenceState(0.2, 0.1, 0.0, 0.0, 3.25, 0.0)
    rng = np.random.default_rng(20260807)
    scales = np.array([.02, .05, .01, .03, .01, .03, *([.01] * 5), *([.03] * 5)])
    maximum = 0.0
    for _ in range(100):
        state = rng.uniform(-1.0, 1.0, 16) * scales
        layout.inject(model, data, snapshot, state, reference)
        maximum = max(maximum, float(np.max(np.abs(layout.extract(model, data, reference) - state))))
    assert maximum < 1e-10


def test_state_order_contains_all_joint_angles_and_velocities():
    from uav_sway.linearization.reduced_state import STATE_NAMES
    assert STATE_NAMES[6:11] == [f"joint_{i}_angle" for i in range(1, 6)]
    assert STATE_NAMES[11:16] == [f"joint_{i}_velocity" for i in range(1, 6)]
