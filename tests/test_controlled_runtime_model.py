import json
from pathlib import Path

import mujoco
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_model_changes_only_wrench_ranges():
    diff = json.loads((ROOT / "artifacts/s3/runtime/runtime_model_diff.json").read_text(encoding="utf-8"))
    assert diff["physics_fingerprint_equal"] is True
    assert diff["actuator_changes_only"] is True
    assert diff["total_mass_kg"] == pytest.approx(13.24, abs=1e-12)
    assert diff["computed_max_thrust_N"] == 285.74568
    model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/s3/runtime/model_5link_controlled.xml"))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert np.allclose(model.actuator_ctrlrange[names.index("thrust_motor")], [0.0, 285.74568])
    for name in ("rotor_motor_0", "rotor_motor_1", "rotor_motor_2", "rotor_motor_3"):
        assert np.allclose(model.actuator_ctrlrange[names.index(name)], [0.0, 10.0])
