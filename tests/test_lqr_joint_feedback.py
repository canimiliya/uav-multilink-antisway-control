from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_all_five_joint_angles_and_velocities_have_nonzero_gain():
    report = json.loads((ROOT / "artifacts/s4/lqr/joint_feedback_audit.json").read_text(encoding="utf-8"))
    assert report["all_joint_angle_gains_nonzero"]
    assert report["all_joint_velocity_gains_nonzero"]
    assert report["minimum_abs_joint_angle_gain"] > 1e-8
    assert report["minimum_abs_joint_velocity_gain"] > 1e-8
