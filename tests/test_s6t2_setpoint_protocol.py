import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from uav_sway.control.base import ReferenceState
from uav_sway.task_space.reference import build_equilibrium_task_pose
from uav_sway.task_space.setpoint_protocol import (
    build_setpoint_protocol,
    protocol_reference_audit,
    write_setpoint_reference,
)


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    model_path = ROOT / "artifacts/s3/runtime/model_5link_controlled.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path)); data = mujoco.MjData(model)
    data.qpos[:] = 0.0; data.qpos[:7] = [0.0, 0.0, 3.2, 1.0, 0.0, 0.0, 0.0]; data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return build_setpoint_protocol(model, data, model_path, yaml.safe_load((ROOT / "configs/s6_taskspace_setpoint.yaml").read_text(encoding="utf-8")))


def _scene(name, start):
    return type("Scene", (), {"name": name, "task_start_time_s": start, "wind_profile": "calm", "source_scenario": "approach_stop"})()


def test_equilibrium_target_is_measured_and_delta_is_exact():
    protocol = _protocol()
    assert np.allclose(protocol.initial_tip_position_m, [0.225, 0.0, 0.39])
    assert np.allclose(protocol.target_tip_position_m, protocol.initial_tip_position_m + [0.30, 0.0, 0.0])
    assert np.allclose(protocol.target_cutter_axis, [1.0, 0.0, 0.0])


def test_calm_setpoint_is_constant_after_one_second_and_not_visible_before_issue(tmp_path):
    protocol = _protocol(); path = tmp_path / "calm.csv"
    write_setpoint_reference(path, protocol, _scene("task_acquire_calm", 1.0))
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    before = [row for row in rows if float(row["time"]) < 1.0]
    after = [row for row in rows if float(row["time"]) >= 1.0]
    assert all(float(row["x_ref"]) == 0.0 for row in before)
    assert all(float(row["x_ref"]) == 0.30 for row in after)
    assert all(float(row["vx_ref"]) == 0.0 and float(row["ax_ref"]) == 0.0 for row in rows)
    assert protocol_reference_audit(path, protocol, _scene("task_acquire_calm", 1.0))["pass"]


def test_crosswind_setpoint_starts_at_three_seconds(tmp_path):
    protocol = _protocol(); path = tmp_path / "crosswind.csv"
    scene = _scene("task_acquire_crosswind", 3.0)
    write_setpoint_reference(path, protocol, scene)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    assert float(rows[600]["time"]) == 3.0
    assert float(rows[599]["x_ref"]) == 0.0
    assert float(rows[600]["x_ref"]) == 0.30
    assert protocol_reference_audit(path, protocol, scene)["pass"]


def test_old_controller_reference_derivatives_are_zero_after_step(tmp_path):
    protocol = _protocol(); path = tmp_path / "reference.csv"
    write_setpoint_reference(path, protocol, _scene("task_acquire_calm", 1.0))
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    after = [row for row in rows if float(row["time"]) >= 1.0]
    references = [ReferenceState(float(row["x_ref"]), float(row["vx_ref"]), float(row["ax_ref"]), 0.0, 3.2, 0.0) for row in after]
    assert all(reference.vx_ref == 0.0 and reference.ax_ref == 0.0 for reference in references)


def test_s6t0_metric_contract_is_not_changed():
    definition = json.loads((ROOT / "artifacts/s6_taskspace/t0/task_metric_definition.json").read_text(encoding="utf-8"))
    assert definition["task_acquisition_time_origin"] == "elapsed from task_start_time, not absolute simulation timestamp"
    assert definition["control_rate_proxy_formula"] == "sum((diff(u) / diff(t))^2 * diff(t))"
