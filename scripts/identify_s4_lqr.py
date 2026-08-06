"""Identify and save the nominal five-link S4 reduced closed loop."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from uav_sway.control.runtime_model import sha256_file
from uav_sway.linearization.analysis import controllability_analysis
from uav_sway.linearization.equilibrium import EQUILIBRIUM_REFERENCE, find_equilibrium, save_equilibrium
from uav_sway.linearization.finite_difference import central_finite_difference
from uav_sway.linearization.reduced_state import STATE_NAMES
from uav_sway.linearization.validation import local_validation, operating_region_validation


ROOT = Path(__file__).resolve().parents[1]


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",", fmt="%.17g")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-model", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    linear = output / "linearization"
    model_path = ROOT / args.runtime_model if not Path(args.runtime_model).is_absolute() else Path(args.runtime_model)
    config_path = ROOT / "configs/lqr.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    result = find_equilibrium(model, config)
    save_equilibrium(result, output / "equilibrium/equilibrium_state.npz", output / "equilibrium/equilibrium_summary.json")
    if result["final_residual"] >= 1e-8:
        raise SystemExit(f"BLOCKED: equilibrium residual {result['final_residual']}")
    eps = config["state_epsilon"]
    state_eps = np.asarray([
        eps["position_error_x"], eps["velocity_error_x"], eps["altitude_error"], eps["vertical_velocity"],
        eps["pitch"], eps["body_pitch_rate"], *([eps["joint_angle_each"]] * 5), *([eps["joint_velocity_each"]] * 5),
    ], dtype=float)
    input_eps = float(config["input_epsilon"])
    phi = result["step"]
    a1, b1 = central_finite_difference(phi, state_eps, input_eps)
    a2, b2 = central_finite_difference(phi, state_eps / 2.0, input_eps / 2.0)
    a_repeat, b_repeat = central_finite_difference(phi, state_eps, input_eps)
    a_half_error = float(np.linalg.norm(a1 - a2, ord="fro") / max(np.linalg.norm(a2, ord="fro"), 1e-12))
    b_half_error = float(np.linalg.norm(b1 - b2, ord="fro") / max(np.linalg.norm(b2, ord="fro"), 1e-12))
    repeat_a = float(np.max(np.abs(a1 - a_repeat)))
    repeat_b = float(np.max(np.abs(b1 - b_repeat)))
    linear.mkdir(parents=True, exist_ok=True)
    np.save(linear / "A.npy", a1); np.save(linear / "B.npy", b1)
    _write_matrix(linear / "A.csv", a1); _write_matrix(linear / "B.csv", b1)
    np.savez(linear / "linear_model.npz", A=a1, B=b1, state_epsilon=state_eps, input_epsilon=input_eps)
    (linear / "state_order.json").write_text(json.dumps({"dimension": 16, "state_order": STATE_NAMES}, indent=2) + "\n", encoding="utf-8", newline="\n")
    fd = {"state_epsilon": state_eps.tolist(), "input_epsilon": input_eps, "half_epsilon_state_relative_fro_error": a_half_error, "half_epsilon_input_relative_fro_error": b_half_error, "repeat_max_abs_A_error": repeat_a, "repeat_max_abs_B_error": repeat_b, "A_shape": list(a1.shape), "B_shape": list(b1.shape), "finite": bool(np.isfinite(a1).all() and np.isfinite(b1).all()), "central_difference": True}
    (linear / "finite_difference.json").write_text(json.dumps(fd, indent=2) + "\n", encoding="utf-8", newline="\n")
    analysis = controllability_analysis(a1, b1)
    (linear / "controllability.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8", newline="\n")
    if not analysis["pbh_stabilizable"]:
        raise SystemExit("BLOCKED_NOT_STABILIZABLE")
    validation = operating_region_validation(phi, a1, b1)
    # Keep the historical filename and also give the retained wide-range
    # result its explicit non-local name.  This is a limitation record, not a
    # local Jacobian pass/fail claim.
    (linear / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8", newline="\n")
    (linear / "operating_region_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8", newline="\n")

    epsilon = state_eps
    phi_zero = np.asarray(phi(np.zeros(16, dtype=float), 0.0), dtype=float)
    local, per_state = local_validation(phi, a1, b1, phi_zero, epsilon, input_eps)
    local["state_order"] = STATE_NAMES
    local["worst_state"] = STATE_NAMES[local["worst_state_index"]]
    for item in local["by_multiplier"].values():
        item["worst_state"] = STATE_NAMES[item["worst_state_index"]]
    per_state_named = {key: {STATE_NAMES[i]: {
        "absolute_rmse": values["absolute_rmse"][i],
        "normalized_rmse": values["normalized_rmse"][i],
        "p95_error": values["p95_error"][i],
    } for i in range(16)} | {"worst_state": STATE_NAMES[values["worst_state_index"]]} for key, values in per_state.items()}
    (linear / "local_validation.json").write_text(json.dumps(local, indent=2) + "\n", encoding="utf-8", newline="\n")
    (linear / "per_state_error.json").write_text(json.dumps(per_state_named, indent=2) + "\n", encoding="utf-8", newline="\n")
    hinge_dofs = [int(model.jnt_dofadr[joint_id]) for joint_id in range(1, model.njnt)]
    friction = float(np.max(model.dof_frictionloss[hinge_dofs])) if hinge_dofs else 0.0
    scale_audit = {
        "finite_difference_state_epsilon": epsilon.tolist(),
        "finite_difference_input_epsilon": input_eps,
        "local_validation_multipliers": [2, 5, 10],
        "local_validation_sample_count": 200,
        "local_validation_seed": 20260808,
        "operating_region_relative_to_epsilon": (np.asarray(validation["state_ranges"]) / epsilon).tolist(),
        "operating_region_input_relative_to_epsilon": 0.05 / input_eps,
        "model_hinge_frictionloss": friction,
        "interpretation": "Friction may contribute to non-smoothness; this record does not establish it as the sole cause without a controlled comparison.",
    }
    (linear / "validation_scale_audit.json").write_text(json.dumps(scale_audit, indent=2) + "\n", encoding="utf-8", newline="\n")
    policy = """# S4 线性验证政策变更

旧方法：
用比有限差分 epsilon 大 50～1000 倍的区域，直接作为平衡点 Jacobian 的局部通过门槛。

问题：
该测试衡量的是较宽运行区域的一阶近似能力，不能单独判断平衡点 Jacobian 是否正确。

新方法：
1. 中心有限差分重复性与半 epsilon 收敛；
2. 10×epsilon 真正局部验证作为 Jacobian 验收；
3. 原宽范围验证完整保留为 operating-region limitation；
4. 最终以真实非线性三场景安全性、位置公平性和摆动改善作为控制器验收。

说明：`model_hinge_frictionloss=0.005` 已记录在 `validation_scale_audit.json`。摩擦可能导致非光滑性，但没有对照实验时不宣称其为唯一原因。
"""
    (linear / "validation_policy_change.md").write_text(policy, encoding="utf-8", newline="\n")
    dependencies = {
        "runtime_model_sha256": sha256_file(model_path),
        "source_model_sha256": sha256_file(ROOT / "artifacts/s1/generated/model_5link.xml"),
        "pid_summary_sha256": sha256_file(ROOT / "artifacts/s3/pid_summary.json"),
        "inner_loop_source_sha256": sha256_file(ROOT / "src/uav_sway/control/geometric_inner_loop.py"),
        "state_reader_source_sha256": sha256_file(ROOT / "src/uav_sway/control/state_reader.py"),
        "physics_dt_s": float(model.opt.timestep), "inner_loop_dt_s": 0.005, "outer_loop_dt_s": 0.05,
        "ax_limits_m_s2": [-2.0, 2.0], "ax_slew_limit_m_s2_per_update": 0.25,
        "runtime_model_expected_sha256": "19105873c0fcc891ebb85efe6c20c378d5b77b6bf9003559e43ae47ca03d153d",
        "source_model_expected_sha256": "a4a0f641ea3c579c893a00c9db52217a066b9de9f8667d40917157a4f3c72a0a",
    }
    expected = {"runtime_model_sha256": dependencies["runtime_model_expected_sha256"], "source_model_sha256": dependencies["source_model_expected_sha256"]}
    if any(dependencies[key].lower() != value.lower() for key, value in expected.items()):
        raise SystemExit("BLOCKED_DEPENDENCY_DRIFT")
    (output / "dependencies.json").write_text(json.dumps(dependencies, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"equilibrium": result["final_residual"], "A_half_error": a_half_error, "B_half_error": b_half_error, "controllability_rank": analysis["rank"], "operating_region_median": validation["median_normalized_error"], "operating_region_p95": validation["p95_normalized_error"], "local_10x_median": local["median_normalized_error"], "local_10x_p95": local["p95_normalized_error"], "local_pass": local["pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
