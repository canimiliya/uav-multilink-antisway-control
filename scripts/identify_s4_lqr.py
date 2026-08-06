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
    rng = np.random.default_rng(20260807)
    scales = np.asarray([0.02, 0.05, 0.01, 0.03, 0.01, 0.03, *([0.01] * 5), *([0.03] * 5)])
    errors = []
    for _ in range(20):
        state = rng.uniform(-1.0, 1.0, 16) * scales
        u = float(rng.uniform(-0.05, 0.05))
        nonlinear = phi(state, u)
        predicted = a1 @ state + b1[:, 0] * u
        normalized = np.linalg.norm((nonlinear - predicted) / np.maximum(scales, 1e-12)) / np.sqrt(16.0)
        errors.append(float(normalized))
    validation = {"seed": 20260807, "sample_count": 20, "normalized_errors": errors, "median_normalized_error": float(np.median(errors)), "p95_normalized_error": float(np.percentile(errors, 95)), "finite": bool(np.isfinite(errors).all())}
    (linear / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8", newline="\n")
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
    print(json.dumps({"equilibrium": result["final_residual"], "A_half_error": a_half_error, "B_half_error": b_half_error, "controllability_rank": analysis["rank"], "validation_median": validation["median_normalized_error"], "validation_p95": validation["p95_normalized_error"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
