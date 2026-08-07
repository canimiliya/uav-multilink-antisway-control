"""Generate reproducibility and analytical evidence for the S5A2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

from uav_sway.control.base import ReferenceState
from uav_sway.mpc.preview_model import PreviewModel


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/s5a/s5a2")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    A = np.load(ROOT / "artifacts/s4/linearization/A.npy")
    B = np.load(ROOT / "artifacts/s4/linearization/B.npy")
    Q = np.load(ROOT / "artifacts/s4/lqr/Q.npy")
    R = np.load(ROOT / "artifacts/s4/lqr/R.npy")
    K_frozen = np.load(ROOT / "artifacts/s4/lqr/K.npy")
    P = solve_discrete_are(A, B, Q, R)
    K_dare = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

    dt = 0.05
    A_simple = np.eye(16)
    A_simple[0, 1] = dt
    B_simple = np.zeros((16, 1))
    B_simple[0, 0] = 0.5 * dt * dt
    B_simple[1, 0] = dt
    model = PreviewModel(A_simple, B_simple, Q, P, np.zeros((1, 16)), 1)
    constant_velocity_0 = ReferenceState(0.0, 0.75, 0.0, 0.0, 3.2, 0.0)
    constant_velocity_1 = ReferenceState(0.75 * dt, 0.75, 0.0, 0.0, 3.2, 0.0)
    cv_shift = model.reference_shift(constant_velocity_0, constant_velocity_1)
    constant_acceleration = 0.5
    constant_acc_0 = ReferenceState(0.0, 0.0, constant_acceleration, 0.0, 3.2, 0.0)
    constant_acc_1 = ReferenceState(0.5 * constant_acceleration * dt**2, constant_acceleration * dt, constant_acceleration, 0.0, 3.2, 0.0)
    ca_shift = model.reference_shift(constant_acc_0, constant_acc_1)
    write_json(output / "reference_shift_audit.json", {
        "formula": "reference_vector(ref_next) - A @ reference_vector(ref_i)",
        "reference_vector": ["x_ref", "vx_ref", "z_ref - 3.2", "zeros(13)"],
        "constant_velocity_old_false_error_m": -0.0375,
        "constant_velocity_new_error_m": float(-cv_shift[0]),
        "constant_velocity_new_error_vector_norm": float(np.linalg.norm(-cv_shift)),
        "constant_acceleration_new_error_vector_norm": float(np.linalg.norm(B_simple[:, 0] * constant_acceleration - ca_shift)),
        "constant_velocity_pass": bool(np.linalg.norm(-cv_shift) < 1e-12),
        "constant_acceleration_pass": bool(np.linalg.norm(B_simple[:, 0] * constant_acceleration - ca_shift) < 1e-12),
    })
    runtime_sha = sha256(ROOT / "artifacts/s3/runtime/model_5link_controlled.xml")
    write_json(output / "dependencies.json", {
        "runtime_model_sha256": runtime_sha,
        "expected_runtime_model_sha256": "19105873c0fcc891ebb85efe6c20c378d5b77b6bf9003559e43ae47ca03d153d",
        "s4_A_sha256": sha256(ROOT / "artifacts/s4/linearization/A.npy"),
        "s4_B_sha256": sha256(ROOT / "artifacts/s4/linearization/B.npy"),
        "s4_Q_sha256": sha256(ROOT / "artifacts/s4/lqr/Q.npy"),
        "s4_R_sha256": sha256(ROOT / "artifacts/s4/lqr/R.npy"),
        "s4_K_sha256": sha256(ROOT / "artifacts/s4/lqr/K.npy"),
        "tip_output_sha256": sha256(ROOT / "artifacts/s5a/model/C_tip.npy"),
        "inner_loop_source_sha256": sha256(ROOT / "src/uav_sway/control/geometric_inner_loop.py"),
        "reduced_state_source_sha256": sha256(ROOT / "src/uav_sway/linearization/reduced_state.py"),
        "physics_dt_s": 0.001,
        "inner_loop_dt_s": 0.005,
        "outer_loop_dt_s": 0.05,
        "ax_limits_m_s2": [-2.0, 2.0],
        "ax_slew_limit_m_s2_per_update": 0.25,
        "A_shape": list(A.shape),
        "B_shape": list(B.shape),
        "s4_dare_K_max_abs_error": float(np.max(np.abs(K_dare - K_frozen))),
        "s4_dare_K_parity": bool(np.max(np.abs(K_dare - K_frozen)) < 1e-10),
        "A_cl_spectral_radius": float(max(abs(np.linalg.eigvals(A - B @ K_frozen)))),
        "dependency_match": bool(runtime_sha == "19105873c0fcc891ebb85efe6c20c378d5b77b6bf9003559e43ae47ca03d153d"),
    })
    write_json(output / "controller_contract.json", {
        "controller": "lqr_stabilized_da_pmpc",
        "control_law": "a_x = a_ref - K e - d_hat - v_MPC",
        "zero_residual_parity_law": "a_x = a_ref - K e",
        "residual_variable": "v_MPC",
        "actual_input_constraints": {"acceleration": [-2.0, 2.0], "slew_per_update": 0.25},
        "horizon_steps": 20,
        "preview_seconds": 1.0,
        "solver": "OSQP warm start",
        "development_scenarios": ["approach_stop", "crosswind_hover"],
        "gust_used": False,
        "random_holdout_used": False,
        "tip_weight_candidates": [20.0, 40.0, 80.0],
        "residual_weight_candidates": [2.0, 8.0, 32.0],
    })


if __name__ == "__main__":
    main()
