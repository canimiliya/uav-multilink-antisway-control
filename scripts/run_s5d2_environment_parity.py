"""Run frozen LQR/LS-PMPC parity in the WSL benchmark environment.

This script is deliberately separate from SEP tuning.  It only reruns the
existing frozen LQR and LS-PMPC development configurations for the two S5D2
scenarios and writes an independent parity record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from uav_sway.disturbances.wind_io import read_wind_csv
from uav_sway.evaluation.da_pmpc_runner import run_scene
from uav_sway.evaluation.lqr_runner import run_lqr_scenario
from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics, load_controlled_csv
from uav_sway.evaluation.s5b_holdout import safety_check


ROOT = Path(__file__).resolve().parents[1]
SCENES = ("approach_stop", "crosswind_hover")
EXPECTED = {
    "LQR": {
        "approach_stop": {"x_rmse_m": 0.1053850612, "tip_rms_m": 0.1688590761},
        "crosswind_hover": {"x_rmse_m": 0.0919234887, "tip_rms_m": 0.0483626135},
    },
    "LS-PMPC": {
        "approach_stop": {"x_rmse_m": 0.0729211832, "tip_rms_m": 0.1414353663},
        "crosswind_hover": {"x_rmse_m": 0.0848853553, "tip_rms_m": 0.0479697696},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_calm(path: Path, reference_path: Path) -> None:
    with reference_path.open("r", encoding="utf-8", newline="") as stream:
        times = [float(row["time"]) for row in csv.DictReader(stream)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("time,wind_x,wind_y,wind_z,profile,seed\n")
        for t in times:
            stream.write(f"{float(t):.17g},0,0,0,calm,\n")


def parity_metrics(path: Path, settling_start_s: float) -> dict:
    metrics = compute_controlled_metrics(path, settling_start_s)
    _, values = load_controlled_csv(path)
    metrics.update({
        "thrust_min_N": float(np.min(values["thrust_cmd_limited_N"])),
        "thrust_max_N": float(np.max(values["thrust_cmd_limited_N"])),
        "torque_max_abs_Nm": float(np.max(np.abs(np.column_stack([
            values["mx_cmd_limited_Nm"], values["my_cmd_limited_Nm"], values["mz_cmd_limited_Nm"]
        ])))),
        "rotor_motor_max_abs_cmd": float(np.max(np.abs(values.get("rotor_motor_max_abs_cmd", np.zeros_like(values["time"]))))),
    })
    return metrics


def run_parity(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    references = {scene: ROOT / "artifacts/s2/references" / f"{scene}.csv" for scene in SCENES}
    wind_paths = {
        "approach_stop": output / "inputs/calm.csv",
        "crosswind_hover": ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv",
    }
    write_calm(wind_paths["approach_stop"], references["approach_stop"])

    lqr_config = yaml.safe_load((ROOT / "configs/lqr.yaml").read_text(encoding="utf-8"))
    da_config = yaml.safe_load((ROOT / "configs/da_pmpc.yaml").read_text(encoding="utf-8"))
    gain = np.load(ROOT / "artifacts/s4/lqr/K.npy")
    rows: dict[str, dict[str, dict]] = {"LQR": {}, "LS-PMPC": {}}
    for scene in SCENES:
        lqr_path = output / "runs/LQR" / scene / "run.csv"
        run_lqr_scenario(
            ROOT / "configs/model_5link.yaml", lqr_config, scene, wind_paths[scene],
            references[scene], lqr_path, ROOT, True, gain=gain,
        )
        rows["LQR"][scene] = parity_metrics(lqr_path, 6.0 if scene == "approach_stop" else 4.0)

        pmpc_path = output / "runs/LS-PMPC" / scene / "run.csv"
        run_scene(
            ROOT / "configs/model_5link.yaml", da_config, scene, wind_paths[scene],
            references[scene], pmpc_path, mode="preview", duration_s=12.0,
        )
        rows["LS-PMPC"][scene] = parity_metrics(pmpc_path, 6.0 if scene == "approach_stop" else 4.0)

    comparisons = {}
    passed = True
    for controller in ("LQR", "LS-PMPC"):
        comparisons[controller] = {}
        for scene in SCENES:
            metrics = rows[controller][scene]
            expected = EXPECTED[controller][scene]
            rel_x = abs(metrics["x_position_rmse_m"] - expected["x_rmse_m"]) / max(abs(expected["x_rmse_m"]), 1e-9)
            rel_tip = abs(metrics["tip_rms_m"] - expected["tip_rms_m"]) / max(abs(expected["tip_rms_m"]), 1e-9)
            safe, reasons = safety_check(metrics)
            item = {
                "sample_count": metrics["sample_count"],
                "x_rmse_m": metrics["x_position_rmse_m"],
                "tip_rms_m": metrics["tip_rms_m"],
                "frozen_x_rmse_m": expected["x_rmse_m"],
                "frozen_tip_rms_m": expected["tip_rms_m"],
                "relative_x_error": rel_x,
                "relative_tip_error": rel_tip,
                "relative_error_limit": 0.005,
                "safe": safe,
                "safety_reasons": reasons,
                "pass": bool(metrics["sample_count"] == 2401 and rel_x < 0.005 and rel_tip < 0.005 and safe),
            }
            comparisons[controller][scene] = item
            passed = passed and item["pass"]

    result = {
        "result": "PASS_ENVIRONMENT_PARITY" if passed else "BLOCKED_ENVIRONMENT_PARITY",
        "pass": bool(passed),
        "environment": "WSL2 same-process MuJoCo/controller benchmark",
        "python": "3.11.0rc1",
        "mujoco": "3.0.1",
        "casadi": "3.7.2",
        "acados_head": "4c23274e49e1304cf3c859d59ea6694ce36305a7",
        "scenarios": list(SCENES),
        "duration_s": 12.0,
        "physics_hz": 1000,
        "inner_hz": 200,
        "outer_hz": 20,
        "same_sample_count_required": 2401,
        "same_reference_sha256": {scene: sha256(references[scene]) for scene in SCENES},
        "wind_sha256": {scene: sha256(wind_paths[scene]) for scene in SCENES},
        "controllers": comparisons,
        "selection_started": False,
    }
    (output.parent / "environment_parity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "artifacts/s5d2/parity"))
    args = parser.parse_args()
    result = run_parity(Path(args.output))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
