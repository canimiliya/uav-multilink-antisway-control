"""Run the frozen 27-point PID grid on the two allowed development cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from uav_sway.control.runtime_model import create_runtime_model
from uav_sway.evaluation.controlled_metrics import load_controlled_csv
from uav_sway.evaluation.controlled_runner import ensure_calm_wind, run_controlled_scenario
from uav_sway.disturbances.wind_io import read_wind_csv


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--controller-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if not args.headless:
        raise SystemExit("S3 tuning requires --headless")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / "artifacts/s3/runtime"
    create_runtime_model(ROOT / "artifacts/s1/generated/model_5link.xml", runtime / "model_5link_controlled.xml", runtime / "runtime_model_diff.json")
    calm = ROOT / "artifacts/s3/inputs/calm.csv"
    ensure_calm_wind(calm, read_wind_csv(ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv")["time"])
    controller = yaml.safe_load(Path(args.controller_config).read_text(encoding="utf-8"))
    candidates = []
    index = 0
    for kp in controller["kp_candidates"]:
        for kd in controller["kd_candidates"]:
            for ki in controller["ki_candidates"]:
                index += 1
                candidate = dict(controller, kp=float(kp), kd=float(kd), ki=float(ki))
                case_metrics = []
                gate = True
                for scenario, wind in (("approach_stop", calm), ("crosswind_hover", ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv")):
                    run = output / "candidates" / f"candidate_{index:02d}" / scenario / "run.csv"
                    metrics = run_controlled_scenario(args.model_config, candidate, scenario, wind, ROOT / "artifacts/s2/references" / f"{scenario}.csv", run, ROOT, True)
                    case_metrics.append(metrics)
                    gate = gate and bool(metrics["finite_outputs"] and metrics["minimum_uav_height_m"] > 0.05 and metrics["minimum_tip_height_m"] > 0.05 and metrics["maximum_abs_roll_rad"] < np.pi / 2 and metrics["maximum_abs_pitch_rad"] < np.pi / 2 and abs(metrics["final_x_error_m"]) <= 0.40)
                score = float(np.mean([m["x_position_rmse_m"] for m in case_metrics]) + 0.25 * np.mean([abs(m["final_x_error_m"]) for m in case_metrics]) + 0.10 * np.mean([m["saturation_rate"] for m in case_metrics]) + 0.001 * np.mean([m["control_rate_proxy"] for m in case_metrics]))
                candidates.append({"index": index, "kp": float(kp), "kd": float(kd), "ki": float(ki), "safe_gate": bool(gate), "score": score, "approach_x_rmse": case_metrics[0]["x_position_rmse_m"], "crosswind_x_rmse": case_metrics[1]["x_position_rmse_m"], "approach_final_x_error": case_metrics[0]["final_x_error_m"], "crosswind_final_x_error": case_metrics[1]["final_x_error_m"]})
                print(json.dumps(candidates[-1]))
    columns = list(candidates[0])
    with (output / "pid_grid.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(candidates)
    valid = [row for row in candidates if row["safe_gate"]]
    if not valid:
        raise SystemExit("BLOCKED: no PID candidate passed the frozen safety gate")
    selected = min(valid, key=lambda row: row["score"])
    selection = {"selected": selected, "grid_size": 27, "development_scenarios": ["approach_stop", "crosswind_hover"], "score_formula": "mean(x_rmse)+0.25*mean(abs(final_x_error))+0.10*mean(saturation_rate)+0.001*mean(control_rate_proxy)", "sway_metrics_used_for_selection": False}
    (output / "pid_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")
    controller["kp"], controller["kd"], controller["ki"] = selected["kp"], selected["kd"], selected["ki"]
    Path(args.controller_config).write_text(yaml.safe_dump(controller, sort_keys=False), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
