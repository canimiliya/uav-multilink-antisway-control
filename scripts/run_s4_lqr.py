"""Run selected S4 LQR on the three frozen scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from uav_sway.disturbances.wind_io import read_wind_csv
from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics
from uav_sway.evaluation.lqr_gate import raw_lqr_gate
from uav_sway.evaluation.lqr_runner import run_lqr_scenario
from uav_sway.evaluation.lqr_schema import lqr_schema_description
from uav_sway.evaluation.controlled_runner import ensure_calm_wind


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--model-config", required=True); parser.add_argument("--lqr-config", required=True); parser.add_argument("--scenarios", nargs="+", required=True); parser.add_argument("--headless", action="store_true"); parser.add_argument("--output-dir", required=True); args = parser.parse_args()
    if not args.headless: raise SystemExit("S4 runner requires --headless")
    output = Path(args.output_dir); config = yaml.safe_load(Path(args.lqr_config).read_text(encoding="utf-8")); gain = np.load(ROOT / "artifacts/s4/lqr/K.npy")
    calm = output / "inputs/calm.csv"; ensure_calm_wind(calm, read_wind_csv(ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv")["time"])
    winds = {"approach_stop": calm, "crosswind_hover": ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv", "gust_micro_adjust": ROOT / "artifacts/s2/wind_bank/one_cosine_gust.csv"}
    paths = {}
    for scenario in args.scenarios:
        path = output / "runs" / scenario / "run.csv"; paths[scenario] = path
        metrics = run_lqr_scenario(args.model_config, config, scenario, winds[scenario], ROOT / "artifacts/s2/references" / f"{scenario}.csv", path, ROOT, True, gain=gain)
        (path.parent / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    pid_paths = {s: ROOT / "artifacts/s3/runs" / s / "run.csv" for s in args.scenarios}
    gate = raw_lqr_gate(paths, pid_paths); (output / "raw_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8", newline="\n")
    summary = []
    for scenario, path in paths.items():
        m = compute_controlled_metrics(path); p = compute_controlled_metrics(pid_paths[scenario]); summary.append({"scenario": scenario, "x_position_rmse_m": m["x_position_rmse_m"], "z_position_rmse_m": m["z_position_rmse_m"], "final_x_error_m": m["final_x_error_m"], "tip_max_abs_m": m["tip_max_abs_m"], "tip_rms_m": m["tip_rms_m"], "pid_tip_rms_m": p["tip_rms_m"], "tip_rms_improvement_percent": 100.0 * (p["tip_rms_m"] - m["tip_rms_m"]) / p["tip_rms_m"], "minimum_tip_height_m": m["minimum_tip_height_m"], "control_rate_proxy": m["control_rate_proxy"], "pid_control_rate_proxy": p["control_rate_proxy"], "saturation_rate": m["saturation_rate"], "maximum_abs_roll_rad": m["maximum_abs_roll_rad"], "maximum_abs_pitch_rad": m["maximum_abs_pitch_rad"]})
    with (output / "lqr_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (output / "lqr_summary.json").write_text(json.dumps({"controller": "lqr", "scenarios": summary, "raw_gate": gate, "schema": lqr_schema_description(5)}, indent=2) + "\n", encoding="utf-8", newline="\n")
    comparison = {"source": "independent_raw_csv_recomputation", "scenarios": {}}
    for item in summary:
        scene = item["scenario"]
        pid = compute_controlled_metrics(pid_paths[scene]); lqr = compute_controlled_metrics(paths[scene])
        comparison["scenarios"][scene] = {"pid_tip_rms_m": pid["tip_rms_m"], "lqr_tip_rms_m": lqr["tip_rms_m"], "tip_rms_improvement_percent": 100.0 * (pid["tip_rms_m"] - lqr["tip_rms_m"]) / pid["tip_rms_m"], "pid_tip_max_m": pid["tip_max_abs_m"], "lqr_tip_max_m": lqr["tip_max_abs_m"], "pid_x_rmse_m": pid["x_position_rmse_m"], "lqr_x_rmse_m": lqr["x_position_rmse_m"], "position_rmse_change_percent": 100.0 * (lqr["x_position_rmse_m"] - pid["x_position_rmse_m"]) / pid["x_position_rmse_m"], "pid_control_rate_proxy": pid["control_rate_proxy"], "lqr_control_rate_proxy": lqr["control_rate_proxy"]}
    (output / "comparison/lqr_vs_pid.json").parent.mkdir(parents=True, exist_ok=True)
    (output / "comparison/lqr_vs_pid.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8", newline="\n")
    if not gate["pass"]: return 2
    return 0


if __name__ == "__main__": raise SystemExit(main())
