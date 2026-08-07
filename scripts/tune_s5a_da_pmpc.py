"""Run the frozen 3x3 tip-weight/residual-weight S5A2 pilot grid."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from uav_sway.evaluation.da_pmpc_gate import gate_scene
from uav_sway.evaluation.da_pmpc_runner import ROOT, run_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--da-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if not args.headless:
        raise SystemExit("S5A2 requires --headless")
    config_path = Path(args.da_config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    candidate_dir = output / "candidates"
    candidate_dir.mkdir(exist_ok=True)

    old_grid = output / "da_pmpc_grid.csv"
    old_selection = output / "da_pmpc_selection.json"
    if old_grid.exists() and not (output / "da_pmpc_grid_pre_s5a2.csv").exists():
        shutil.copyfile(old_grid, output / "da_pmpc_grid_pre_s5a2.csv")
    if old_selection.exists() and not (output / "da_pmpc_selection_pre_s5a2.json").exists():
        shutil.copyfile(old_selection, output / "da_pmpc_selection_pre_s5a2.json")

    lqr_paths = {scene: ROOT / "artifacts/s4/runs" / scene / "run.csv"
                 for scene in config["development_scenarios"]}
    wind_paths = {
        "approach_stop": ROOT / "artifacts/s4/inputs/calm.csv",
        "crosswind_hover": ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv",
    }
    rows = []
    index = 0
    for tip_weight in config["tip_weight_candidates"]:
        for residual_weight in config["residual_weight_candidates"]:
            index += 1
            scene_results = {}
            failure_reasons = []
            for scene in config["development_scenarios"]:
                local = dict(config)
                local["selected_tip_weight"] = float(tip_weight)
                local["selected_residual_weight"] = float(residual_weight)
                path = candidate_dir / f"candidate_{index:02d}_{scene}.csv"
                run_scene(
                    args.model_config, local, scene, wind_paths[scene],
                    ROOT / "artifacts/s2/references" / f"{scene}.csv", path, mode="full",
                )
                result = gate_scene(path, lqr_paths[scene], scene, local)
                scene_results[scene] = result
                failure_reasons.extend(f"{scene}:{reason}" for reason in result["failure_reasons"])

            approach = scene_results["approach_stop"]
            crosswind = scene_results["crosswind_hover"]
            tip_ratios = [
                approach["metric"]["tip_rms_m"] / approach["lqr_metric"]["tip_rms_m"],
                crosswind["metric"]["tip_rms_m"] / crosswind["lqr_metric"]["tip_rms_m"],
            ]
            residual_activity = [
                approach["metric"].get("control_rate_proxy", 0.0),
                crosswind["metric"].get("control_rate_proxy", 0.0),
            ]
            safe = not failure_reasons
            row = {
                "candidate_index": index,
                "tip_weight": float(tip_weight),
                "residual_weight": float(residual_weight),
                "safe": safe,
                "score": float(np.mean(tip_ratios) + 0.001 * np.mean(residual_activity)) if safe else float("inf"),
                "approach_x_rmse": approach["metric"]["x_position_rmse_m"],
                "approach_lqr_x_rmse": approach["lqr_metric"]["x_position_rmse_m"],
                "approach_tip_rms": approach["metric"]["tip_rms_m"],
                "approach_lqr_tip_rms": approach["lqr_metric"]["tip_rms_m"],
                "crosswind_x_rmse": crosswind["metric"]["x_position_rmse_m"],
                "crosswind_lqr_x_rmse": crosswind["lqr_metric"]["x_position_rmse_m"],
                "crosswind_tip_rms": crosswind["metric"]["tip_rms_m"],
                "crosswind_lqr_tip_rms": crosswind["lqr_metric"]["tip_rms_m"],
                "approach_position_pass": not any(reason.endswith(":approach_position") for reason in failure_reasons),
                "approach_tip_pass": not any(reason.endswith(":approach_tip") for reason in failure_reasons),
                "crosswind_position_pass": not any(reason.endswith(":crosswind_position") for reason in failure_reasons),
                "crosswind_tip_pass": not any(reason.endswith(":crosswind_tip") for reason in failure_reasons),
                "dynamics_safe": not any(any(token in reason for token in ("finite", "anchor", "uav_z", "tip_z", "joint", "roll", "pitch")) for reason in failure_reasons),
                "input_safe": not any(any(token in reason for token in ("ax_limit", "ax_slew")) for reason in failure_reasons),
                "actuator_safe": not any(any(token in reason for token in ("thrust", "_Nm", "rotor")) for reason in failure_reasons),
                "failure_reasons": ";".join(failure_reasons),
                "approach_final_d_hat": approach["metric"].get("final_d_hat", 0.0),
                "approach_max_abs_d_hat": approach["metric"].get("max_abs_d_hat", 0.0),
                "crosswind_final_d_hat": crosswind["metric"].get("final_d_hat", 0.0),
                "crosswind_max_abs_d_hat": crosswind["metric"].get("max_abs_d_hat", 0.0),
                "mean_abs_raw_ax": float(np.mean([approach["metric"].get("mean_abs_raw_ax", 0.0), crosswind["metric"].get("mean_abs_raw_ax", 0.0)])),
                "mean_abs_limited_ax": float(np.mean([approach["metric"].get("mean_abs_limited_ax", 0.0), crosswind["metric"].get("mean_abs_limited_ax", 0.0)])),
                "qp_limiter_mismatch_max": float(max(approach["metric"].get("qp_limiter_mismatch_max", 0.0), crosswind["metric"].get("qp_limiter_mismatch_max", 0.0))),
                "development_scenarios": "approach_stop,crosswind_hover",
                "gust_used_for_selection": False,
            }
            rows.append(row)
            for scene in config["development_scenarios"]:
                (candidate_dir / f"candidate_{index:02d}_{scene}.csv").unlink(missing_ok=True)

    grid_path = output / "da_pmpc_grid.csv"
    with grid_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    safe_rows = [row for row in rows if row["safe"]]
    selection = {
        "grid_size": len(rows),
        "safe_candidate_count": len(safe_rows),
        "development_scenarios": config["development_scenarios"],
        "gust_used_for_selection": False,
        "selected": None,
    }
    if safe_rows:
        selected = min(safe_rows, key=lambda row: row["score"])
        selection.update({
            "result": "selected",
            "selected": selected,
            "selected_tip_weight": selected["tip_weight"],
            "selected_residual_weight": selected["residual_weight"],
        })
        config["selected_tip_weight"] = selected["tip_weight"]
        config["selected_residual_weight"] = selected["residual_weight"]
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
        exit_code = 0
    else:
        selection["result"] = "BLOCKED_LS_DA_PMPC_PILOT"
        exit_code = 2
    (output / "da_pmpc_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
