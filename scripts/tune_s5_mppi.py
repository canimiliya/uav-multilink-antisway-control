"""Run the frozen six-point MPPI tuning grid on the two development scenes."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics
from uav_sway.evaluation.mppi_runner import run_mppi_scenario
from uav_sway.mppi.cost import mppi_candidate_score

ROOT = Path(__file__).resolve().parents[1]


def evaluate_candidate(index, temperature, sigma, model_config, cfg, output, lqr_paths, winds):
    candidate_metrics = {}
    safe = True
    for scene in ("approach_stop", "crosswind_hover"):
        scratch = output / f"candidate_{index:02d}_{scene}.csv"
        metric = run_mppi_scenario(
            model_config, cfg, scene, winds[scene],
            ROOT / "artifacts/s2/references" / f"{scene}.csv", scratch,
            ROOT, seed=int(cfg["tuning_seed"]), temperature=temperature,
            noise_sigma=sigma,
        )
        candidate_metrics[scene] = metric
        baseline = compute_controlled_metrics(lqr_paths[scene], cfg["settling_start_s"][scene])
        safe = safe and bool(metric["finite_outputs"] and not metric["anchor_active_any"]
                             and metric["minimum_tip_height_m"] > 0.05
                             and metric["minimum_uav_height_m"] > 0.05
                             and metric["maximum_abs_pitch_rad"] < np.deg2rad(25.0)
                             and metric["maximum_abs_roll_rad"] < np.deg2rad(25.0)
                             and metric["x_position_rmse_m"] <= 1.10 * baseline["x_position_rmse_m"]
                             and metric["tip_rms_m"] <= (0.95 if scene == "approach_stop" else 1.10) * baseline["tip_rms_m"])
        try:
            scratch.unlink()
        except FileNotFoundError:
            pass
    lqr_metrics = {scene: compute_controlled_metrics(lqr_paths[scene], cfg["settling_start_s"][scene])
                   for scene in candidate_metrics}
    tip_ratios = [candidate_metrics[s]["tip_rms_m"] / lqr_metrics[s]["tip_rms_m"] for s in candidate_metrics]
    pos_ratios = [candidate_metrics[s]["x_position_rmse_m"] / lqr_metrics[s]["x_position_rmse_m"] for s in candidate_metrics]
    rate_ratios = [candidate_metrics[s]["control_rate_proxy"] / max(lqr_metrics[s]["control_rate_proxy"], 1e-9) for s in candidate_metrics]
    sat = [candidate_metrics[s]["saturation_rate"] for s in candidate_metrics]
    score = mppi_candidate_score(tip_ratios, pos_ratios, rate_ratios, sat) if safe else float("inf")
    return {"candidate_index": index, "temperature": temperature, "noise_sigma": sigma,
            "safe": safe, "score": score,
            "approach_x_rmse": candidate_metrics["approach_stop"]["x_position_rmse_m"],
            "crosswind_x_rmse": candidate_metrics["crosswind_hover"]["x_position_rmse_m"],
            "approach_tip_rms": candidate_metrics["approach_stop"]["tip_rms_m"],
            "crosswind_tip_rms": candidate_metrics["crosswind_hover"]["tip_rms_m"],
            "tuning_seed": int(cfg["tuning_seed"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--mppi-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if not args.headless:
        raise SystemExit("S5 tuning requires --headless")
    cfg = yaml.safe_load(Path(args.mppi_config).read_text(encoding="utf-8"))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    lqr_paths = {scene: ROOT / "artifacts/s4/runs" / scene / "run.csv"
                 for scene in ("approach_stop", "crosswind_hover")}
    winds = {"approach_stop": ROOT / "artifacts/s4/inputs/calm.csv",
             "crosswind_hover": ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv"}
    rows = []
    candidates = [(float(temp), float(sigma)) for temp in cfg["temperature_candidates"]
                  for sigma in cfg["noise_sigma_candidates"]]
    with ProcessPoolExecutor(max_workers=len(candidates)) as executor:
        futures = [executor.submit(evaluate_candidate, index, temperature, sigma,
                                   args.model_config, cfg, output, lqr_paths, winds)
                   for index, (temperature, sigma) in enumerate(candidates, start=1)]
        rows = [future.result() for future in futures]
    grid_path = output / "mppi_grid.csv"
    with grid_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    safe_rows = [row for row in rows if row["safe"]]
    if not safe_rows:
        selection = {"result": "BLOCKED_NO_SAFE_MPPI", "grid_size": len(rows), "selected": None}
        (output / "mppi_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")
        return 2
    selected = min(safe_rows, key=lambda row: row["score"])
    selection = {"result": "selected", "grid_size": len(rows), "selected_candidate_index": selected["candidate_index"],
                 "temperature": selected["temperature"], "noise_sigma": selected["noise_sigma"],
                 "score": selected["score"], "development_scenarios": ["approach_stop", "crosswind_hover"],
                 "gust_used_for_selection": False, "safe_candidate_count": len(safe_rows)}
    (output / "mppi_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")
    cfg_path = Path(args.mppi_config)
    cfg["selected_temperature"] = selected["temperature"]
    cfg["selected_noise_sigma"] = selected["noise_sigma"]
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
