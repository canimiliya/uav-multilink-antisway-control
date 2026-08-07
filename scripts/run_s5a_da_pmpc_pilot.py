"""Run the two-scene S5A2 pilot and its required ablation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics
from uav_sway.evaluation.da_pmpc_gate import raw_da_pmpc_gate
from uav_sway.evaluation.da_pmpc_runner import ROOT, run_scene


SCENES = ("approach_stop", "crosswind_hover")
WIND_PATHS = {
    "approach_stop": ROOT / "artifacts/s4/inputs/calm.csv",
    "crosswind_hover": ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv",
}


def _run_mode(model_config, config, output, mode):
    paths = {}
    metrics = {}
    for scene in SCENES:
        path = output / "runs" / scene / "run.csv" if mode == "full" else output / "ablation" / mode / scene / "run.csv"
        paths[scene] = path
        metrics[scene] = run_scene(
            model_config, config, scene, WIND_PATHS[scene],
            ROOT / "artifacts/s2/references" / f"{scene}.csv", path, mode=mode,
        )
        (path.parent / "metrics.json").write_text(
            json.dumps(metrics[scene], indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    return paths, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--da-config", required=True)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if not args.headless:
        raise SystemExit("S5A2 requires --headless")
    if tuple(args.scenarios) != SCENES:
        raise SystemExit("S5A2 runs only approach_stop and crosswind_hover")
    config = yaml.safe_load(Path(args.da_config).read_text(encoding="utf-8"))
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    selection_path = output / "tuning" / "da_pmpc_selection.json"
    if not selection_path.exists():
        selection_path = ROOT / "artifacts/s5a/tuning/da_pmpc_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    selected = selection.get("selected")
    if selected is not None:
        config["selected_tip_weight"] = float(selected["tip_weight"])
        config["selected_residual_weight"] = float(selected["residual_weight"])
    else:
        # Diagnostic-only center candidate.  It is never reported as selected.
        config["selected_tip_weight"] = float(config["tip_weight_candidates"][1])
        config["selected_residual_weight"] = float(config["residual_weight_candidates"][1])

    full_paths, full_metrics = _run_mode(args.model_config, config, output, "full")
    lqr_paths, lqr_metrics = _run_mode(args.model_config, config, output, "lqr")
    observer_paths, observer_metrics = _run_mode(args.model_config, config, output, "lqr_observer")
    preview_paths, preview_metrics = _run_mode(args.model_config, config, output, "preview")
    gate = raw_da_pmpc_gate(
        full_paths,
        {scene: ROOT / "artifacts/s4/runs" / scene / "run.csv" for scene in SCENES},
        config,
    )
    gate["selected_candidate_present"] = selected is not None
    gate["pass"] = bool(gate["pass"] and selected is not None)
    gate["status"] = "PASS" if gate["pass"] else "BLOCKED_LS_DA_PMPC_PILOT"
    gate["source"] = "independent_raw_csv_recomputation"
    (output / "raw_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    ablation = {}
    for scene in SCENES:
        ablation[scene] = {
            "lqr": {"x_rmse": lqr_metrics[scene]["x_position_rmse_m"], "tip_rms": lqr_metrics[scene]["tip_rms_m"]},
            "lqr_plus_observer": {"x_rmse": observer_metrics[scene]["x_position_rmse_m"], "tip_rms": observer_metrics[scene]["tip_rms_m"]},
            "lqr_plus_preview": {"x_rmse": preview_metrics[scene]["x_position_rmse_m"], "tip_rms": preview_metrics[scene]["tip_rms_m"]},
            "full_ls_da_pmpc": {"x_rmse": full_metrics[scene]["x_position_rmse_m"], "tip_rms": full_metrics[scene]["tip_rms_m"]},
        }
    (output / "ablation.json").write_text(
        json.dumps({"scenarios": ablation, "gust_used": False}, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )

    parity = {}
    for scene in SCENES:
        lqr_csv = np.genfromtxt(ROOT / "artifacts/s4/runs" / scene / "run.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
        parity_csv = np.genfromtxt(lqr_paths[scene], delimiter=",", names=True, dtype=None, encoding="utf-8")
        parity[scene] = {
            "first_command_error": float(abs(float(parity_csv["ax_cmd_raw"][0]) - float(lqr_csv["ax_cmd_raw"][0]))),
            "x_rmse_error": abs(lqr_metrics[scene]["x_position_rmse_m"] - compute_controlled_metrics(ROOT / "artifacts/s4/runs" / scene / "run.csv", float(config["settling_start_s"][scene]))["x_position_rmse_m"]),
            "tip_rms_error": abs(lqr_metrics[scene]["tip_rms_m"] - compute_controlled_metrics(ROOT / "artifacts/s4/runs" / scene / "run.csv", float(config["settling_start_s"][scene]))["tip_rms_m"]),
        }
    (output / "lqr_parity.json").write_text(
        json.dumps({"scenarios": parity, "max_first_command_error": max(x["first_command_error"] for x in parity.values()), "max_x_rmse_error": max(x["x_rmse_error"] for x in parity.values()), "max_tip_rms_error": max(x["tip_rms_error"] for x in parity.values())}, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )

    summary = {
        "controller": "lqr_stabilized_da_pmpc",
        "selected_candidate": selected,
        "pilot_gate": gate,
        "ablation": ablation,
        "lqr_parity": parity,
        "development_scenarios": list(SCENES),
        "gust_used": False,
        "random_holdout_used": False,
    }
    (output / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    raise SystemExit(0 if gate["pass"] else 2)


if __name__ == "__main__":
    main()
