"""Run the pre-registered SEP-NMPC development grid in WSL."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from uav_sway.paper_baseline.sep_nmpc_controller import FormalSEPController
from uav_sway.paper_baseline.sep_nmpc_model import PlanarParameters
from uav_sway.paper_baseline.sep_nmpc_ocp import SEPTrackingConfig, frozen_parameter_grid
from uav_sway.paper_baseline.sep_nmpc_runtime import run_sep_scene

from .sep_nmpc_gate import evaluate_candidate, load_metrics, passivity_summary, write_grid


ROOT = Path(__file__).resolve().parents[3]
SCENES = ("approach_stop", "crosswind_hover")
SETTLING = {"approach_stop": 6.0, "crosswind_hover": 4.0}


def run_development_grid(output: str | Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    equivalent = json.loads((ROOT / "artifacts/s5d1/equivalent_parameters.json").read_text(encoding="utf-8"))
    parameters = PlanarParameters(equivalent["m_Q"], equivalent["m_L"], equivalent["l_eq"], equivalent["g"])
    configs = frozen_parameter_grid()
    lqr_frozen = {
        "approach_stop": {"x_position_rmse_m": 0.1053850612, "tip_rms_m": 0.1688590761},
        "crosswind_hover": {"x_position_rmse_m": 0.0919234887, "tip_rms_m": 0.0483626135},
    }
    rows = []
    passivity_rows = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="s5d2_acados_") as generated_root:
        for index, config in enumerate(configs, start=1):
            candidate = f"candidate_{index:02d}"
            controller = FormalSEPController(parameters, config, Path(generated_root) / candidate)
            scene_metrics = {}
            prediction_slacks: list[float] = []
            prediction_residuals: list[float] = []
            for scene in SCENES:
                wind = ROOT / "artifacts/s5d2/parity/inputs/calm.csv" if scene == "approach_stop" else ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv"
                if scene == "approach_stop" and not wind.exists():
                    raise FileNotFoundError(wind)
                reference = ROOT / "artifacts/s2/references" / f"{scene}.csv"
                run_path = output / "runs" / candidate / scene / "run.csv"
                try:
                    run_sep_scene(ROOT / "configs/model_5link.yaml", ROOT / "configs/sep_nmpc_adapted.yaml", scene, wind, reference, run_path, controller)
                    scene_metrics[scene] = load_metrics(run_path, SETTLING[scene])
                except Exception as exc:
                    failure = {"candidate": candidate, "scene": scene, "error": repr(exc)}
                    failures.append(failure)
                    (output / "runs" / candidate / scene).mkdir(parents=True, exist_ok=True)
                    (output / "runs" / candidate / scene / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8", newline="\n")
                    scene_metrics[scene] = {
                        "x_position_rmse_m": float("inf"), "tip_rms_m": float("inf"), "saturation_rate": 1.0,
                        "safe": False, "safety_reasons": ["acados_solver_failure"], "solver_failure_count": 1,
                    }
                prediction_slacks.extend(controller.prediction_slacks_history)
                prediction_residuals.extend(controller.prediction_residuals_history)
            passivity_path = next((output / "runs" / candidate / scene / "run.csv" for scene in SCENES if (output / "runs" / candidate / scene / "run.csv").exists()), None)
            if passivity_path is not None:
                passivity = passivity_summary(passivity_path, prediction_slacks, prediction_residuals)
            else:
                slacks = np.asarray(prediction_slacks if prediction_slacks else [5.0], dtype=float)
                residuals = np.asarray(prediction_residuals if prediction_residuals else [float("inf")], dtype=float)
                passivity = {"passivity_residual_max": float(np.max(residuals)), "slack_max": float(np.max(slacks)), "slack_mean": float(np.mean(slacks)), "slack_rms": float(np.sqrt(np.mean(slacks**2))), "slack_saturation_rate": float(np.mean(slacks >= 5.0 - 1e-8)), "prediction_node_count": int(slacks.size)}
            gate = evaluate_candidate(scene_metrics, lqr_frozen, passivity)
            for scene in SCENES:
                metrics = scene_metrics[scene]
                (output / "runs" / candidate / scene / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
            passivity_rows.append({"candidate": candidate, **passivity})
            rows.append({"candidate": candidate, "index": index, "K_e": config.k_e, "rho": config.rho, "epsilon": config.epsilon, **gate})
    write_grid(output / "grid/sep_grid.csv", rows)
    (output / "passivity_summary.json").write_text(json.dumps(passivity_rows, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "failure.log").write_text("\n".join(json.dumps(item, sort_keys=True) for item in failures) + ("\n" if failures else ""), encoding="utf-8", newline="\n")
    usable = [row for row in rows if row["candidate_usable"]]
    selected = min(usable, key=lambda row: row["selection_score"]) if usable else None
    selection = {
        "result": "PASS_DEVELOPMENT_SELECTION" if selected else "BLOCKED_NO_USABLE_SEP_BASELINE",
        "selected": selected,
        "grid_size": len(rows),
        "usable_count": len(usable),
        "selection_uses_ls_pmpc": False,
        "selection_normalization": "frozen LQR development metrics only",
        "holdout_used": False,
        "gust_used": False,
        "random_holdout_used": False,
    }
    (output / "grid/selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")
    return selection
