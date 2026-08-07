"""Run the selected MPPI controller on all three frozen S5 scenarios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics
from uav_sway.evaluation.mppi_gate import raw_mppi_gate
from uav_sway.evaluation.mppi_runner import run_mppi_scenario
from uav_sway.evaluation.mppi_schema import mppi_schema_description
from uav_sway.evaluation.controlled_runner import ensure_calm_wind
from uav_sway.disturbances.wind_io import read_wind_csv

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def write_protocol_artifacts(output: Path, cfg: dict) -> None:
    runtime = ROOT / "artifacts/s3/runtime/model_5link_controlled.xml"
    source = ROOT / "artifacts/s1/generated/model_5link.xml"
    q_path, r_path, k_path = (ROOT / "artifacts/s4/lqr" / name for name in ("Q.npy", "R.npy", "K.npy"))
    if sha256(runtime) != "19105873c0fcc891ebb85efe6c20c378d5b77b6bf9003559e43ae47ca03d153d":
        raise SystemExit("BLOCKED_DEPENDENCY_DRIFT: runtime model SHA mismatch")
    expected = {"runtime_model_sha256": sha256(runtime), "source_model_sha256": sha256(source),
                "s4_q_sha256": sha256(q_path), "s4_r_sha256": sha256(r_path), "s4_k_sha256": sha256(k_path),
                "inner_loop_source_sha256": sha256(ROOT / "src/uav_sway/control/geometric_inner_loop.py"),
                "reduced_state_source_sha256": sha256(ROOT / "src/uav_sway/linearization/reduced_state.py"),
                "physics_dt_s": float(cfg["physics_dt_s"]), "inner_dt_s": float(cfg["inner_dt_s"]),
                "outer_dt_s": float(cfg["outer_dt_s"]), "ax_limits_m_s2": [float(cfg["ax_min_m_s2"]), float(cfg["ax_max_m_s2"])],
                "ax_slew_limit_m_s2_per_update": float(cfg["ax_slew_limit_m_s2_per_update"]),
                "s2_reference_sha256": {s: sha256(ROOT / "artifacts/s2/references" / f"{s}.csv") for s in ("approach_stop", "crosswind_hover", "gust_micro_adjust")},
                "s2_wind_sha256": {n: sha256(ROOT / "artifacts/s2/wind_bank" / n) for n in ("constant_crosswind.csv", "one_cosine_gust.csv")}}
    if expected["physics_dt_s"] != 0.001 or expected["inner_dt_s"] != 0.005 or expected["outer_dt_s"] != 0.05 or expected["ax_limits_m_s2"] != [-2.0, 2.0] or expected["ax_slew_limit_m_s2_per_update"] != 0.25:
        raise SystemExit("BLOCKED_DEPENDENCY_DRIFT: S5 timing or limiter drift")
    (output / "dependencies.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "controller_contract.json").write_text(json.dumps({"controller": "mppi", "optimized_input": "delta_a_x", "reference_preview": True, "external_wind_preview": False, "rollout_wind_x_m_s": 0.0, "horizon_steps": 12, "num_rollouts": 64, "iterations": 1, "anchor_active": False, "rotor_motors": "zero", "shared_inner_loop": True, "limiter": {"amplitude": [-2.0, 2.0], "slew_per_outer_update": 0.25}}, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "algorithm_audit.json").write_text(json.dumps({"rollout_engine": "independent mujoco.MjData nonlinear physics", "linear_AB_used": False, "simplified_pendulum_used": False, "future_external_wind_used": False, "future_reference_used": True, "integral_state": False, "controller_mix": None, "warm_start": "shift_old_sequence_and_zero_tail", "sampler": "numpy.random.Generator(PCG64)"}, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "reproducibility.json").write_text(json.dumps({"formal_seed": int(cfg["formal_seed"]), "tuning_seed": int(cfg["tuning_seed"]), "rng": "numpy.random.PCG64", "same_seed_policy": "same state/reference/nominal/noise produces identical control and state columns except solve_time_ms"}, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--model-config", required=True)
    parser.add_argument("--mppi-config", required=True); parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--headless", action="store_true"); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not args.headless: raise SystemExit("S5 runner requires --headless")
    cfg = yaml.safe_load(Path(args.mppi_config).read_text(encoding="utf-8")); output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_protocol_artifacts(output, cfg)
    if "selected_temperature" not in cfg or "selected_noise_sigma" not in cfg:
        raise SystemExit("BLOCKED_NO_SAFE_MPPI: no selected candidate in configs/mppi.yaml")
    temperature = float(cfg["selected_temperature"])
    sigma = float(cfg["selected_noise_sigma"])
    calm = output / "inputs/calm.csv"
    ensure_calm_wind(calm, read_wind_csv(ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv")["time"])
    winds = {"approach_stop": calm, "crosswind_hover": ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv",
             "gust_micro_adjust": ROOT / "artifacts/s2/wind_bank/one_cosine_gust.csv"}
    paths = {}
    for scenario in args.scenarios:
        path = output / "runs" / scenario / "run.csv"; paths[scenario] = path
        metric = run_mppi_scenario(args.model_config, cfg, scenario, winds[scenario],
                                   ROOT / "artifacts/s2/references" / f"{scenario}.csv", path,
                                   ROOT, seed=int(cfg["formal_seed"]), temperature=temperature,
                                   noise_sigma=sigma)
        (path.parent / "metrics.json").write_text(json.dumps(metric, indent=2) + "\n", encoding="utf-8", newline="\n")
    lqr_paths = {s: ROOT / "artifacts/s4/runs" / s / "run.csv" for s in args.scenarios}
    pid_paths = {s: ROOT / "artifacts/s3/runs" / s / "run.csv" for s in args.scenarios}
    gate = raw_mppi_gate(paths, lqr_paths, cfg, pid_paths); (output / "raw_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8", newline="\n")
    summary = []
    comparison = {"source": "independent_raw_csv_recomputation", "scenarios": {}}
    for scenario in args.scenarios:
        m = compute_controlled_metrics(paths[scenario], cfg["settling_start_s"][scenario])
        l = compute_controlled_metrics(lqr_paths[scenario], cfg["settling_start_s"][scenario])
        p = compute_controlled_metrics(pid_paths[scenario], cfg["settling_start_s"][scenario])
        improvement = 100.0 * (l["tip_rms_m"] - m["tip_rms_m"]) / l["tip_rms_m"]
        summary.append({"scenario": scenario, "x_rmse_m": m["x_position_rmse_m"], "lqr_x_rmse_m": l["x_position_rmse_m"], "pid_x_rmse_m": p["x_position_rmse_m"], "tip_rms_m": m["tip_rms_m"], "lqr_tip_rms_m": l["tip_rms_m"], "pid_tip_rms_m": p["tip_rms_m"], "tip_improvement_vs_lqr_percent": improvement, "solve_time_mean_ms": m["solve_time_mean_ms"], "solve_time_p95_ms": m["solve_time_p95_ms"], "solve_time_max_ms": m["solve_time_max_ms"], "control_rate_proxy": m["control_rate_proxy"]})
        comparison["scenarios"][scenario] = {"mppi_tip_rms_m": m["tip_rms_m"], "lqr_tip_rms_m": l["tip_rms_m"], "pid_tip_rms_m": p["tip_rms_m"], "mppi_tip_max_m": m["tip_max_abs_m"], "lqr_tip_max_m": l["tip_max_abs_m"], "pid_tip_max_m": p["tip_max_abs_m"], "mppi_x_rmse_m": m["x_position_rmse_m"], "lqr_x_rmse_m": l["x_position_rmse_m"], "pid_x_rmse_m": p["x_position_rmse_m"], "tip_improvement_vs_lqr_percent": improvement, "mppi_control_rate_proxy": m["control_rate_proxy"], "lqr_control_rate_proxy": l["control_rate_proxy"], "pid_control_rate_proxy": p["control_rate_proxy"]}
    (output / "mppi_summary.csv").parent.mkdir(parents=True, exist_ok=True)
    with (output / "mppi_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (output / "mppi_summary.json").write_text(json.dumps({"controller": "mppi", "temperature": temperature, "noise_sigma": sigma, "scenarios": summary, "raw_gate": gate, "schema": mppi_schema_description(5)}, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "comparison").mkdir(parents=True, exist_ok=True)
    (output / "comparison/mppi_vs_lqr_pid.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8", newline="\n")
    all_times = []
    per_scene_time = {}
    for scenario, path in paths.items():
        import pandas as pd
        values = pd.read_csv(path)["solve_time_ms"].to_numpy(dtype=float)
        all_times.extend(values.tolist())
        per_scene_time[scenario] = {"mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)), "max_ms": float(np.max(values))}
    (output / "computation_time.json").write_text(json.dumps({"source": "raw_run_csv_solve_time_ms", "all_scenes": {"mean_ms": float(np.mean(all_times)), "median_ms": float(np.median(all_times)), "p95_ms": float(np.percentile(all_times, 95)), "max_ms": float(np.max(all_times))}, "scenarios": per_scene_time}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0 if gate["pass"] else 2


if __name__ == "__main__": raise SystemExit(main())
