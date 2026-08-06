"""Generate the frozen S2 wind bank, reference bank, and protocol metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from uav_sway.disturbances.wind_io import sha256_file, write_wind_csv
from uav_sway.disturbances.wind_profiles import GENERATOR_VERSION, generate_wind_profile, load_wind_config
from uav_sway.evaluation.metrics import control_rate_formula_audit
from uav_sway.evaluation.schema import schema_description
from uav_sway.scenarios.reference_profiles import generate_reference
from uav_sway.scenarios.scenario_config import load_scenario_config


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_reference(path: Path, reference: dict[str, np.ndarray]) -> None:
    columns = ["time", "x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref", "event", "control_tick"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for index in range(len(reference["time"])):
            time = float(reference["time"][index])
            writer.writerow([
                format(time, ".17g"), format(float(reference["x_ref"][index]), ".17g"),
                format(float(reference["vx_ref"][index]), ".17g"), format(float(reference["ax_ref"][index]), ".17g"),
                format(float(reference["y_ref"][index]), ".17g"), format(float(reference["z_ref"][index]), ".17g"),
                format(float(reference["yaw_ref"][index]), ".17g"), str(reference["event"][index]),
                str(int(round(time / 0.05))) if np.isclose(time / 0.05, round(time / 0.05), atol=1e-10) else "-1",
            ])


def _reference_continuity_audit(config: dict) -> dict:
    epsilon = 1e-7
    boundaries = [1.0, 2.0, 5.0, 6.0]
    maximum_x_jump = 0.0
    maximum_vx_jump = 0.0
    maximum_ax_jump = 0.0
    for boundary in boundaries:
        times = np.asarray([boundary - epsilon, boundary, boundary + epsilon], dtype=float)
        reference = generate_reference("approach_stop", times, config)
        maximum_x_jump = max(maximum_x_jump, float(np.max(np.abs(np.diff(reference["x_ref"])))), float(np.ptp(reference["x_ref"])))
        maximum_vx_jump = max(maximum_vx_jump, float(np.max(np.abs(np.diff(reference["vx_ref"])))), float(np.ptp(reference["vx_ref"])))
        maximum_ax_jump = max(maximum_ax_jump, float(np.max(np.abs(np.diff(reference["ax_ref"])))), float(np.ptp(reference["ax_ref"])))
    sampled_time = np.arange(2401, dtype=float) * 0.005
    sampled = generate_reference("approach_stop", sampled_time, config)
    jump_2 = abs(float(sampled["vx_ref"][400] - sampled["vx_ref"][399]))
    jump_6 = abs(float(sampled["vx_ref"][1200] - sampled["vx_ref"][1199]))
    return {
        "scenario": "approach_stop",
        "epsilon_s": epsilon,
        "boundaries_s": boundaries,
        "maximum_x_jump_m": maximum_x_jump,
        "maximum_vx_jump_m_s": maximum_vx_jump,
        "maximum_ax_jump_m_s2": maximum_ax_jump,
        "sampled_vx_jump_at_2s_m_s": jump_2,
        "sampled_vx_jump_at_6s_m_s": jump_6,
        "continuous_x": bool(maximum_x_jump < 1e-6),
        "continuous_vx": bool(maximum_vx_jump < 1e-5),
        "continuous_ax": bool(maximum_ax_jump < 1e-4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wind-config", required=True)
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    wind_dir = output / "wind_bank"
    ref_dir = output / "references"
    wind_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    wind_config_path = Path(args.wind_config)
    scenario_config_path = Path(args.config)
    wind_config = load_wind_config(wind_config_path)
    scenario_config = load_scenario_config(scenario_config_path)
    config_hash = sha256_file(wind_config_path)
    seeds_start, seeds_end = (int(value) for value in args.seeds.split("-", 1))
    wind_entries = []
    for profile, filename, seed in (("constant_crosswind", "constant_crosswind.csv", None), ("one_cosine_gust", "one_cosine_gust.csv", None)):
        series = generate_wind_profile(profile, wind_config, seed=seed, dt=float(scenario_config["wind_and_log_dt"]))
        path = wind_dir / filename
        write_wind_csv(path, series)
        wind_entries.append({"profile": profile, "seed": seed, "sample_count": len(series.time), "dt": series.dt, "duration": series.duration, "sha256": sha256_file(path), "config_sha256": config_hash, "path": path.as_posix(), "generator_version": GENERATOR_VERSION})
    for seed in range(seeds_start, seeds_end + 1):
        series = generate_wind_profile("low_frequency_random", wind_config, seed=seed, dt=float(scenario_config["wind_and_log_dt"]))
        path = wind_dir / f"random_seed_{seed:03d}.csv"
        write_wind_csv(path, series)
        wind_entries.append({"profile": series.profile, "seed": seed, "sample_count": len(series.time), "dt": series.dt, "duration": series.duration, "sha256": sha256_file(path), "config_sha256": config_hash, "path": path.as_posix(), "generator_version": GENERATOR_VERSION})
    (wind_dir / "manifest.json").write_text(json.dumps({"generator_version": GENERATOR_VERSION, "config_sha256": config_hash, "files": wind_entries}, indent=2) + "\n", encoding="utf-8", newline="\n")

    reference_entries = []
    time = np.arange(int(round(float(scenario_config["duration_s"]) / float(scenario_config["wind_and_log_dt"]))) + 1, dtype=float) * float(scenario_config["wind_and_log_dt"])
    for scenario in ("approach_stop", "crosswind_hover", "gust_micro_adjust"):
        reference = generate_reference(scenario, time, scenario_config)
        path = ref_dir / f"{scenario}.csv"
        _write_reference(path, reference)
        reference_entries.append({"scenario": scenario, "sample_count": len(time), "dt": float(scenario_config["wind_and_log_dt"]), "duration": float(scenario_config["duration_s"]), "sha256": sha256_file(path), "config_sha256": sha256_file(scenario_config_path), "path": path.as_posix()})
    (ref_dir / "manifest.json").write_text(json.dumps({"files": reference_entries}, indent=2) + "\n", encoding="utf-8", newline="\n")

    (output / "reference_continuity_audit.json").write_text(
        json.dumps(_reference_continuity_audit(scenario_config), indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    (output / "metrics_formula_audit.json").write_text(
        json.dumps(control_rate_formula_audit(), indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )

    protocol = {
        "physics_dt_s": float(scenario_config["physics_dt"]), "signal_dt_s": float(scenario_config["wind_and_log_dt"]),
        "future_control_dt_s": float(scenario_config["future_outer_control_dt"]), "duration_s": float(scenario_config["duration_s"]),
        "signal_sample_count": len(time), "wind_profiles": ["constant_crosswind", "one_cosine_gust", "low_frequency_random"],
        "scenarios": ["approach_stop", "crosswind_hover", "gust_micro_adjust"], "random_seeds": list(range(seeds_start, seeds_end + 1)),
        "controller_implemented": False, "wind_axis": "world_x", "headless_smoke_mode": "anchored_wind_validation",
        "wind_config_sha256": config_hash,
    }
    (output / "protocol_summary.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "schema.json").write_text(json.dumps(schema_description(5), indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "aerodynamic_assumptions.md").write_text("""# S2 aerodynamic assumptions\n\nThe first S2 protocol uses world-x quadratic drag proxies with air density 1.225 kg/m^3. The airframe dimensions are a conservative dimension-envelope assumption; link and cutter coefficients, capsule/box proxies, and the no-torque choice are simulation assumptions. These values are not official DJI wind-drag parameters. Wind is applied independently at each rigid body's center of mass using its own MuJoCo Jacobian velocity.\n\nS2 uses an anchored validation model and no controller. `ax_cmd_raw`, `ax_cmd_limited`, saturation, and solve-time fields are protocol placeholders, not controller results.\n""", encoding="utf-8", newline="\n")
    print(json.dumps(protocol, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
