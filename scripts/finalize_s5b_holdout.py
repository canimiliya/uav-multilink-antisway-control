"""Finalize S5B evidence from the already retained raw holdout files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_s5b_holdout import (
    MODE_LABELS, MODES, ROOT, SCENES, _direct_metrics, bootstrap_mean_ci,
    make_figures, make_statistics, paired_row, safety_check, write_json, write_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/s5b")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    random_rows = []
    for path in sorted((output / "random").glob("*/ *".replace(" ", ""))):
        pass
    for path in sorted((output / "random").glob("*/ */ */run.csv.gz".replace(" ", ""))):
        mode_label, scene, seed_label = path.parts[-4:-1]
        seed = int(seed_label.split("_")[-1])
        metrics = _direct_metrics(path, {"approach_stop": 6.0, "crosswind_hover": 4.0, "gust_micro_adjust": 5.0}[scene])
        safe, reasons = safety_check(metrics)
        metrics.update({"seed": seed, "scenario": scene, "mode": mode_label, "safe": safe, "failure_reasons": reasons})
        random_rows.append(metrics)
    if len(random_rows) != 180:
        raise RuntimeError(f"expected 180 random raw runs, found {len(random_rows)}")
    paired = []
    for seed in range(20):
        for scene in SCENES:
            lqr = next(r for r in random_rows if r["seed"] == seed and r["scenario"] == scene and r["mode"] == "LQR")
            primary = next(r for r in random_rows if r["seed"] == seed and r["scenario"] == scene and r["mode"] == "LS-PMPC")
            paired.append(paired_row(lqr, primary))
    write_rows(output / "random/per_seed_metrics.csv", random_rows)
    write_rows(output / "random/primary_vs_lqr_paired.csv", paired)
    summary, bootstrap, _ = make_statistics(random_rows, paired, output)
    gust_rows = []
    for mode in MODES:
        path = output / "gust/runs/gust_micro_adjust" / MODE_LABELS[mode] / "seed_gust/run.csv"
        metrics = _direct_metrics(path, 5.0)
        safe, reasons = safety_check(metrics)
        metrics.update({"seed": "gust", "scenario": "gust_micro_adjust", "mode": MODE_LABELS[mode], "safe": safe, "failure_reasons": reasons, "source_csv": str(path)})
        gust_rows.append(metrics)
    write_rows(output / "gust/metrics.csv", gust_rows)
    primary_rows = [r for r in random_rows if r["mode"] == "LS-PMPC"]
    safety_failures = [r for r in primary_rows if not r["safe"]]
    scene_position = {}
    scene_tip = {}
    for scene in SCENES:
        p = [r for r in paired if r["scenario"] == scene]
        scene_position[scene] = float(np.mean([r["primary_x_rmse"] for r in p]) / np.mean([r["lqr_x_rmse"] for r in p]))
        scene_tip[scene] = float(np.mean([r["tip_improvement_percent"] for r in p]))
    overall_position = float(np.mean([r["primary_x_rmse"] for r in paired]) / np.mean([r["lqr_x_rmse"] for r in paired]))
    overall_tip = float(np.mean([r["tip_improvement_percent"] for r in paired]))
    gust_lookup = {r["mode"]: r for r in gust_rows}
    lqr_gust, primary_gust = gust_lookup["LQR"], gust_lookup["LS-PMPC"]
    tip_ci = bootstrap["scenarios"]["overall"]["tip_improvement_percent_mean_ci95"]
    solve_p95 = float(np.percentile([r["solve_time_p95_ms"] for r in primary_rows], 95))
    conditions = {
        "safety_zero_failures": len(safety_failures) == 0,
        "position_each_scene_le_1p05": all(v <= 1.05 for v in scene_position.values()),
        "overall_position_not_worse": overall_position <= 1.0,
        "at_least_two_scene_tip_mean_ge_10": sum(v >= 10.0 for v in scene_tip.values()) >= 2,
        "overall_tip_mean_ge_10": overall_tip >= 10.0,
        "bootstrap_tip_ci_lower_gt_0": tip_ci[0] > 0.0,
        "gust_position_le_1p05": primary_gust["x_rmse_m"] <= 1.05 * lqr_gust["x_rmse_m"],
        "gust_tip_le_0p95": primary_gust["tip_rms_m"] <= 0.95 * lqr_gust["tip_rms_m"],
        "solve_time_p95_lt_50_ms": solve_p95 < 50.0,
    }
    gate = {
        "source": "independent_raw_csv_recomputation", "pass": bool(all(conditions.values())),
        "primary_safety_failures": len(safety_failures), "scene_position_ratios": scene_position,
        "scene_tip_improvement_percent": scene_tip, "overall_position_ratio": overall_position,
        "overall_tip_improvement_percent": overall_tip,
        "gust_position_ratio": primary_gust["x_rmse_m"] / lqr_gust["x_rmse_m"],
        "gust_tip_ratio": primary_gust["tip_rms_m"] / lqr_gust["tip_rms_m"],
        "bootstrap_tip_ci95": tip_ci, "primary_solve_time_p95_ms": solve_p95,
        "primary_run_count": len(primary_rows), "random_seed_count": 20, "conditions": conditions,
    }
    gate["status"] = "PASS" if gate["pass"] else ("BLOCKED_SAFETY" if safety_failures else "BLOCKED_HOLDOUT_GENERALIZATION")
    write_json(output / "raw_gate.json", gate)
    observer_case_rows = []
    observer_rows = []
    for scene in SCENES:
        p = [r for r in random_rows if r["scenario"] == scene and r["mode"] == "LS-PMPC"]
        d = [r for r in random_rows if r["scenario"] == scene and r["mode"] == "LS-DA-PMPC"]
        for seed in range(20):
            primary = next(r for r in p if r["seed"] == seed)
            secondary = next(r for r in d if r["seed"] == seed)
            observer_case_rows.append({"scenario": scene, "seed": seed, "primary_tip_rms": primary["tip_rms_m"], "secondary_tip_rms": secondary["tip_rms_m"], "secondary_minus_primary_tip_rms": secondary["tip_rms_m"] - primary["tip_rms_m"], "primary_x_rmse": primary["x_rmse_m"], "secondary_x_rmse": secondary["x_rmse_m"]})
        delta = float(np.mean([r["tip_rms_m"] for r in d]) - np.mean([r["tip_rms_m"] for r in p]))
        observer_rows.append({"scenario": scene, "primary_tip_rms_mean": float(np.mean([r["tip_rms_m"] for r in p])), "secondary_tip_rms_mean": float(np.mean([r["tip_rms_m"] for r in d])), "secondary_minus_primary_tip_rms": delta, "conclusion": "helps" if delta < -1e-12 else ("hurts" if delta > 1e-12 else "neutral")})
    observer_total = sum(r["secondary_minus_primary_tip_rms"] for r in observer_rows)
    write_rows(output / "observer_ablation.csv", observer_case_rows)
    write_json(output / "statistics/observer_ablation.json", {"source": "paired_random_holdout", "scenarios": observer_rows, "overall_conclusion": "helps" if observer_total < -1e-12 else ("hurts" if observer_total > 1e-12 else "neutral"), "primary_never_reselected": True})
    make_figures(output, gust_rows, random_rows, paired)
    write_json(output / "final_status.json", {"result": gate["status"], "pass": bool(gate["pass"]), "primary_method": "LS-PMPC", "method_frozen_before_holdout": True, "primary_dev_freeze_gate": json.loads((output / "primary_dev_freeze_gate.json").read_text(encoding="utf-8")), "raw_gate": gate})
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
