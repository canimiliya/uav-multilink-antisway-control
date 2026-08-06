"""Run the three S2 headless wind smoke cases and compute raw-CSV metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_sway.evaluation.batch_runner import run_wind_validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--s2-dir", default="artifacts/s2")
    args = parser.parse_args()
    if not args.headless:
        raise SystemExit("S2 smoke runner requires --headless; no renderer is created")
    s2_dir = Path(args.s2_dir)
    output_dir = Path(args.output_dir)
    scenario_by_profile = {"constant_crosswind": "crosswind_hover", "one_cosine_gust": "gust_micro_adjust", "low_frequency_random": "approach_stop"}
    wind_filename = {"constant_crosswind": "constant_crosswind.csv", "one_cosine_gust": "one_cosine_gust.csv", "low_frequency_random": f"random_seed_{args.random_seed:03d}.csv"}
    results = {}
    for profile in args.profiles:
        scenario = scenario_by_profile[profile]
        case = profile if profile != "low_frequency_random" else f"random_seed_{args.random_seed:03d}"
        case_dir = output_dir / case
        run_path = case_dir / "run.csv"
        metrics = run_wind_validation(
            args.model_config, scenario, s2_dir / "wind_bank" / wind_filename[profile],
            s2_dir / "references" / f"{scenario}.csv", run_path,
            "configs/aerodynamics.yaml", "configs/scenarios.yaml",
        )
        metrics_path = case_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
        results[case] = metrics
        print(json.dumps({case: metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
