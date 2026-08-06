"""CLI for the wind-free, uncontrolled passive validation run."""

import argparse
import json
from pathlib import Path

from uav_sway.models.build_planar_chain import build_planar_chain_model
from uav_sway.models.model_config import load_model_config
from uav_sway.models.passive_sim import simulate_passive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--initial-angle-deg", type=float, default=0.0)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_model_config(args.config)
    output_dir = Path(args.output_dir)
    generated = output_dir / "generated"
    xml_path = generated / f"model_{config.n_links}link.xml"
    build_planar_chain_model(args.config, xml_path)
    csv_path = output_dir / f"passive_decay_{config.n_links}link.csv"
    render_path = output_dir / f"model_{config.n_links}link.png" if args.render else None
    result = simulate_passive(args.config, args.initial_angle_deg, args.duration, csv_path, render_path, model_path=xml_path)
    summary_path = output_dir / "passive_run_summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
