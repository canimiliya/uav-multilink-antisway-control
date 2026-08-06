"""CLI for generating the frozen 4/5/6-link XML variants."""

import argparse
from pathlib import Path

from uav_sway.models.build_planar_chain import build_planar_chain_model
from uav_sway.models.model_config import load_model_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    for config_path in args.configs:
        config = load_model_config(config_path)
        output = output_dir / f"model_{config.n_links}link.xml"
        print(build_planar_chain_model(config_path, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
