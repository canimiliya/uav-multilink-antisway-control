"""Run exactly the frozen S5D2 SEP-NMPC development grid."""

from __future__ import annotations

import argparse
import json

from uav_sway.evaluation.sep_nmpc_runner import run_development_grid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/s5d2")
    args = parser.parse_args()
    result = run_development_grid(args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["selected"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
