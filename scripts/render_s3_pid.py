"""Render the three raw S3 PID runs into one diagnostic figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_sway.evaluation.controlled_metrics import load_controlled_csv


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input-dir", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for scenario in ("approach_stop", "crosswind_hover", "gust_micro_adjust"):
        _, v = load_controlled_csv(Path(args.input_dir) / scenario / "run.csv")
        t = v["time"]
        axes[0].plot(t, v["x_ref"], "--", label=f"{scenario} x_ref")
        axes[0].plot(t, v["uav_x"], label=f"{scenario} uav_x")
        axes[1].plot(t, v["tip_displacement"], label=scenario)
        axes[2].plot(t, v["ax_cmd_limited"], label=scenario)
        axes[3].plot(t, v["roll_rad"], label=f"{scenario} roll")
        axes[3].plot(t, v["pitch_rad"], ":", label=f"{scenario} pitch")
        axes[2].plot(t, v["wind_x"], "--", alpha=0.55, label=f"{scenario} wind_x")
    axes[0].set_ylabel("x (m)"); axes[1].set_ylabel("tip displacement (m)"); axes[2].set_ylabel("ax / wind x"); axes[3].set_ylabel("attitude (rad)"); axes[3].set_xlabel("time (s)")
    for ax in axes: ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=8)
    fig.suptitle("S3 free-flight position PID: three frozen scenarios")
    fig.tight_layout(); Path(args.output).parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.output, dpi=160); plt.close(fig); return 0


if __name__ == "__main__":
    raise SystemExit(main())
