"""Render compact S5 MPPI comparison figures from raw CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--pid-dir", required=True); parser.add_argument("--lqr-dir", required=True); parser.add_argument("--mppi-dir", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    scenes = ["approach_stop", "crosswind_hover", "gust_micro_adjust"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=False)
    for row, scene in enumerate(scenes):
        pid = pd.read_csv(Path(args.pid_dir) / scene / "run.csv"); lqr = pd.read_csv(Path(args.lqr_dir) / scene / "run.csv"); mppi = pd.read_csv(Path(args.mppi_dir) / scene / "run.csv")
        ax = axes[row, 0]; ax.plot(pid.time, pid.uav_x, label="PID", alpha=.7); ax.plot(lqr.time, lqr.uav_x, label="LQR", alpha=.7); ax.plot(mppi.time, mppi.uav_x, label="MPPI", linewidth=1.2); ax.plot(mppi.time, mppi.x_ref, "k--", label="reference"); ax.set_ylabel(f"{scene}\nx (m)"); ax.grid(alpha=.2)
        ax = axes[row, 1]; ax.plot(mppi.time, mppi.tip_displacement, label="tip displacement"); ax.plot(mppi.time, mppi.ax_cmd_limited, label="ax command"); ax.plot(mppi.time, mppi.wind_x, label="wind x"); ax.set_ylabel(scene); ax.grid(alpha=.2)
    axes[0, 0].legend(ncol=4, fontsize=8); axes[0, 1].legend(ncol=3, fontsize=8); axes[-1, 0].set_xlabel("time (s)"); axes[-1, 1].set_xlabel("time (s)"); fig.tight_layout(); fig.savefig(out / "mppi_three_scenarios.png", dpi=160); plt.close(fig)
    for scene, name in (("approach_stop", "pid_lqr_mppi_approach_stop.png"), ("gust_micro_adjust", "pid_lqr_mppi_gust.png")):
        fig, ax = plt.subplots(figsize=(10, 5));
        for label, directory in (("PID", args.pid_dir), ("LQR", args.lqr_dir), ("MPPI", args.mppi_dir)):
            df = pd.read_csv(Path(directory) / scene / "run.csv"); ax.plot(df.time, df.tip_displacement, label=label)
        ax.axhline(0.0, color="k", linewidth=.7); ax.set_title(f"{scene}: tip displacement"); ax.set_xlabel("time (s)"); ax.set_ylabel("m"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out / name, dpi=160); plt.close(fig)
    return 0


if __name__ == "__main__": raise SystemExit(main())
