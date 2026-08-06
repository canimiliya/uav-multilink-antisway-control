"""Render S4 LQR and PID comparison figures from raw CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream: rows = list(csv.DictReader(stream))
    numeric = {}
    for key in rows[0]:
        try: numeric[key] = np.asarray([float(row[key]) for row in rows], dtype=float)
        except ValueError: pass
    return numeric


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--pid-dir", required=True); parser.add_argument("--lqr-dir", required=True); parser.add_argument("--output-dir", required=True); args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); scenarios = ["approach_stop", "crosswind_hover", "gust_micro_adjust"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for scene in scenarios:
        p = _read(Path(args.pid_dir) / scene / "run.csv"); l = _read(Path(args.lqr_dir) / scene / "run.csv")
        axes[0].plot(l["time"], l["x_ref"], "k--", alpha=0.5); axes[0].plot(l["time"], l["uav_x"], label=f"LQR {scene}"); axes[0].plot(p["time"], p["uav_x"], ":", label=f"PID {scene}")
        axes[1].plot(l["time"], l["tip_displacement"], label=scene); axes[2].plot(l["time"], l["ax_cmd_limited"], label=scene); axes[3].plot(l["time"], l["pitch_rad"], label=scene)
    axes[0].set_ylabel("x (m)"); axes[1].set_ylabel("tip displacement (m)"); axes[2].set_ylabel("ax (m/s²)"); axes[3].set_ylabel("pitch (rad)"); axes[3].set_xlabel("time (s)")
    for ax in axes: ax.grid(True, alpha=0.25); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "lqr_three_scenarios.png", dpi=160); plt.close(fig)
    p = _read(Path(args.pid_dir) / "approach_stop/run.csv"); l = _read(Path(args.lqr_dir) / "approach_stop/run.csv")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True); axes[0].plot(l["time"], l["x_ref"], "k--", label="reference"); axes[0].plot(p["time"], p["uav_x"], label="PID"); axes[0].plot(l["time"], l["uav_x"], label="LQR"); axes[1].plot(p["time"], p["tip_displacement"], label="PID"); axes[1].plot(l["time"], l["tip_displacement"], label="LQR"); axes[2].plot(p["time"], p["ax_cmd_limited"], label="PID"); axes[2].plot(l["time"], l["ax_cmd_limited"], label="LQR")
    for ax in axes: ax.grid(True, alpha=0.25); ax.legend();
    axes[0].set_ylabel("x (m)"); axes[1].set_ylabel("tip (m)"); axes[2].set_ylabel("ax (m/s²)"); axes[2].set_xlabel("time (s)"); fig.tight_layout(); fig.savefig(out / "pid_vs_lqr_approach_stop.png", dpi=160); plt.close(fig)
    return 0


if __name__ == "__main__": raise SystemExit(main())
