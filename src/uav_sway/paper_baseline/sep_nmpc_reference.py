"""Reference loading and 2-second preview construction for SEP-NMPC."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_reference(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty reference")
    names = ("time", "x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref")
    result = {name: np.asarray([float(row[name]) for row in rows], dtype=float) for name in names}
    result["event"] = np.asarray([row.get("event", "") for row in rows], dtype=object)
    return result


def preview(reference: dict[str, np.ndarray], start: int, nodes: int = 40) -> dict[str, np.ndarray]:
    if nodes != 40:
        raise ValueError("S5D2 requires 40 shooting intervals")
    indices = np.minimum(np.arange(start, start + nodes + 1), len(reference["time"]) - 1)
    return {name: np.asarray(reference[name][indices], dtype=float) for name in ("x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref")}
