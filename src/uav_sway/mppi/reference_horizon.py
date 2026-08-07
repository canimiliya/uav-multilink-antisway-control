"""Reference-window handling for the fixed 200 Hz S2 signal contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sway.control.base import ReferenceState


@dataclass(frozen=True)
class ReferenceHorizon:
    samples: tuple[ReferenceState, ...]
    indices: tuple[int, ...]
    times: np.ndarray

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ReferenceState:
        return self.samples[index]


def make_reference_horizon(reference: dict[str, np.ndarray], signal_index: int,
                           horizon_steps: int) -> ReferenceHorizon:
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    count = len(reference["time"])
    if count == 0:
        raise ValueError("reference is empty")
    indices = tuple(min(count - 1, int(signal_index + 10 * j)) for j in range(horizon_steps))
    samples = tuple(
        ReferenceState(*(float(reference[name][idx]) for name in
                         ("x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref")))
        for idx in indices
    )
    times = np.asarray([float(reference["time"][idx]) for idx in indices], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0.0):
        raise ValueError("reference horizon times must be finite and nondecreasing")
    return ReferenceHorizon(samples, indices, times)
