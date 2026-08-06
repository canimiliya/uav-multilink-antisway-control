"""Analytic, reference-only x trajectories for the three S2 scenarios."""

from __future__ import annotations

import numpy as np


def smoothstep(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=float)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


def _segment_move(t: np.ndarray, start: float, end: float, distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = end - start
    tau = np.clip((t - start) / duration, 0.0, 1.0)
    s, ds, dds = smoothstep(tau)
    return distance * s, distance / duration * ds, distance / duration**2 * dds


def generate_reference(scenario: str, time: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    t = np.asarray(time, dtype=float)
    if not np.isfinite(t).all():
        raise ValueError("time contains non-finite values")
    x = np.zeros_like(t)
    vx = np.zeros_like(t)
    ax = np.zeros_like(t)
    event = np.full(t.shape, "hover", dtype=object)
    if scenario == "approach_stop":
        m = (t >= 1.0) & (t < 2.0)
        dx, dv, da = _segment_move(t[m], 1.0, 2.0, 0.375)
        x[m], vx[m], ax[m] = dx, dv, da
        m = (t >= 2.0) & (t < 5.0)
        x[m], vx[m] = 0.375 + 0.75 * (t[m] - 2.0), 0.75
        m = (t >= 5.0) & (t < 6.0)
        dx, dv, da = _segment_move(t[m], 5.0, 6.0, 0.375)
        x[m], vx[m], ax[m] = 2.625 + dx, 0.75 - dv, -da
        x[t >= 6.0], vx[t >= 6.0], ax[t >= 6.0] = 3.0, 0.0, 0.0
        event[(t >= 1.0) & (t < 2.0)] = "accelerate"
        event[(t >= 2.0) & (t < 5.0)] = "cruise"
        event[(t >= 5.0) & (t < 6.0)] = "decelerate"
        event[t >= 6.0] = "hold"
    elif scenario == "crosswind_hover":
        event[t >= 4.0] = "wind_onset"
    elif scenario == "gust_micro_adjust":
        m = (t >= 3.0) & (t < 5.0)
        dx, dv, da = _segment_move(t[m], 3.0, 5.0, 0.30)
        x[m], vx[m], ax[m] = dx, dv, da
        x[t >= 5.0], vx[t >= 5.0], ax[t >= 5.0] = 0.30, 0.0, 0.0
        event[(t >= 3.0) & (t < 5.0)] = "micro_adjust"
        event[(t >= 5.0) & (t <= 7.0)] = "gust"
        event[t > 7.0] = "hold"
    else:
        raise KeyError(scenario)
    y = np.full_like(t, float(config[scenario]["y_ref"]))
    z = np.full_like(t, float(config[scenario]["z_ref"]))
    yaw = np.full_like(t, float(config[scenario]["yaw_ref"]))
    return {"time": t, "x_ref": x, "vx_ref": vx, "ax_ref": ax, "y_ref": y, "z_ref": z, "yaw_ref": yaw, "event": event}
