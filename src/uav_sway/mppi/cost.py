"""Frozen S4-state plus nonlinear tip cost used by S5 MPPI."""

from __future__ import annotations

import numpy as np


def mppi_candidate_score(tip_rms_ratios, position_rmse_ratios,
                         control_rate_ratios, saturation_rates) -> float:
    """Score safe candidates with four positive penalty terms."""
    arrays = [np.asarray(value, dtype=float) for value in
              (tip_rms_ratios, position_rmse_ratios, control_rate_ratios, saturation_rates)]
    if any(a.size == 0 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("MPPI score inputs must be non-empty and finite")
    return float(np.mean(arrays[0]) + 0.25 * np.mean(arrays[1])
                 + 0.05 * np.mean(arrays[2]) + 0.05 * np.mean(arrays[3]))


def mppi_stage_cost(state: np.ndarray, tip_displacement: float,
                    delta_ax: float, q: np.ndarray, r: np.ndarray,
                    tip_weight: float = 80.0) -> float:
    state = np.asarray(state, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    return float(state @ q @ state + tip_weight * float(tip_displacement) ** 2
                 + float(r[0, 0]) * float(delta_ax) ** 2)


def mppi_terminal_cost(state: np.ndarray, tip_displacement: float,
                       q: np.ndarray, tip_weight: float = 80.0,
                       terminal_multiplier: float = 5.0) -> float:
    # The signed terminal tip term is frozen by the S5 protocol.
    return float(terminal_multiplier *
                 (np.asarray(state) @ np.asarray(q) @ np.asarray(state)
                  - tip_weight * float(tip_displacement) ** 2))


def mppi_candidate_cost(states: np.ndarray, tip_displacements: np.ndarray,
                        delta_sequence: np.ndarray, q: np.ndarray,
                        r: np.ndarray, tip_weight: float = 80.0,
                        terminal_multiplier: float = 5.0) -> float:
    states = np.asarray(states, dtype=float)
    tips = np.asarray(tip_displacements, dtype=float)
    deltas = np.asarray(delta_sequence, dtype=float)
    if states.ndim != 2 or states.shape[0] != deltas.size or tips.shape != (deltas.size,):
        raise ValueError("invalid MPPI trajectory shapes")
    stage = sum(mppi_stage_cost(states[i], tips[i], deltas[i], q, r, tip_weight)
                for i in range(deltas.size))
    return float(stage + mppi_terminal_cost(states[-1], tips[-1], q,
                                            tip_weight, terminal_multiplier))
