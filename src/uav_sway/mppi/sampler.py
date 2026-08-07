"""Deterministic Gaussian sampling and numerically stable MPPI update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MPPIUpdate:
    updated_sequence: np.ndarray
    weights: np.ndarray
    weight_sum: float
    weight_max: float
    effective_sample_size: float
    cost_min: float
    cost_mean: float
    cost_std: float


def stable_mppi_update(nominal: np.ndarray, noise: np.ndarray, costs: np.ndarray,
                       temperature: float) -> MPPIUpdate:
    nominal = np.asarray(nominal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if nominal.ndim != 1 or noise.ndim != 2 or noise.shape[1] != nominal.size:
        raise ValueError("incompatible nominal/noise shapes")
    if costs.shape != (noise.shape[0],) or temperature <= 0.0:
        raise ValueError("invalid MPPI costs or temperature")
    if not np.isfinite(costs).all():
        raise ValueError("MPPI costs must be finite")
    rho = float(np.min(costs))
    logits = -(costs - rho) / float(temperature)
    logits -= float(np.max(logits))
    unnormalized = np.exp(logits)
    weights = unnormalized / float(np.sum(unnormalized))
    updated = nominal + weights @ noise
    return MPPIUpdate(updated, weights, float(np.sum(weights)),
                      float(np.max(weights)), float(1.0 / np.sum(weights ** 2)),
                      float(np.min(costs)), float(np.mean(costs)), float(np.std(costs)))
