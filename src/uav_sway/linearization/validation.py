"""Independent operating-region and genuinely local linearization validation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _aggregate(errors: np.ndarray, scales: np.ndarray) -> dict:
    normalized_vectors = errors / scales.reshape(1, -1)
    normalized_norms = np.linalg.norm(normalized_vectors, axis=1) / np.sqrt(errors.shape[1])
    per_state_abs = np.sqrt(np.mean(errors**2, axis=0))
    per_state_norm = np.sqrt(np.mean(normalized_vectors**2, axis=0))
    per_state_p95 = np.percentile(np.abs(normalized_vectors), 95, axis=0)
    return {
        "sample_count": int(len(errors)),
        "median_normalized_error": float(np.median(normalized_norms)),
        "p95_normalized_error": float(np.percentile(normalized_norms, 95)),
        "maximum_normalized_error": float(np.max(normalized_norms)),
        "finite": bool(np.isfinite(errors).all() and np.isfinite(normalized_norms).all()),
        "absolute_rmse_by_state": per_state_abs.tolist(),
        "normalized_rmse_by_state": per_state_norm.tolist(),
        "p95_error_by_state": per_state_p95.tolist(),
        "worst_state_index": int(np.argmax(per_state_p95)),
    }


def operating_region_validation(
    phi: Callable[[np.ndarray, float], np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    seed: int = 20260807,
    sample_count: int = 20,
) -> dict:
    """Preserve the original wide-range 20-point validation unchanged."""
    scales = np.asarray(
        [0.02, 0.05, 0.01, 0.03, 0.01, 0.03, *([0.01] * 5), *([0.03] * 5)],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(sample_count):
        state = rng.uniform(-1.0, 1.0, 16) * scales
        command = float(rng.uniform(-0.05, 0.05))
        nonlinear = np.asarray(phi(state, command), dtype=float)
        predicted = a @ state + b[:, 0] * command
        errors.append(float(np.linalg.norm((nonlinear - predicted) / scales) / np.sqrt(16.0)))
    errors_array = np.asarray(errors, dtype=float)
    return {
        "seed": seed,
        "sample_count": sample_count,
        "state_ranges": scales.tolist(),
        "input_range": [-0.05, 0.05],
        "normalized_errors": errors_array.tolist(),
        "median_normalized_error": float(np.median(errors_array)),
        "p95_normalized_error": float(np.percentile(errors_array, 95)),
        "finite": bool(np.isfinite(errors_array).all()),
        "result": "operating_region_limitation",
    }


def local_validation(
    phi: Callable[[np.ndarray, float], np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    phi_zero: np.ndarray,
    epsilon: np.ndarray,
    input_epsilon: float,
    seed: int = 20260808,
    sample_count: int = 200,
    multipliers: tuple[int, ...] = (2, 5, 10),
) -> dict:
    """Validate ``Phi(0,0) + A x + B u`` with mirrored local samples."""
    if sample_count % 2:
        raise ValueError("local sample_count must be even for mirror samples")
    rng = np.random.default_rng(seed)
    half = sample_count // 2
    scales_by_multiplier: dict[str, list[float]] = {}
    results: dict[str, dict] = {}
    per_state: dict[str, dict] = {}
    for multiplier in multipliers:
        state_scale = np.asarray(epsilon, dtype=float) * multiplier
        input_scale = float(input_epsilon) * multiplier
        errors = []
        for _ in range(half):
            state = rng.uniform(-1.0, 1.0, 16) * state_scale
            command = float(rng.uniform(-1.0, 1.0) * input_scale)
            for sign in (1.0, -1.0):
                signed_state = sign * state
                signed_command = sign * command
                nonlinear = np.asarray(phi(signed_state, signed_command), dtype=float)
                predicted = np.asarray(phi_zero, dtype=float) + a @ signed_state + b[:, 0] * signed_command
                errors.append(nonlinear - predicted)
        errors_array = np.asarray(errors, dtype=float)
        key = f"{multiplier}x_epsilon"
        scales_by_multiplier[key] = state_scale.tolist()
        result = _aggregate(errors_array, state_scale)
        result.update({"multiplier": multiplier, "state_scale": state_scale.tolist(), "input_scale": input_scale, "mirror_samples": True})
        results[key] = result
        per_state[key] = {
            "absolute_rmse": result["absolute_rmse_by_state"],
            "normalized_rmse": result["normalized_rmse_by_state"],
            "p95_error": result["p95_error_by_state"],
            "worst_state_index": result["worst_state_index"],
        }
    final = results["10x_epsilon"]
    return {
        "seed": seed,
        "sample_count": sample_count,
        "base_sample_count": half,
        "mirror_samples": True,
        "finite_difference_state_epsilon": np.asarray(epsilon, dtype=float).tolist(),
        "finite_difference_input_epsilon": float(input_epsilon),
        "phi_zero": np.asarray(phi_zero, dtype=float).tolist(),
        "scales": scales_by_multiplier,
        "by_multiplier": results,
        "median_normalized_error": final["median_normalized_error"],
        "p95_normalized_error": final["p95_normalized_error"],
        "maximum_normalized_error": final["maximum_normalized_error"],
        "worst_state_index": final["worst_state_index"],
        "finite": bool(all(item["finite"] for item in results.values())),
        "pass": bool(final["finite"] and final["median_normalized_error"] < 0.10 and final["p95_normalized_error"] < 0.25),
        "local_validation_reference": "10x_epsilon",
    }, per_state
