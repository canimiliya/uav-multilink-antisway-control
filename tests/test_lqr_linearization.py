from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_central_difference_artifacts_are_finite_and_converged():
    A = np.load(ROOT / "artifacts/s4/linearization/A.npy")
    B = np.load(ROOT / "artifacts/s4/linearization/B.npy")
    assert A.shape == (16, 16)
    assert B.shape == (16, 1)
    assert np.isfinite(A).all() and np.isfinite(B).all()
    import json
    audit = json.loads((ROOT / "artifacts/s4/linearization/finite_difference.json").read_text(encoding="utf-8"))
    assert audit["half_epsilon_state_relative_fro_error"] < 0.05
    assert audit["half_epsilon_input_relative_fro_error"] < 0.05
    assert audit["repeat_max_abs_A_error"] < 1e-10
    assert audit["repeat_max_abs_B_error"] < 1e-10


def test_linearization_validation_is_recorded_without_overclaiming():
    import json
    validation = json.loads((ROOT / "artifacts/s4/linearization/validation.json").read_text(encoding="utf-8"))
    assert validation["sample_count"] == 20
    assert validation["finite"]
    assert validation["median_normalized_error"] > 0.10 or validation["p95_normalized_error"] > 0.25
