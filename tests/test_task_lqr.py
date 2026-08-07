from pathlib import Path

import numpy as np

from uav_sway.control.task_lqr import build_task_lqr


ROOT = Path(__file__).resolve().parents[1]


def test_task_lqr_uses_s4_ab_and_has_stable_dare_solution():
    a = np.load(ROOT / "artifacts/s4/linearization/A.npy")
    b = np.load(ROOT / "artifacts/s4/linearization/B.npy")
    c = np.zeros((4, 16), dtype=float)
    result = build_task_lqr(a, b, c, 80.0, 20.0, 1.0)
    assert result["K"].shape == (1, 16)
    assert np.isfinite(result["K"]).all()
    assert result["spectral_radius"] < 1.0
