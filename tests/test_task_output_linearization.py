from pathlib import Path

import numpy as np

from uav_sway.evaluation.task_baseline_runner import make_task_output_map
from uav_sway.linearization.task_output import identify_task_output_jacobian, validate_task_output_local


ROOT = Path(__file__).resolve().parents[1]


def test_task_output_jacobian_and_mirrored_validation():
    _, _, task_map, _ = make_task_output_map(ROOT)
    epsilon = np.asarray([1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-4, *([1e-5] * 5), *([1e-4] * 5)])
    c_task, y0 = identify_task_output_jacobian(task_map, epsilon)
    report = validate_task_output_local(task_map, c_task, epsilon, sample_count=8)
    assert c_task.shape == (4, 16)
    assert np.isfinite(c_task).all()
    assert np.allclose(y0, 0.0, atol=1e-10)
    assert report["pass"]
