from uav_sway.evaluation.task_baseline_scoring import task_baseline_score


def test_task_baseline_score_penalizes_each_worse_term():
    base = task_baseline_score([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
    assert task_baseline_score([1.1, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]) > base
    assert task_baseline_score([1.0, 1.0], [1.1, 1.0], [1.0, 1.0], [1.0, 1.0]) > base
    assert task_baseline_score([1.0, 1.0], [1.0, 1.0], [1.1, 1.0], [1.0, 1.0]) > base
    assert task_baseline_score([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.1, 1.0]) > base


def test_not_acquired_and_slow_acquisition_are_penalized():
    acquired_fast = task_baseline_score([1.0], [1.0], [0.2], [1.0])
    acquired_slow = task_baseline_score([1.0], [1.0], [1.0], [1.0])
    not_acquired = task_baseline_score([1.0], [1.0], [1.5], [1.0])
    assert acquired_slow > acquired_fast
    assert not_acquired > acquired_slow
