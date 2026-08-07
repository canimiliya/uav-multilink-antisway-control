import numpy as np

from uav_sway.evaluation.task_space_metrics import first_continuous_acquisition, task_acquisition_mask


def test_acquisition_requires_one_second_continuous_hold():
    time = np.arange(0.0, 2.01, 0.05)
    mask = np.ones_like(time, dtype=bool)
    acquired, at = first_continuous_acquisition(time, mask)
    assert acquired and np.isclose(at, 0.0)


def test_brief_entry_does_not_acquire():
    time = np.arange(0.0, 2.01, 0.05)
    mask = np.zeros_like(time, dtype=bool)
    mask[5:10] = True
    acquired, at = first_continuous_acquisition(time, mask)
    assert not acquired and at is None


def test_unreachable_returns_null():
    time = np.arange(0.0, 2.01, 0.05)
    acquired, at = first_continuous_acquisition(time, np.zeros_like(time, dtype=bool))
    assert not acquired and at is None


def test_acquisition_time_is_elapsed_from_task_start():
    time = np.arange(0.0, 3.01, 0.5)
    mask = np.ones_like(time, dtype=bool)
    acquired, timestamp = first_continuous_acquisition(time, mask, hold_time_s=1.0, start_time_s=1.0)
    assert acquired and np.isclose(timestamp, 1.0)
    assert np.isclose(timestamp - 1.0, 0.0)


def test_acquisition_timestamp_and_elapsed_time_are_distinct():
    time = np.arange(0.0, 4.01, 0.5)
    mask = np.zeros_like(time, dtype=bool)
    mask[5:] = True
    acquired, timestamp = first_continuous_acquisition(time, mask, hold_time_s=1.0, start_time_s=1.0)
    assert acquired and np.isclose(timestamp, 2.5)
    assert np.isclose(timestamp - 1.0, 1.5)


def test_mask_uses_all_three_task_conditions():
    mask = task_acquisition_mask(np.array([0.04, 0.04]), np.array([4.0, 6.0]), np.array([0.09, 0.09]))
    assert mask.tolist() == [True, False]
