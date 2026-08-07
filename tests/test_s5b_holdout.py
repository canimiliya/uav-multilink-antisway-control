from __future__ import annotations

import numpy as np

from uav_sway.evaluation.s5b_holdout import bootstrap_mean_ci, percentile_summary, safety_check


def test_percentile_summary_contains_frozen_descriptive_statistics():
    summary = percentile_summary(np.arange(1.0, 21.0))
    assert summary["mean"] == 10.5
    assert summary["median"] == 10.5
    assert summary["p25"] == 5.75
    assert summary["p75"] == 15.25
    assert summary["p95"] == 19.05


def test_bootstrap_uses_a_seeded_generator_and_returns_ci():
    values = np.asarray([1.0, 2.0, 3.0])
    a = bootstrap_mean_ci(values, np.random.Generator(np.random.PCG64(20260812)), count=1000)
    b = bootstrap_mean_ci(values, np.random.Generator(np.random.PCG64(20260812)), count=1000)
    assert a == b
    assert a[0] <= 2.0 <= a[1]


def test_safety_check_rejects_each_hard_failure_class():
    base = {
        "finite_outputs": True, "anchor_active_any": False, "minimum_uav_height_m": 3.0,
        "minimum_tip_height_m": 0.3, "maximum_abs_joint_angle_rad": 0.1,
        "maximum_abs_roll_rad": 0.1, "maximum_abs_pitch_rad": 0.1,
        "maximum_abs_ax_m_s2": 1.0, "maximum_ax_step_change_m_s2": 0.1,
        "thrust_min_N": 100.0, "thrust_max_N": 150.0, "torque_max_abs_Nm": 1.0,
        "rotor_motor_max_abs_cmd": 0.0,
    }
    safe, reasons = safety_check(base)
    assert safe and not reasons
    bad = dict(base, maximum_ax_step_change_m_s2=0.3, rotor_motor_max_abs_cmd=1.0)
    safe, reasons = safety_check(bad)
    assert not safe
    assert {"ax_slew", "rotor_motors"}.issubset(reasons)
