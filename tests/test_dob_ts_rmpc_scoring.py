from uav_sway.evaluation.dob_ts_rmpc_gate import mpc_contribution


def _m(acquired, pos=1.0, ori=1.0, acq=2.0):
    return {"task_acquired": acquired, "tip_task_position_rmse_m": pos,
            "cutter_orientation_rmse_deg": ori, "task_acquisition_time_s": acq}


def test_acquisition_dominance_is_supported():
    full = {"task_acquire_calm": _m(True), "task_acquire_crosswind": _m(True)}
    dob = {"task_acquire_calm": _m(True), "task_acquire_crosswind": _m(False)}
    result = mpc_contribution(full, dob)
    assert result["acquisition_dominance"]
    assert result["pass"]


def test_mpc_score_rejects_degradation_before_contribution():
    full = {"task_acquire_calm": _m(True, 2.0), "task_acquire_crosswind": _m(True)}
    dob = {"task_acquire_calm": _m(True), "task_acquire_crosswind": _m(True)}
    assert not mpc_contribution(full, dob)["pass"]

