from uav_sway.evaluation.its_rmpc_gate import candidate_score


def _m(position=1.0, orientation=1.0, acquisition=1.0, rate=1.0):
    return {"tip_task_position_rmse_m": position, "cutter_orientation_rmse_deg": orientation, "task_acquisition_time_s": acquisition, "control_rate_proxy": rate}


def test_score_monotonicity():
    trad = {"task_acquire_calm": {"best_position_rmse": 1.0, "best_orientation_rmse": 1.0, "available_task_time_s": 10.0}, "task_acquire_crosswind": {"best_position_rmse": 1.0, "best_orientation_rmse": 1.0, "available_task_time_s": 10.0}}
    old = {"task_acquire_calm": {"control_rate_proxy": 1.0}, "task_acquire_crosswind": {"control_rate_proxy": 1.0}}
    better = candidate_score({s: _m() for s in trad}, trad, old)["score"]
    worse = candidate_score({s: _m(position=1.2, orientation=1.1, acquisition=2.0, rate=2.0) for s in trad}, trad, old)["score"]
    assert worse > better
