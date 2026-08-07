import csv

from uav_sway.evaluation.its_rmpc_gate import candidate_score, competence_gate, final_tip_speed_from_csv, mpc_contribution


def _m(position=1.0, orientation=1.0, acquisition=1.0, rate=1.0):
    return {"tip_task_position_rmse_m": position, "cutter_orientation_rmse_deg": orientation, "task_acquisition_time_s": acquisition, "control_rate_proxy": rate}


def test_score_monotonicity():
    trad = {"task_acquire_calm": {"best_position_rmse": 1.0, "best_orientation_rmse": 1.0, "available_task_time_s": 10.0}, "task_acquire_crosswind": {"best_position_rmse": 1.0, "best_orientation_rmse": 1.0, "available_task_time_s": 10.0}}
    old = {"task_acquire_calm": {"control_rate_proxy": 1.0}, "task_acquire_crosswind": {"control_rate_proxy": 1.0}}
    better = candidate_score({s: _m() for s in trad}, trad, old)["score"]
    worse = candidate_score({s: _m(position=1.2, orientation=1.1, acquisition=2.0, rate=2.0) for s in trad}, trad, old)["score"]
    assert worse > better


def _gate_metrics(*, acquired=True, final_speed=0.05, rms_speed=0.20, position=1.0, orientation=1.0, acquisition=1.0):
    return {
        "task_acquired": acquired,
        "tip_task_position_rmse_m": position,
        "cutter_orientation_rmse_deg": orientation,
        "final_tip_position_error_m": 0.01,
        "final_orientation_error_deg": 1.0,
        "final_tip_speed_m_s": final_speed,
        "tip_speed_rms_m_s": rms_speed,
        "task_acquisition_time_s": acquisition if acquired else None,
    }


def test_final_speed_gate_uses_final_sample_not_whole_run_rms():
    traditional = {scene: {"best_position_rmse": 1.0, "best_orientation_rmse": 1.0} for scene in ("task_acquire_calm", "task_acquire_crosswind")}
    metrics = {scene: _gate_metrics(final_speed=0.05, rms_speed=0.20) for scene in traditional}
    assert competence_gate(metrics, traditional)["pass"]
    metrics = {scene: _gate_metrics(final_speed=0.11, rms_speed=0.05) for scene in traditional}
    assert not competence_gate(metrics, traditional)["pass"]


def test_final_tip_speed_reads_last_csv_sample(tmp_path):
    path = tmp_path / "run.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time", "tip_speed_m_s"])
        writer.writeheader(); writer.writerow({"time": 0.0, "tip_speed_m_s": 0.2}); writer.writerow({"time": 1.0, "tip_speed_m_s": 0.04})
    assert final_tip_speed_from_csv(path) == 0.04


def test_acquisition_dominance_when_lqi_fails_and_its_acquires():
    lqi = {scene: _gate_metrics(acquired=False) for scene in ("task_acquire_calm", "task_acquire_crosswind")}
    its = {"task_acquire_calm": _gate_metrics(acquired=True), "task_acquire_crosswind": _gate_metrics(acquired=True)}
    result = mpc_contribution(its, lqi)
    assert result["acquisition_dominance"] is True
    assert result["pass"] is True


def test_acquisition_loss_fails_contribution():
    lqi = {"task_acquire_calm": _gate_metrics(acquired=True), "task_acquire_crosswind": _gate_metrics(acquired=True)}
    its = {"task_acquire_calm": _gate_metrics(acquired=True), "task_acquire_crosswind": _gate_metrics(acquired=False)}
    result = mpc_contribution(its, lqi)
    assert result["acquisition_dominance"] is False
    assert result["pass"] is False


def test_both_acquired_uses_acquisition_time_improvement():
    lqi = {scene: _gate_metrics(acquired=True, acquisition=10.0) for scene in ("task_acquire_calm", "task_acquire_crosswind")}
    its = {scene: _gate_metrics(acquired=True, acquisition=9.0) for scene in ("task_acquire_calm", "task_acquire_crosswind")}
    result = mpc_contribution(its, lqi)
    assert result["acquisition_time_comparison_available"] is True
    assert result["acquisition_improvement_vs_lqi"] == 0.1
    assert result["pass"] is True
