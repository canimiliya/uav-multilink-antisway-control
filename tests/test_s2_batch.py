import hashlib
import json
from pathlib import Path

from uav_sway.evaluation.metrics import compute_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_and_headless_smoke_outputs_are_reproducible():
    s2 = ROOT / "artifacts/s2"
    protocol = json.loads((s2 / "protocol_summary.json").read_text(encoding="utf-8"))
    assert protocol["signal_sample_count"] == 2401
    assert protocol["controller_implemented"] is False
    wind_manifest = json.loads((s2 / "wind_bank/manifest.json").read_text(encoding="utf-8"))
    assert len(wind_manifest["files"]) == 22
    for entry in wind_manifest["files"]:
        path = ROOT / entry["path"]
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    for case in ("constant_crosswind", "one_cosine_gust", "random_seed_000"):
        run = s2 / "smoke" / case / "run.csv"
        metrics_path = s2 / "smoke" / case / "metrics.json"
        assert run.exists() and metrics_path.exists()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        recomputed = compute_metrics(run, settling_start_s={"constant_crosswind": 4.0, "one_cosine_gust": 5.0, "random_seed_000": 6.0}[case])
        assert metrics["sample_count"] == 2401
        assert metrics["finite_outputs"] is True
        assert recomputed["tip_max_abs_m"] == metrics["tip_max_abs_m"]
        assert recomputed["tip_rms_m"] == metrics["tip_rms_m"]
        assert not (s2 / "smoke" / case / "render.png").exists()


def test_reference_and_metric_audits_pass():
    s2 = ROOT / "artifacts/s2"
    continuity = json.loads((s2 / "reference_continuity_audit.json").read_text(encoding="utf-8"))
    assert continuity["continuous_x"] is True
    assert continuity["continuous_vx"] is True
    assert continuity["continuous_ax"] is True
    assert continuity["sampled_vx_jump_at_2s_m_s"] < 1e-3
    assert continuity["sampled_vx_jump_at_6s_m_s"] < 1e-3

    formula = json.loads((s2 / "metrics_formula_audit.json").read_text(encoding="utf-8"))
    assert formula["uniform_case"]["computed"] == 5.0
    assert formula["nonuniform_case"]["computed"] == 8.0
    assert formula["constant_case_computed"] == 0.0
