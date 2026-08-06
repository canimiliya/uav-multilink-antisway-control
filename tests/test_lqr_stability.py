from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_dare_and_closed_loop_stability_artifact():
    report = json.loads((ROOT / "artifacts/s4/lqr/closed_loop_eigenvalues.json").read_text(encoding="utf-8"))
    assert report["spectral_radius"] < 0.999
    assert report["dare_residual_norm"] < 1e-8
    assert report["p_symmetry_error"] < 1e-9
    assert report["p_min_eigenvalue"] >= -1e-10
