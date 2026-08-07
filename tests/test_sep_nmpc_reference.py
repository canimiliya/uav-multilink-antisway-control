from pathlib import Path

import numpy as np

from uav_sway.paper_baseline.sep_nmpc_reference import load_reference, preview


def test_preview_has_41_boundaries_and_holds_final_sample():
    ref = load_reference(Path("artifacts/s2/references/approach_stop.csv"))
    result = preview(ref, len(ref["time"]) - 2)
    assert all(value.shape == (41,) for value in result.values())
    assert result["x_ref"][-1] == result["x_ref"][-2]
    assert np.isfinite(result["ax_ref"]).all()
