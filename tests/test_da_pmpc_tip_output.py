import numpy as np
from pathlib import Path

def test_tip_output_artifact_if_identified():
    path=Path("artifacts/s5a/model/C_tip.npy")
    if path.exists():
        c=np.load(path); assert c.shape==(1,16) and np.isfinite(c).all()
