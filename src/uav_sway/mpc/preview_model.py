"""Fixed-horizon affine preview model used by the DA-PMPC pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreviewResult:
    states: np.ndarray
    input_matrix: np.ndarray


class PreviewModel:
    """Builds x[i+1] = A x[i] - B(a[i]+d) - c[i].

    The minus sign is the frozen DA-PMPC error-coordinate contract.  The
    imported S4 B is retained separately in ``physical_B`` for auditability.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray,
                 P: np.ndarray, C_tip: np.ndarray, horizon_steps: int = 20):
        self.A = np.asarray(A, dtype=float)
        self.physical_B = np.asarray(B, dtype=float).reshape(16, 1)
        self.B = -self.physical_B
        self.Q = np.asarray(Q, dtype=float)
        self.P = np.asarray(P, dtype=float)
        self.C_tip = np.asarray(C_tip, dtype=float).reshape(1, 16)
        self.horizon_steps = int(horizon_steps)
        if self.A.shape != (16, 16) or self.B.shape != (16, 1):
            raise ValueError("preview model requires A 16x16 and B 16x1")

    @staticmethod
    def reference_shift(reference_i, reference_next) -> np.ndarray:
        return np.asarray([
            float(reference_next.x_ref - reference_i.x_ref),
            float(reference_next.vx_ref - reference_i.vx_ref),
            float(reference_next.z_ref - reference_i.z_ref),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=float)

    def rollout(self, x0: np.ndarray, actions: np.ndarray, references,
                disturbance: float = 0.0) -> PreviewResult:
        x = np.asarray(x0, dtype=float).reshape(16)
        actions = np.asarray(actions, dtype=float).reshape(self.horizon_steps)
        if len(references) != self.horizon_steps + 1:
            raise ValueError("preview requires H+1 reference boundaries")
        states = np.zeros((self.horizon_steps + 1, 16), dtype=float)
        states[0] = x
        for i in range(self.horizon_steps):
            c = self.reference_shift(references[i], references[i + 1])
            states[i + 1] = self.A @ states[i] - self.B[:, 0] * (actions[i] + float(disturbance)) - c
        return PreviewResult(states=states, input_matrix=self.B.copy())
