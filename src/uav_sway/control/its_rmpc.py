"""Integral Task-Space Residual MPC (ITS-RMPC) for S6T3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sway.control.acceleration_limiter import AccelerationLimiter
from uav_sway.mpc.task_residual_qp import TaskResidualQP


@dataclass(frozen=True)
class ITSDiagnostics:
    eta: float
    lqi_feedback_ax: float
    residual_v: float
    ax_cmd_raw: float
    ax_cmd_amplitude_limited: float
    ax_cmd_limited: float
    ax_saturated: bool
    ax_slew_limited: bool
    predicted_first_action: float
    first_action_mismatch: float
    qp_status: str
    qp_status_val: int
    solve_time_ms: float


class TaskLQI:
    """Integral task-space stabilizer used as the matched ablation."""

    def __init__(self, gain: np.ndarray, ax_min=-2.0, ax_max=2.0, slew_limit=0.25):
        self.gain = np.asarray(gain, dtype=float).reshape(1, 17)
        self.limiter = AccelerationLimiter(ax_min, ax_max, slew_limit)
        self.reset()

    def reset(self, state=None, reference=None) -> None:
        del state, reference
        self.eta = 0.0
        self.limiter.reset(0.0)
        self.diagnostics = ITSDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          False, False, 0.0, 0.0, "not_solved", -1, 0.0)

    def command(self, state: np.ndarray, measured_task_error_x: float,
                reference=None, dt: float = 0.05) -> float:
        del reference
        self.eta = float(np.clip(self.eta + float(dt) * float(measured_task_error_x), -1.0, 1.0))
        z = np.concatenate((np.asarray(state, dtype=float).reshape(16), [self.eta]))
        feedback = float((-self.gain @ z.reshape(-1, 1))[0, 0])
        raw = feedback
        limited = self.limiter.limit(raw)
        diag = self.limiter.diagnostics
        self.diagnostics = ITSDiagnostics(self.eta, feedback, 0.0, raw,
                                          float(diag.amplitude_limited), float(limited),
                                          bool(diag.saturated), bool(diag.slew_limited),
                                          raw, 0.0, "ablation", 1, 0.0)
        return float(limited)


class ITSRMPC(TaskLQI):
    """Task-LQI stabilizer plus a constrained residual action."""

    def __init__(self, gain: np.ndarray, qp: TaskResidualQP, ax_min=-2.0,
                 ax_max=2.0, slew_limit=0.25):
        self.qp = qp
        super().__init__(gain, ax_min, ax_max, slew_limit)

    def command(self, state: np.ndarray, measured_task_error_x: float,
                reference=None, dt: float = 0.05) -> float:
        del reference
        self.eta = float(np.clip(self.eta + float(dt) * float(measured_task_error_x), -1.0, 1.0))
        z = np.concatenate((np.asarray(state, dtype=float).reshape(16), [self.eta]))
        feedback = float((-self.gain @ z.reshape(-1, 1))[0, 0])
        qp_result = self.qp.solve(z, self.limiter.previous)
        residual = float(qp_result["v"])
        raw = feedback - residual
        limited = self.limiter.limit(raw)
        diag = self.limiter.diagnostics
        predicted = float(qp_result.get("predicted_first_action", raw))
        self.diagnostics = ITSDiagnostics(
            self.eta, feedback, residual, raw, float(diag.amplitude_limited),
            float(limited), bool(diag.saturated), bool(diag.slew_limited),
            predicted, float(predicted - limited), str(qp_result["status"]),
            int(qp_result["status_val"]), float(qp_result["solve_time_ms"]),
        )
        return float(limited)
