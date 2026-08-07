"""DOB-TS-RMPC: frozen DOB-Task-LQR plus a constrained task residual."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sway.control.dob_task_lqr import DOBTaskLQR
from uav_sway.mpc.dob_task_residual_qp import DOBTaskResidualQP


@dataclass(frozen=True)
class DOBTSRMPCDiagnostics:
    lqr_feedback_ax: float
    disturbance_hat: float
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


class DOBTSRMPC(DOBTaskLQR):
    def __init__(self, gain, A, B, qp: DOBTaskResidualQP, observer_gain,
                 disturbance_limit=2.0, ax_min=-2.0, ax_max=2.0,
                 slew_limit=0.25, use_observer=True):
        self.qp = qp
        super().__init__(gain, A, B, observer_gain, disturbance_limit,
                         ax_min, ax_max, slew_limit, use_observer)

    def command(self, state, reference, dt=0.05):
        x = np.asarray(state, dtype=float).reshape(16)
        if self.use_observer:
            shift = None
            if self._previous_reference is not None:
                from uav_sway.mpc.preview_model import PreviewModel
                shift = PreviewModel.static_reference_shift(self.A, self._previous_reference, reference)
            d_hat = self.observer.update(x, self.limiter.previous, shift)
            self._observer_initialized = bool(self.observer._initialized)
        else:
            d_hat = 0.0
        self._previous_reference = reference
        feedback = float((-self.gain @ x.reshape(-1, 1))[0, 0])
        result = self.qp.solve(x, d_hat, self.limiter.previous)
        residual = float(result["v"])
        raw = feedback - float(d_hat) - residual
        limited = self.limiter.limit(raw)
        diag = self.limiter.diagnostics
        predicted = float(result.get("predicted_first_action", raw))
        self.diagnostics = DOBTSRMPCDiagnostics(
            feedback, float(d_hat), residual, raw, float(diag.amplitude_limited),
            float(limited), bool(diag.saturated), bool(diag.slew_limited),
            predicted, float(predicted - limited), str(result["status"]),
            int(result["status_val"]), float(result["solve_time_ms"]))
        return float(limited)

