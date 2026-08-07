"""Cutter-task-space PID outer acceleration controller for S6T1."""

from __future__ import annotations

from dataclasses import dataclass

from .acceleration_limiter import AccelerationLimiter


@dataclass(frozen=True)
class TaskPIDDiagnostics:
    task_position_error_x: float
    task_velocity_error_x: float
    pid_integral_x: float
    ax_pid_feedback: float
    ax_reference_feedforward: float
    ax_cmd_raw: float
    ax_cmd_amplitude_limited: float
    ax_cmd_limited: float
    ax_saturated: bool
    ax_slew_limited: bool


class TaskPID:
    """PID using cutter-tip x position/velocity rather than UAV x state."""

    def __init__(self, kp: float, kd: float, ki: float, ax_min: float = -2.0,
                 ax_max: float = 2.0, slew_limit: float = 0.25,
                 integral_limit: float = 1.0):
        self.kp = float(kp)
        self.kd = float(kd)
        self.ki = float(ki)
        self.ax_min = float(ax_min)
        self.ax_max = float(ax_max)
        self.slew_limit = float(slew_limit)
        self.integral_limit = float(integral_limit)
        self.integral = 0.0
        self.limiter = AccelerationLimiter(self.ax_min, self.ax_max, self.slew_limit)
        self.diagnostics = TaskPIDDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

    def reset(self, state=None, reference=None) -> None:
        del state, reference
        self.integral = 0.0
        self.limiter.reset(0.0)
        self.diagnostics = TaskPIDDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)

    def command(self, tip_x: float, tip_vx: float, tip_x_ref: float,
                vx_ref: float, ax_ref: float, dt: float) -> float:
        if dt <= 0.0:
            raise ValueError("Task-PID dt must be positive")
        ex = float(tip_x - tip_x_ref)
        ev = float(tip_vx - vx_ref)
        feedback = -self.kp * ex - self.kd * ev - self.ki * self.integral
        raw = float(ax_ref + feedback)
        amplitude = min(self.ax_max, max(self.ax_min, raw))
        amplitude_saturated = amplitude != raw

        candidate_integral = self.integral + ex * dt
        candidate_integral = min(self.integral_limit, max(-self.integral_limit, candidate_integral))
        pushing_high = amplitude_saturated and raw > self.ax_max and ex < 0.0
        pushing_low = amplitude_saturated and raw < self.ax_min and ex > 0.0
        if not (pushing_high or pushing_low):
            self.integral = candidate_integral
        self.integral = min(self.integral_limit, max(-self.integral_limit, self.integral))

        feedback = -self.kp * ex - self.kd * ev - self.ki * self.integral
        raw = float(ax_ref + feedback)
        amplitude = min(self.ax_max, max(self.ax_min, raw))
        limited = self.limiter.limit(raw)
        limiter_diag = self.limiter.diagnostics
        self.diagnostics = TaskPIDDiagnostics(
            ex, ev, float(self.integral), float(feedback), float(ax_ref), float(raw),
            float(amplitude), float(limited), bool(limiter_diag.saturated),
            bool(limiter_diag.slew_limited),
        )
        return float(limited)
