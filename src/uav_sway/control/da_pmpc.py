"""Disturbance-aware preview MPC pilot controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acceleration_limiter import AccelerationLimiter
from .base import ReferenceState
from .disturbance_observer import MatchedDisturbanceObserver
from uav_sway.mpc.qp_builder import build_preview_qp
from uav_sway.mpc.preview_model import PreviewModel


@dataclass(frozen=True)
class DAPMPCDiagnostics:
    ax_cmd_raw: float
    ax_cmd_limited: float
    delta_ax: float
    disturbance_hat: float
    solve_time_ms: float
    status: str
    iterations: int


class DAPMPC:
    def __init__(self, A, B, Q, P, C_tip, position_scale, tip_weight,
                 solver, observer=None, horizon_steps=20, ax_min=-2.0,
                 ax_max=2.0, slew_limit=0.25):
        self.A=np.asarray(A); self.B=np.asarray(B); self.Q=np.asarray(Q); self.P=np.asarray(P); self.C_tip=np.asarray(C_tip)
        self.position_scale=float(position_scale); self.tip_weight=float(tip_weight)
        self.solver=solver; self.horizon_steps=int(horizon_steps)
        self.ax_min=float(ax_min); self.ax_max=float(ax_max); self.slew_limit=float(slew_limit)
        self.limiter=AccelerationLimiter(ax_min, ax_max, slew_limit)
        self.observer=observer or MatchedDisturbanceObserver(A,B)
        self.diagnostics=DAPMPCDiagnostics(0,0,0,0,0,"reset",0)

    def reset(self, state=None, reference=None):
        del reference
        self.limiter.reset(0.0); self.observer.reset(state, 0.0, 0.0)
        self.diagnostics=DAPMPCDiagnostics(0,0,0,0,0,"reset",0)

    def command(self, state, horizon, solve_time_ms=0.0):
        state=np.asarray(state,dtype=float).reshape(16)
        d_hat=self.observer.update(state, self.limiter.previous)
        references=horizon.boundary_samples
        preview = PreviewModel(self.A, self.B, self.Q, self.P, self.C_tip, self.horizon_steps)
        qp=build_preview_qp(preview, state, references, self.limiter.previous,
                            self.position_scale, self.tip_weight, self.ax_min,
                            self.ax_max, self.slew_limit, d_hat)
        solution, info=self.solver.solve(qp)
        delta=float(solution[0]); raw=float(horizon.action_reference(0).ax_ref + delta)
        limited=float(self.limiter.limit(raw))
        d=self.limiter.diagnostics
        self.diagnostics=DAPMPCDiagnostics(raw, limited, delta, d_hat, float(solve_time_ms), str(info.status), int(info.iter))
        return limited
