"""Runtime wrapper around the formal acados SEP-NMPC OCP."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sep_nmpc_acados import AcadosBuildSpec, build_formal_ocp
from .sep_nmpc_model import PlanarParameters
from .sep_nmpc_ocp import SEPTrackingConfig


@dataclass(frozen=True)
class SEPCommandDiagnostics:
    u_ae: float
    slack: float
    ax_cmd_raw: float
    passivity_residual: float
    acados_status: int
    qp_iterations: float
    solve_time_ms: float
    first_action_slew_residual: float


class FormalSEPController:
    """A scalar-a_x controller with no wind or plant access in its API."""

    def __init__(self, parameters: PlanarParameters, config: SEPTrackingConfig, code_export_directory: str | Path):
        self.parameters = parameters
        self.config = config
        self.code_export_directory = Path(code_export_directory)
        self.code_export_directory.mkdir(parents=True, exist_ok=True)
        self.solver, self.model, self.parameter_values = build_formal_ocp(
            AcadosBuildSpec(parameters, config, str(self.code_export_directory))
        )
        self.diagnostics = SEPCommandDiagnostics(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)
        self._initialized = False
        self.prediction_slacks_history: list[float] = []
        self.prediction_residuals_history: list[float] = []

    def reset(self, state: np.ndarray | None = None) -> None:
        if state is None:
            state = np.zeros(4)
        state = np.asarray(state, dtype=float)
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("SEP initial state must be finite [ex,ev,alpha,alphadot]")
        for stage in range(self.config.shooting_nodes + 1):
            self.solver.set(stage, "x", state)
        for stage in range(self.config.shooting_nodes):
            self.solver.set(stage, "u", np.zeros(2))
        self._initialized = True
        self.prediction_slacks_history = []
        self.prediction_residuals_history = []

    def command(self, state: np.ndarray, reference_preview: dict[str, np.ndarray], previous_applied_ax: float) -> float:
        z = np.asarray(state, dtype=float)
        if z.shape != (4,) or not np.isfinite(z).all():
            raise ValueError("SEP state must be finite [ex,ev,alpha,alphadot]")
        if set(reference_preview) != {"x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref"}:
            raise ValueError("reference preview must contain all six reference arrays")
        for value in reference_preview.values():
            if np.asarray(value).shape != (self.config.shooting_nodes + 1,):
                raise ValueError("reference preview requires 41 boundary samples")
        if not self._initialized:
            self.reset(z)
        self.solver.set(0, "lbx", z)
        self.solver.set(0, "ubx", z)
        p = self.parameter_values.copy()
        p[1] = float(previous_applied_ax)
        for stage in range(self.config.shooting_nodes):
            p[0] = float(reference_preview["ax_ref"][stage])
            self.solver.set(stage, "p", p)
        p[-1] = self.parameters.m_L * self.parameters.l
        started = time.perf_counter_ns()
        status = int(self.solver.solve())
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if status != 0:
            raise RuntimeError(f"acados SQP_RTI status {status}")
        u0 = np.asarray(self.solver.get(0, "u"), dtype=float).reshape(2)
        u_ae, slack = float(u0[0]), float(u0[1])
        ax_ref = float(reference_preview["ax_ref"][0])
        ax_raw = ax_ref + (u_ae - self.config.k_e * z[0]) / self.parameters.m_T
        residual = u_ae * z[1] + self.config.rho * z[1] ** 2 + self.config.epsilon * u_ae ** 2 - slack
        for stage in range(self.config.shooting_nodes):
            z_stage = np.asarray(self.solver.get(stage, "x"), dtype=float).reshape(4)
            u_stage = np.asarray(self.solver.get(stage, "u"), dtype=float).reshape(2)
            stage_residual = u_stage[0] * z_stage[1] + self.config.rho * z_stage[1] ** 2 + self.config.epsilon * u_stage[0] ** 2 - u_stage[1]
            self.prediction_slacks_history.append(float(u_stage[1]))
            self.prediction_residuals_history.append(float(stage_residual))
        self.diagnostics = SEPCommandDiagnostics(
            u_ae, slack, ax_raw, float(residual), status,
            float(np.asarray(self.solver.get_stats("qp_iter"), dtype=float).reshape(-1)[0]) if self._has_stat("qp_iter") else 0.0,
            float(elapsed_ms), float(abs(ax_raw - float(previous_applied_ax)) - 0.25),
        )
        return ax_raw

    def _has_stat(self, name: str) -> bool:
        try:
            self.solver.get_stats(name)
            return True
        except Exception:
            return False
