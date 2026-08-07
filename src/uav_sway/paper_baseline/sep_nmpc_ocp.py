"""CasADi/acados-facing OCP metadata without a benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sep_nmpc_model import PlanarParameters


@dataclass(frozen=True)
class SEPTrackingConfig:
    """All development-only numbers frozen before performance evaluation."""

    horizon_seconds: float = 2.0
    shooting_nodes: int = 40
    k_e: float = 10.0
    rho: float = 0.05
    epsilon: float = 0.005
    slack_max: float = 5.0
    w_x: float = 20.0
    w_v: float = 4.0
    w_alpha: float = 40.0
    w_omega: float = 6.0
    w_u: float = 1.0
    w_s: float = 1e5

    @property
    def dt(self) -> float:
        return self.horizon_seconds / self.shooting_nodes

    def validate(self) -> None:
        if self.horizon_seconds != 2.0 or self.shooting_nodes != 40:
            raise ValueError("SEP horizon must remain T=2.0 s, N=40")
        if self.slack_max != 5.0:
            raise ValueError("s_p_max is frozen at 5.0")
        if min(self.k_e, self.rho, self.epsilon) <= 0:
            raise ValueError("SEP tuning values must be positive")
        if min(self.w_x, self.w_v, self.w_alpha, self.w_omega, self.w_u, self.w_s) <= 0:
            raise ValueError("OCP weights must be positive")


def frozen_parameter_grid() -> list[SEPTrackingConfig]:
    return [
        SEPTrackingConfig(k_e=k_e, rho=rho, epsilon=epsilon)
        for k_e in (10.0, 40.0)
        for rho in (0.05, 0.20)
        for epsilon in (0.005, 0.020)
    ]


def casadi_tracking_rhs(z, u_ae, a_ref, parameters: PlanarParameters, k_e: float = 10.0):
    """Return CasADi symbolic dynamics for the nonlinear tracking model."""

    try:
        import casadi as ca
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("CasADi is required for symbolic SEP construction") from exc
    alpha = z[2]
    alpha_dot = z[3]
    c = ca.cos(alpha)
    s = ca.sin(alpha)
    mass = ca.vertcat(
        ca.horzcat(parameters.m_T, parameters.m_L * parameters.l * c),
        ca.horzcat(parameters.m_L * parameters.l * c, parameters.m_L * parameters.l**2),
    )
    force_x = parameters.m_T * a_ref + u_ae - k_e * z[0]
    rhs = ca.vertcat(
        force_x + parameters.m_L * parameters.l * s * alpha_dot**2,
        -parameters.m_L * parameters.g * parameters.l * s,
    )
    acceleration = ca.solve(mass, rhs)
    return ca.vertcat(z[1], acceleration[0] - a_ref, z[3], acceleration[1])


def build_synthetic_casadi_opti(
    parameters: PlanarParameters,
    config: SEPTrackingConfig,
    initial_state: np.ndarray | None = None,
):
    """Build a one-shot synthetic OCP for environment smoke testing.

    This is intentionally not connected to MuJoCo or any benchmark scenario.
    It uses CasADi's SQP interface on Windows; the WSL acados run is recorded
    separately when the acados toolchain is available.
    """

    try:
        import casadi as ca
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("CasADi is required for the synthetic OCP smoke") from exc
    config.validate()
    z0 = np.asarray(initial_state if initial_state is not None else [0.02, 0.01, 0.03, 0.0], dtype=float)
    if z0.shape != (4,):
        raise ValueError("initial_state must have four entries")
    opti = ca.Opti()
    z = opti.variable(4, config.shooting_nodes + 1)
    u = opti.variable(1, config.shooting_nodes)
    slack = opti.variable(1, config.shooting_nodes)
    opti.subject_to(z[:, 0] == z0)
    opti.subject_to(opti.bounded(0.0, slack, config.slack_max))
    stage_cost = 0
    for k in range(config.shooting_nodes):
        zk = z[:, k]
        uk = u[0, k]
        dynamics = casadi_tracking_rhs(zk, uk, 0.0, parameters, config.k_e)
        opti.subject_to(z[:, k + 1] == zk + config.dt * dynamics)
        opti.subject_to(opti.bounded(-2.0 * parameters.m_T, uk, 2.0 * parameters.m_T))
        opti.subject_to(
            uk * zk[1]
            <= -config.rho * zk[1] ** 2 - config.epsilon * uk**2 + slack[0, k]
        )
        opti.subject_to(zk[2] <= np.deg2rad(60.0))
        opti.subject_to(zk[2] >= -np.deg2rad(60.0))
        stage_cost += (
            config.w_x * zk[0] ** 2
            + config.w_v * zk[1] ** 2
            + config.w_alpha * zk[2] ** 2
            + config.w_omega * zk[3] ** 2
            + config.w_u * uk**2
            + config.w_s * slack[0, k] ** 2
        )
    opti.minimize(stage_cost)
    opti.set_initial(z, np.zeros((4, config.shooting_nodes + 1)))
    opti.set_initial(u, np.zeros((1, config.shooting_nodes)))
    opti.set_initial(slack, np.zeros((1, config.shooting_nodes)))
    opti.solver(
        "sqpmethod",
        {
            "qpsol": "qrqp",
            "print_header": False,
            "print_iteration": False,
            "print_time": False,
        },
    )
    return opti, z, u, slack
