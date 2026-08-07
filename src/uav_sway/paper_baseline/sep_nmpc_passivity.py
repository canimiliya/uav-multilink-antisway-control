"""Tracking-shaped storage and passivity contracts for SEP-NMPC-adapted."""

from __future__ import annotations

import numpy as np

from .sep_nmpc_model import PlanarParameters, planar_mass_matrix


def tracking_errors(x: float, vx: float, x_ref: float, vx_ref: float) -> tuple[float, float]:
    return float(x - x_ref), float(vx - vx_ref)


def paper_storage(
    q_dot: np.ndarray,
    alpha: float,
    e_x: float,
    ke: float,
    parameters: PlanarParameters,
) -> float:
    """The preserved planar form of the paper's original storage function."""

    velocity = np.asarray(q_dot, dtype=float)
    if velocity.shape != (2,):
        raise ValueError("q_dot must contain [x_dot, alpha_dot]")
    if ke < 0:
        raise ValueError("ke must be non-negative")
    kinetic = 0.5 * velocity @ planar_mass_matrix(alpha, parameters) @ velocity
    potential = parameters.m_L * parameters.g * parameters.l * (1.0 - np.cos(alpha))
    return float(kinetic + potential + 0.5 * ke * e_x**2)


def tracking_storage(
    state: np.ndarray,
    e_x: float,
    ke: float,
    parameters: PlanarParameters,
) -> float:
    """V_tr for [e_x,e_v,alpha,alpha_dot]."""

    z = np.asarray(state, dtype=float)
    if z.shape != (4,):
        raise ValueError("state must be [e_x, e_v, alpha, alpha_dot]")
    if ke < 0:
        raise ValueError("ke must be non-negative")
    velocity = np.array([z[1], z[3]], dtype=float)
    kinetic = 0.5 * velocity @ planar_mass_matrix(z[2], parameters) @ velocity
    potential = parameters.m_L * parameters.g * parameters.l * (1.0 - np.cos(z[2]))
    return float(kinetic + potential + 0.5 * ke * float(e_x) ** 2)


def tracking_shaped_input(
    force_x: float,
    force_feedforward: float,
    ke: float,
    e_x: float,
) -> float:
    return float(force_x - force_feedforward + ke * e_x)


def force_from_tracking_shaped_input(
    u_ae: float,
    a_ref: float,
    ke: float,
    e_x: float,
    parameters: PlanarParameters,
) -> float:
    return float(parameters.m_T * a_ref + float(u_ae) - ke * float(e_x))


def acceleration_from_tracking_shaped_input(
    u_ae: float,
    a_ref: float,
    ke: float,
    e_x: float,
    parameters: PlanarParameters,
) -> float:
    return float(a_ref + (float(u_ae) - ke * float(e_x)) / parameters.m_T)


def passivity_residual(u_ae: float, e_v: float, rho: float, epsilon: float, slack: float = 0.0) -> float:
    """Return lhs-rhs of the frozen inequality; <=0 means satisfied."""

    if rho < 0 or epsilon < 0 or slack < 0:
        raise ValueError("rho, epsilon, and slack must be non-negative")
    return float(u_ae * e_v + rho * e_v**2 + epsilon * u_ae**2 - slack)


def passivity_satisfied(u_ae: float, e_v: float, rho: float, epsilon: float, slack: float = 0.0) -> bool:
    return bool(passivity_residual(u_ae, e_v, rho, epsilon, slack) <= 1e-12)
