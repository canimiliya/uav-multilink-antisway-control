"""Planar/x-only specialization of the SEP-NMPC internal model.

The equations here preserve the nonlinear point-payload/massless-cable
specialization of the selected paper.  They are not a model of the actual
five-link MuJoCo plant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanarParameters:
    """Parameters for the paper-equivalent planar model."""

    m_Q: float
    m_L: float
    l: float
    g: float = 9.81

    def __post_init__(self) -> None:
        if self.m_Q <= 0 or self.m_L <= 0 or self.l <= 0 or self.g <= 0:
            raise ValueError("planar parameters must be positive")

    @property
    def m_T(self) -> float:
        return self.m_Q + self.m_L


def planar_mass_matrix(alpha: float, parameters: PlanarParameters) -> np.ndarray:
    """Return the frozen paper specialization M_p(alpha)."""

    c = float(np.cos(alpha))
    return np.array(
        [
            [parameters.m_T, parameters.m_L * parameters.l * c],
            [parameters.m_L * parameters.l * c, parameters.m_L * parameters.l**2],
        ],
        dtype=float,
    )


def planar_generalized_rhs(
    alpha: float,
    alpha_dot: float,
    force_x: float,
    parameters: PlanarParameters,
) -> np.ndarray:
    """Return the right side of M_p q_ddot = rhs.

    This is exactly the rearrangement of
    ``m_T*x_ddot + m_L*l*cos(alpha)*alpha_ddot
    - m_L*l*sin(alpha)*alpha_dot**2 = F_x`` and
    ``m_L*l*cos(alpha)*x_ddot + m_L*l**2*alpha_ddot
    + m_L*g*l*sin(alpha) = 0``.
    """

    return np.array(
        [
            float(force_x)
            + parameters.m_L * parameters.l * np.sin(alpha) * alpha_dot**2,
            -parameters.m_L * parameters.g * parameters.l * np.sin(alpha),
        ],
        dtype=float,
    )


def planar_acceleration(
    alpha: float,
    alpha_dot: float,
    force_x: float,
    parameters: PlanarParameters,
) -> np.ndarray:
    """Solve the nonlinear planar dynamics for ``[x_ddot, alpha_ddot]``."""

    return np.linalg.solve(
        planar_mass_matrix(alpha, parameters),
        planar_generalized_rhs(alpha, alpha_dot, force_x, parameters),
    )


def planar_dynamics(
    state: np.ndarray,
    force_x: float,
    a_ref: float,
    parameters: PlanarParameters,
) -> np.ndarray:
    """Tracking-state dynamics for z=[e_x,e_v,alpha,alpha_dot]."""

    z = np.asarray(state, dtype=float)
    if z.shape != (4,):
        raise ValueError("state must be [e_x, e_v, alpha, alpha_dot]")
    x_ddot, alpha_ddot = planar_acceleration(z[2], z[3], force_x, parameters)
    return np.array([z[1], x_ddot - float(a_ref), z[3], alpha_ddot], dtype=float)
