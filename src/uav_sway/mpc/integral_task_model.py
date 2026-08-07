"""Frozen 16D task model augmentation and Task-LQI construction for S6T3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, solve_discrete_are


Q_S4 = np.diag([80.0, 4.0, 8.0, 2.0, 4.0, 1.0,
                20.0, 20.0, 20.0, 20.0, 20.0,
                12.0, 12.0, 12.0, 12.0, 12.0])
W_TASK = np.diag([80.0, 20.0, 5.0, 1.25])


@dataclass(frozen=True)
class AugmentedTaskModel:
    A: np.ndarray
    B: np.ndarray
    C_task: np.ndarray
    dt: float
    A_I: np.ndarray
    B_I: np.ndarray
    C_p: np.ndarray


def build_augmented_task_model(A: np.ndarray, B: np.ndarray, C_task: np.ndarray,
                               dt: float = 0.05) -> AugmentedTaskModel:
    A = np.asarray(A, dtype=float).reshape(16, 16)
    B = np.asarray(B, dtype=float).reshape(16, 1)
    C_task = np.asarray(C_task, dtype=float).reshape(4, 16)
    dt = float(dt)
    if dt <= 0.0 or not np.isfinite(np.concatenate((A.ravel(), B.ravel(), C_task.ravel()))).all():
        raise ValueError("invalid frozen task model")
    C_p = C_task[0:1, :]
    A_I = np.block([[A, np.zeros((16, 1))], [dt * C_p, np.ones((1, 1))]])
    B_I = np.vstack((B, np.zeros((1, 1))))
    return AugmentedTaskModel(A, B, C_task, dt, A_I, B_I, C_p)


def pbh_stabilizable(A: np.ndarray, B: np.ndarray, tolerance: float = 1.0e-9) -> bool:
    """PBH stabilizability check for all discrete-time unstable modes."""

    eigenvalues = np.linalg.eigvals(A)
    for eigenvalue in eigenvalues:
        if abs(eigenvalue) < 1.0 - tolerance:
            continue
        matrix = np.hstack((eigenvalue * np.eye(A.shape[0]) - A, B))
        if np.linalg.matrix_rank(matrix, tol=tolerance) < A.shape[0]:
            return False
    return True


def build_task_lqi(model: AugmentedTaskModel, q_eta: float,
                   r_value: float = 1.0) -> dict:
    q_eta = float(q_eta)
    if q_eta <= 0.0 or not np.isfinite(q_eta):
        raise ValueError("q_eta must be positive and finite")
    q_task = 0.05 * Q_S4 + model.C_task.T @ W_TASK @ model.C_task
    q_i = block_diag(q_task, np.asarray([[q_eta]], dtype=float))
    r = np.asarray([[float(r_value)]], dtype=float)
    if not pbh_stabilizable(model.A_I, model.B_I):
        raise ValueError("augmented model is not stabilizable")
    p_i = solve_discrete_are(model.A_I, model.B_I, q_i, r)
    gain = np.linalg.solve(r + model.B_I.T @ p_i @ model.B_I, model.B_I.T @ p_i @ model.A_I)
    closed_loop = model.A_I - model.B_I @ gain
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(closed_loop))))
    dare_residual = p_i - (model.A_I.T @ p_i @ model.A_I
                           - model.A_I.T @ p_i @ model.B_I
                           @ np.linalg.solve(r + model.B_I.T @ p_i @ model.B_I,
                                             model.B_I.T @ p_i @ model.A_I)
                           + q_i)
    result = {
        "q_eta": q_eta,
        "Q_task": q_task,
        "Q_I": q_i,
        "R": r,
        "P_I": p_i,
        "K_I": gain,
        "spectral_radius": spectral_radius,
        "dare_residual_norm": float(np.linalg.norm(dare_residual)),
        "pbh_stabilizable": True,
    }
    if not np.isfinite(gain).all() or not np.isfinite(p_i).all() or spectral_radius >= 1.0:
        raise ValueError("Task-LQI solution is not finite and stable")
    return result
