"""Condensed OSQP residual MPC for the frozen augmented task model."""

from __future__ import annotations

import time

import numpy as np
import osqp
from scipy import sparse


class TaskResidualQP:
    """Solve for residual ``v`` with ``a = -K_I z - v``.

    The condensed prediction keeps the paper/project linear dynamics unchanged.
    Actual acceleration amplitude and inter-stage slew constraints are imposed
    directly on the reconstructed acceleration sequence.
    """

    def __init__(self, A_I: np.ndarray, B_I: np.ndarray, K_I: np.ndarray,
                 Q_I: np.ndarray, P_I: np.ndarray, horizon_steps: int,
                 residual_weight: float, ax_min: float = -2.0,
                 ax_max: float = 2.0, slew_limit: float = 0.25):
        self.A = np.asarray(A_I, dtype=float).reshape(17, 17)
        self.B = np.asarray(B_I, dtype=float).reshape(17, 1)
        self.K = np.asarray(K_I, dtype=float).reshape(1, 17)
        self.Q = np.asarray(Q_I, dtype=float).reshape(17, 17)
        self.P = np.asarray(P_I, dtype=float).reshape(17, 17)
        self.horizon_steps = int(horizon_steps)
        self.residual_weight = float(residual_weight)
        self.ax_min = float(ax_min); self.ax_max = float(ax_max)
        self.slew_limit = float(slew_limit)
        if self.horizon_steps < 1 or self.residual_weight <= 0.0:
            raise ValueError("invalid residual QP settings")
        self._build_prediction_matrices()
        self._solver = osqp.OSQP()
        self._solver.setup(P=sparse.csc_matrix(self._P), q=np.zeros(self.horizon_steps),
                           A=sparse.csc_matrix(self._constraint_matrix),
                           l=np.zeros(self._constraint_matrix.shape[0]),
                           u=np.zeros(self._constraint_matrix.shape[0]),
                           warm_start=True, verbose=False, polish=False,
                           eps_abs=1.0e-7, eps_rel=1.0e-7, max_iter=4000)
        self.last = {"status": "not_solved", "status_val": -1, "solve_time_ms": 0.0,
                     "v": 0.0, "predicted_first_action": 0.0,
                     "passivity": None}

    def _build_prediction_matrices(self) -> None:
        n, h = 17, self.horizon_steps
        d = np.zeros((h + 1, n, n)); d[0] = np.eye(n)
        g = np.zeros((h + 1, n, h))
        for k in range(h):
            d[k + 1] = self.A @ d[k]
            g[k + 1] = self.A @ g[k]
            g[k + 1, :, k] += self.B[:, 0]
        self._d = d
        self._g = g
        pmat = np.zeros((h, h));
        q_const = np.zeros((h, n))
        for k in range(h):
            pmat += g[k].T @ self.Q @ g[k]
            q_const += g[k].T @ self.Q @ d[k]
        pmat += g[h].T @ self.P @ g[h]
        q_const += g[h].T @ self.P @ d[h]
        pmat += self.residual_weight * np.eye(h)
        self._P = 2.0 * (pmat + 1.0e-10 * np.eye(h))
        self._q_const = 2.0 * q_const
        # a_k = c_k(z) + D_k v; first row is the current action.
        action_d = np.vstack([-self.K @ d[k] for k in range(h)])
        action_g = np.vstack([-self.K @ g[k] for k in range(h)])
        action_g[np.arange(h), np.arange(h)] -= 1.0
        self._action_d = action_d
        self._action_g = action_g
        slew_matrix = np.zeros((h, h)); slew_matrix[0] = action_g[0]
        slew_matrix[1:] = action_g[1:] - action_g[:-1]
        self._slew_g = slew_matrix
        self._constraint_matrix = np.vstack((action_g, slew_matrix))

    def solve(self, z: np.ndarray, previous_applied_ax: float) -> dict:
        z = np.asarray(z, dtype=float).reshape(17)
        if not np.isfinite(z).all() or not np.isfinite(previous_applied_ax):
            raise ValueError("QP state must be finite")
        q = self._q_const @ z
        action_c = self._action_d @ z
        slew_c = np.empty(self.horizon_steps, dtype=float)
        slew_c[0] = action_c[0] - previous_applied_ax
        if self.horizon_steps > 1:
            slew_c[1:] = action_c[1:] - action_c[:-1]
        lower = np.concatenate((np.full(self.horizon_steps, self.ax_min) - action_c,
                                -self.slew_limit - slew_c))
        upper = np.concatenate((np.full(self.horizon_steps, self.ax_max) - action_c,
                                self.slew_limit - slew_c))
        self._solver.update(q=q, l=lower, u=upper)
        started = time.perf_counter_ns()
        result = self._solver.solve()
        elapsed = (time.perf_counter_ns() - started) / 1.0e6
        status = str(result.info.status)
        status_val = int(result.info.status_val)
        if result.x is None or status_val not in (1, 2):
            self.last = {"status": status, "status_val": status_val,
                         "solve_time_ms": elapsed, "v": 0.0,
                         "predicted_first_action": float(action_c[0]),
                         "passivity": None}
            return self.last.copy()
        v = np.asarray(result.x, dtype=float)
        actions = action_c + self._action_g @ v
        self.last = {"status": status, "status_val": status_val,
                     "solve_time_ms": elapsed, "v": float(v[0]),
                     "v_sequence": v, "action_sequence": actions,
                     "predicted_first_action": float(actions[0]),
                     "passivity": None}
        return self.last.copy()
