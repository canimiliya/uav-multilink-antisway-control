"""Condensed residual QP for the frozen DOB-Task-LQR model."""

from __future__ import annotations

import time

import numpy as np
import osqp
from scipy import sparse


class DOBTaskResidualQP:
    """Optimize residual ``v`` with physical action ``a=-Kx-d_hat-v``.

    The nominal predictor is ``x+ = (A-BK)x-Bv``.  The observer
    compensation is already in the applied command and is intentionally not
    added a second time in the predictor.
    """

    def __init__(self, A, B, K, Q, P, horizon_steps, residual_weight,
                 ax_min=-2.0, ax_max=2.0, slew_limit=0.25):
        self.A = np.asarray(A, dtype=float).reshape(16, 16)
        self.B = np.asarray(B, dtype=float).reshape(16, 1)
        self.K = np.asarray(K, dtype=float).reshape(1, 16)
        self.Q = np.asarray(Q, dtype=float).reshape(16, 16)
        self.P = np.asarray(P, dtype=float).reshape(16, 16)
        self.horizon_steps = int(horizon_steps)
        self.residual_weight = float(residual_weight)
        self.ax_min = float(ax_min); self.ax_max = float(ax_max)
        self.slew_limit = float(slew_limit)
        if self.horizon_steps < 1 or self.residual_weight <= 0:
            raise ValueError("invalid residual QP settings")
        self._build_prediction_matrices()
        self._solver = osqp.OSQP()
        rows = self._constraint_matrix.shape[0]
        self._solver.setup(P=sparse.csc_matrix(self._P), q=np.zeros(self.horizon_steps),
                           A=sparse.csc_matrix(self._constraint_matrix),
                           l=np.full(rows, -np.inf), u=np.full(rows, np.inf),
                           warm_start=True, verbose=False, polish=False,
                           eps_abs=1.0e-8, eps_rel=1.0e-8, max_iter=4000)
        self.last = {"status": "not_solved", "status_val": -1,
                     "solve_time_ms": 0.0, "v": 0.0,
                     "predicted_first_action": 0.0}

    def _build_prediction_matrices(self):
        h = self.horizon_steps
        acl = self.A - self.B @ self.K
        d = np.zeros((h + 1, 16, 16)); d[0] = np.eye(16)
        g = np.zeros((h + 1, 16, h))
        for k in range(h):
            d[k + 1] = acl @ d[k]
            g[k + 1] = acl @ g[k]
            g[k + 1, :, k] -= self.B[:, 0]
        self._d = d; self._g = g
        pmat = np.zeros((h, h)); qmat = np.zeros((h, 16))
        for k in range(h):
            pmat += g[k].T @ self.Q @ g[k]
            qmat += g[k].T @ self.Q @ d[k]
        pmat += g[h].T @ self.P @ g[h]
        qmat += g[h].T @ self.P @ d[h]
        pmat += self.residual_weight * np.eye(h)
        self._P = 2.0 * (pmat + 1.0e-10 * np.eye(h))
        self._q_const = 2.0 * qmat
        action_d = np.vstack([-self.K @ d[k] for k in range(h)])
        action_g = np.vstack([-self.K @ g[k] for k in range(h)])
        action_g[np.arange(h), np.arange(h)] -= 1.0
        self._action_d = action_d; self._action_g = action_g
        slew = np.zeros((h, h)); slew[0] = action_g[0]
        if h > 1: slew[1:] = action_g[1:] - action_g[:-1]
        self._slew_g = slew
        self._constraint_matrix = np.vstack((action_g, slew))

    def solve(self, state, disturbance_hat, previous_applied_ax):
        x = np.asarray(state, dtype=float).reshape(16)
        d_hat = float(disturbance_hat); prev = float(previous_applied_ax)
        action_c = self._action_d @ x - d_hat
        slew_c = np.empty(self.horizon_steps)
        slew_c[0] = action_c[0] - prev
        if self.horizon_steps > 1: slew_c[1:] = action_c[1:] - action_c[:-1]
        q = self._q_const @ x
        lower = np.r_[np.full(self.horizon_steps, self.ax_min) - action_c,
                      -self.slew_limit - slew_c]
        upper = np.r_[np.full(self.horizon_steps, self.ax_max) - action_c,
                      self.slew_limit - slew_c]
        self._solver.update(q=q, l=lower, u=upper)
        started = time.perf_counter_ns(); result = self._solver.solve()
        elapsed = (time.perf_counter_ns() - started) / 1.0e6
        status = str(result.info.status); status_val = int(result.info.status_val)
        if result.x is None or status_val not in (1, 2):
            self.last = {"status": status, "status_val": status_val,
                         "solve_time_ms": elapsed, "v": 0.0,
                         "predicted_first_action": float(action_c[0])}
            return self.last.copy()
        v = np.asarray(result.x, dtype=float); actions = action_c + self._action_g @ v
        self.last = {"status": status, "status_val": status_val,
                     "solve_time_ms": elapsed, "v": float(v[0]),
                     "v_sequence": v, "action_sequence": actions,
                     "predicted_first_action": float(actions[0])}
        return self.last.copy()

