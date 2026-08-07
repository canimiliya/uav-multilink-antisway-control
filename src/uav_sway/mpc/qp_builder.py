"""Dense QP construction for the 20-step constrained preview problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QPData:
    P: np.ndarray
    q: np.ndarray
    A: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    affine_states: np.ndarray
    state_maps: np.ndarray


def build_preview_qp(model, x0, references, previous_action: float,
                     position_scale: float, tip_weight: float,
                     ax_min: float = -2.0, ax_max: float = 2.0,
                     slew_limit: float = 0.25, disturbance: float = 0.0) -> QPData:
    H = model.horizon_steps
    n = 16
    x0 = np.asarray(x0, dtype=float).reshape(n)
    # x_k = f_k + G_k u, where u is delta-a_x and action ref is an affine term.
    # The physical input matrix has an explicit positive sign in the model.
    f = np.zeros((H + 1, n), dtype=float); f[0] = x0
    G = np.zeros((H + 1, n, H), dtype=float)
    Q = model.Q.copy()
    Q[0, 0] *= float(position_scale); Q[1, 1] *= float(position_scale)
    W = Q + float(tip_weight) * (model.C_tip.T @ model.C_tip)
    for k in range(H):
        c = model.reference_shift(references[k], references[k + 1])
        f[k + 1] = model.A @ f[k] + model.B[:, 0] * (float(references[k].ax_ref) + float(disturbance)) - c
        G[k + 1] = model.A @ G[k]
        G[k + 1, :, k] += model.B[:, 0]
    P = np.zeros((H, H), dtype=float); q = np.zeros(H, dtype=float)
    stage_weight = W
    for k in range(1, H):
        P += 2.0 * (G[k].T @ stage_weight @ G[k])
        q += 2.0 * (G[k].T @ stage_weight @ f[k])
    terminal_W = float(tip_weight) * (model.C_tip.T @ model.C_tip) + model.P
    P += 2.0 * (G[H].T @ terminal_W @ G[H])
    q += 2.0 * (G[H].T @ terminal_W @ f[H])
    # OSQP uses 1/2*u'Pu + q'u.  Keep the control cost in every input,
    # including the first constrained move.
    P += 2.0 * float(getattr(model, "control_weight", 1.0)) * np.eye(H)
    P = (P + P.T) / 2.0
    rows = []; lower = []; upper = []
    for k in range(H):
        row = np.zeros(H); row[k] = 1.0; rows.append(row)
        lower.append(float(ax_min - references[k].ax_ref)); upper.append(float(ax_max - references[k].ax_ref))
        row = np.zeros(H); row[k] = 1.0
        if k > 0: row[k - 1] = -1.0
        rows.append(row)
        if k == 0:
            # |(ax_ref + delta_0) - previous_action| <= slew_limit.
            lower.append(float(previous_action - references[k].ax_ref - slew_limit))
            upper.append(float(previous_action - references[k].ax_ref + slew_limit))
        else:
            lower.append(float(-slew_limit - references[k].ax_ref + references[k - 1].ax_ref))
            upper.append(float(slew_limit - references[k].ax_ref + references[k - 1].ax_ref))
    return QPData(P, q, np.asarray(rows), np.asarray(lower), np.asarray(upper), f, G)
