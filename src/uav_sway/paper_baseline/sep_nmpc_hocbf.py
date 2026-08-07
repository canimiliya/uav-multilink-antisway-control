"""HOCBF interface retained from SEP-NMPC, with empty-obstacle parity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HOCBFConstraintRows:
    """Affine rows A u >= b for relative-degree-two obstacle constraints."""

    A: np.ndarray
    b: np.ndarray
    active: bool


def build_hocbf_constraints(state, obstacles, *, kappa_1: float = 1.0, kappa_2: float = 1.0) -> HOCBFConstraintRows:
    """Build HOCBF rows or return zero rows for the frozen empty set.

    The obstacle-specific acceleration decomposition follows paper Eq.
    (16)-(21); the current project intentionally does not fabricate obstacle
    rows.  Non-empty obstacle support remains an explicit future extension.
    """

    del state
    if kappa_1 <= 0 or kappa_2 <= 0:
        raise ValueError("HOCBF gains must be positive")
    if obstacles is None:
        obstacles = []
    if len(obstacles) == 0:
        return HOCBFConstraintRows(np.empty((0, 1), dtype=float), np.empty((0,), dtype=float), False)
    raise NotImplementedError(
        "non-empty obstacle rows require the paper's p_j, f_v,j, and G_v data; "
        "the frozen benchmark has no obstacle set"
    )
