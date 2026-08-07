"""MPPI outer controller backed by independent nonlinear MuJoCo rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from uav_sway.control.acceleration_limiter import AccelerationLimiter
from uav_sway.control.base import ReferenceState
from uav_sway.control.geometric_inner_loop import GeometricInnerLoop
from uav_sway.control.state_reader import StateReader
from uav_sway.linearization.reduced_state import ReducedStateLayout
from uav_sway.models.state_io import capture_state
from uav_sway.mppi.reference_horizon import ReferenceHorizon
from uav_sway.mppi.cost import candidate_acceleration
from uav_sway.mppi.rollout_engine import NonlinearRolloutEngine
from uav_sway.mppi.sampler import MPPIUpdate, stable_mppi_update


@dataclass(frozen=True)
class MPPIDiagnostics:
    nominal_first: float
    cost_min: float
    cost_mean: float
    cost_std: float
    weight_max: float
    effective_sample_size: float
    invalid_rollouts: int
    rollout_physics_steps: int
    rollout_calls: int


class MuJoCoMPPI:
    def __init__(self, model, q: np.ndarray, r: np.ndarray,
                 total_mass: float, inertia: np.ndarray, n_links: int,
                 equilibrium_relative_x: float, temperature: float,
                 noise_sigma: float, seed: int, horizon_steps: int = 12,
                 num_rollouts: int = 64, tip_weight: float = 80.0,
                 terminal_multiplier: float = 5.0, ax_min: float = -2.0,
                 ax_max: float = 2.0, slew_limit: float = 0.25,
                 model_config=None, aerodynamic_config=None):
        self.model = model
        self.q = np.asarray(q, dtype=float).copy()
        self.r = np.asarray(r, dtype=float).copy()
        self.horizon_steps = int(horizon_steps)
        self.num_rollouts = int(num_rollouts)
        self.temperature = float(temperature)
        self.noise_sigma = float(noise_sigma)
        self.tip_weight = float(tip_weight)
        self.terminal_multiplier = float(terminal_multiplier)
        self.rng = np.random.Generator(np.random.PCG64(int(seed)))
        self.nominal = np.zeros(self.horizon_steps, dtype=float)
        self.limiter = AccelerationLimiter(ax_min, ax_max, slew_limit)
        self.layout = ReducedStateLayout(model)
        self.reader = StateReader(model, n_links, equilibrium_relative_x)
        self.inner = GeometricInnerLoop(total_mass, np.asarray(inertia), 4.0, 0.9,
                                        1.5, 2.0, 4.0, 3.5)
        self.engine = NonlinearRolloutEngine(
            model, self.q, self.r, self.layout, self.reader, self.inner,
            n_links, equilibrium_relative_x, tip_weight, terminal_multiplier,
            ax_min, ax_max, slew_limit,
            model_config=model_config, aerodynamic_config=aerodynamic_config,
        )
        self.last_update: MPPIUpdate | None = None
        self.diagnostics = MPPIDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)

    def reset(self, value: float = 0.0) -> None:
        self.nominal[:] = 0.0
        self.limiter.reset(value)
        self.last_update = None
        self.diagnostics = MPPIDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)

    def solve(self, data, horizon: ReferenceHorizon) -> float:
        snapshot = capture_state(self.model, data)
        noise = self.rng.normal(0.0, self.noise_sigma,
                                size=(self.num_rollouts, self.horizon_steps))
        costs = np.empty(self.num_rollouts, dtype=float)
        invalid = 0
        for k in range(self.num_rollouts):
            costs[k], valid = self.engine.rollout(snapshot, self.nominal + noise[k],
                                                   horizon, self.limiter.previous)
            if not valid:
                invalid += 1
        update = stable_mppi_update(self.nominal, noise, costs, self.temperature)
        self.last_update = update
        self.nominal[:-1] = update.updated_sequence[1:]
        self.nominal[-1] = 0.0
        correction = float(update.updated_sequence[0])
        # The real plant command uses r0, not the first preview state r1.
        command = self.limiter.limit(candidate_acceleration(horizon.action_reference(0).ax_ref, correction))
        self.diagnostics = MPPIDiagnostics(
            correction, update.cost_min, update.cost_mean, update.cost_std,
            update.weight_max, update.effective_sample_size, invalid,
            self.engine.physics_steps_per_rollout, self.num_rollouts,
        )
        return command
