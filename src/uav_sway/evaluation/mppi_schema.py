"""S5 controlled CSV extension of the frozen S3 contract."""

from __future__ import annotations

from .controlled_schema import controlled_schema_columns


def mppi_schema_columns(n_links: int) -> list[str]:
    return controlled_schema_columns(n_links) + [
        "mppi_seed", "mppi_horizon_steps", "mppi_num_rollouts",
        "mppi_temperature", "mppi_noise_sigma", "mppi_nominal_first",
        "mppi_cost_min", "mppi_cost_mean", "mppi_cost_std",
        "mppi_weight_max", "mppi_effective_sample_size",
        "mppi_invalid_rollouts", "mppi_rollout_physics_steps",
        "mppi_rollout_calls", "rotor_motor_max_abs_cmd",
    ]


def mppi_schema_description(n_links: int) -> dict:
    return {
        "columns": mppi_schema_columns(n_links),
        "protocol_mode": "free_flight_controlled",
        "controller": "mppi",
        "anchor_active": False,
        "rollout_wind_policy": "zero_external_wind_with_static_aerodynamic_drag",
    }
