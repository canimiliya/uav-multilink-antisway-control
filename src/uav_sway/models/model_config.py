"""Validated YAML configuration for the generated planar chain."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class ModelConfig:
    n_links: int
    total_length: float
    hinge_axis: tuple[float, float, float]
    hinge_damping: float
    hinge_frictionloss: float
    joint_range_deg: tuple[float, float]

    @property
    def link_length(self) -> float:
        return self.total_length / self.n_links

    @property
    def joint_range_rad(self) -> tuple[float, float]:
        return tuple(float(np.deg2rad(x)) for x in self.joint_range_deg)

    def validate(self) -> None:
        if self.n_links not in (4, 5, 6):
            raise ValueError("n_links must be 4, 5, or 6")
        if self.total_length <= 0:
            raise ValueError("total_length must be positive")
        axis = np.asarray(self.hinge_axis, dtype=float)
        if not np.allclose(axis, [0.0, 1.0, 0.0]):
            raise ValueError("hinge_axis is frozen to [0, 1, 0]")
        if self.hinge_damping < 0 or self.hinge_frictionloss < 0:
            raise ValueError("joint dissipation values must be non-negative")
        if self.joint_range_deg[0] >= self.joint_range_deg[1]:
            raise ValueError("joint_range_deg must be increasing")


def load_model_config(path: str | Path) -> ModelConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    config = ModelConfig(
        n_links=int(raw["n_links"]),
        total_length=float(raw["total_length"]),
        hinge_axis=tuple(float(x) for x in raw["hinge_axis"]),
        hinge_damping=float(raw["hinge_damping"]),
        hinge_frictionloss=float(raw["hinge_frictionloss"]),
        joint_range_deg=tuple(float(x) for x in raw["joint_range_deg"]),
    )
    config.validate()
    return config
