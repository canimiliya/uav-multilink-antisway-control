"""Uncontrolled, wind-free passive simulation for the generated model."""

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from .build_planar_chain import build_planar_chain_model
from .model_config import load_model_config


def _named_id(model, object_type, name: str) -> int:
    index = mujoco.mj_name2id(model, object_type, name)
    if index < 0:
        raise KeyError(f"MuJoCo name not found: {name}")
    return index


def _energy(model, data) -> tuple[float, float, float]:
    mujoco.mj_energyPos(model, data)
    potential = float(data.energy[0])
    mass_matrix = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, mass_matrix, data.qM)
    kinetic = float(0.5 * data.qvel @ mass_matrix @ data.qvel)
    return potential, kinetic, potential + kinetic


def _activate_anchor(model, data) -> None:
    equality_id = _named_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "passive_anchor")
    data.eq_active[:] = 0
    data.eq_active[equality_id] = 1


def _set_initial_state(model, data, initial_angle_deg: float) -> None:
    mujoco.mj_resetData(model, data)
    quad_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    data.qpos[:7] = np.array([0.0, 0.0, 3.2, 1.0, 0.0, 0.0, 0.0])
    joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_1")
    data.qpos[model.jnt_qposadr[joint_id]] = np.deg2rad(initial_angle_deg)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    _activate_anchor(model, data)
    mujoco.mj_forward(model, data)
    # A name lookup is intentional: no fixed body ID is used for the quadrotor.
    assert quad_id == _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")


def _frame(model, data) -> np.ndarray:
    renderer = mujoco.Renderer(model, height=720, width=960)
    renderer.update_scene(data, camera="main_camera")
    image = renderer.render()
    del renderer
    return image


def simulate_passive(
    config_path: str | Path,
    initial_angle_deg: float,
    duration: float,
    csv_path: str | Path | None = None,
    render_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict:
    """Run a passive simulation with a model-level quadrotor weld anchor."""
    config = load_model_config(config_path)
    output_xml = Path(model_path) if model_path is not None else Path(config_path).parent / f"model_{config.n_links}link.xml"
    if not output_xml.exists():
        build_planar_chain_model(config_path, output_xml)
    model = mujoco.MjModel.from_xml_path(str(output_xml))
    data = mujoco.MjData(model)
    _set_initial_state(model, data, initial_angle_deg)
    initial_frame = _frame(model, data) if render_path else None

    joint_ids = [_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}") for i in range(1, config.n_links + 1)]
    joint_qpos = [model.jnt_qposadr[index] for index in joint_ids]
    joint_qvel = [model.jnt_dofadr[index] for index in joint_ids]
    tip_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip")
    quad_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    sample_every = max(1, int(round(0.01 / model.opt.timestep)))
    rows: list[list[float]] = []
    steps = int(round(duration / model.opt.timestep))
    for step in range(steps + 1):
        if step % sample_every == 0 or step == steps:
            potential, kinetic, total = _energy(model, data)
            tip = np.array(data.site_xpos[tip_id], copy=True)
            quad = np.array(data.xpos[quad_id], copy=True)
            angles = [float(data.qpos[index]) for index in joint_qpos]
            velocities = [float(data.qvel[index]) for index in joint_qvel]
            rows.append([
                float(data.time), *angles, *velocities, float(tip[0]), float(tip[1]), float(tip[2]),
                float(tip[0] - quad[0]), potential, kinetic, total,
            ])
        if step < steps:
            mujoco.mj_step(model, data)

    values = np.asarray(rows, dtype=float)
    tip_x = values[:, 1 + config.n_links * 2]
    total_energy = values[:, -1]
    initial_rms = float(np.sqrt(np.mean((tip_x[: max(1, int(2 / 0.01))] - tip_x[0]) ** 2)))
    final_rms = float(np.sqrt(np.mean((tip_x[-max(1, int(2 / 0.01)):] - tip_x[-1]) ** 2)))
    result = {
        "n_links": config.n_links,
        "duration_s": float(duration),
        "initial_angle_deg": float(initial_angle_deg),
        "initial_tip_rms_m": initial_rms,
        "final_tip_rms_m": final_rms,
        "decay_ratio": final_rms / initial_rms if initial_rms > 0 else 0.0,
        "initial_total_energy_j": float(total_energy[0]),
        "final_total_energy_j": float(total_energy[-1]),
        "max_abs_joint_angle_rad": float(np.max(np.abs(values[:, 1 : 1 + config.n_links]))),
        "max_abs_joint_velocity_rad_s": float(np.max(np.abs(values[:, 1 + config.n_links : 1 + 2 * config.n_links]))),
        "min_tip_z_m": float(np.min(values[:, 1 + 2 * config.n_links + 2])),
        "finite": bool(np.isfinite(values).all()),
    }
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        columns = ["time"] + [f"joint_{i}_angle" for i in range(1, config.n_links + 1)]
        columns += [f"joint_{i}_velocity" for i in range(1, config.n_links + 1)]
        columns += ["tip_x", "tip_y", "tip_z", "tip_relative_x", "potential_energy", "kinetic_energy", "total_energy"]
        np.savetxt(csv_path, values, delimiter=",", header=",".join(columns), comments="", fmt="%.12g")
    if render_path is not None and initial_frame is not None:
        render_path = Path(render_path)
        render_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(render_path, initial_frame)
    return result
