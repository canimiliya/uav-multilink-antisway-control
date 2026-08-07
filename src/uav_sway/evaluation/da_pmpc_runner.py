"""Free-flight DA-PMPC runner for the two S5A development scenes."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from uav_sway.control.base import ReferenceState
from uav_sway.control.da_pmpc import DAPMPC
from uav_sway.control.geometric_inner_loop import GeometricInnerLoop
from uav_sway.control.state_reader import StateReader
from uav_sway.disturbances.aerodynamics import load_aerodynamic_config
from uav_sway.disturbances.wind_applier import clear_and_apply_wind
from uav_sway.disturbances.wind_io import read_wind_csv
from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics
from uav_sway.evaluation.da_pmpc_gate import raw_da_pmpc_gate
from uav_sway.models.model_config import load_model_config
from uav_sway.mppi.reference_horizon import make_reference_horizon
from uav_sway.linearization.reduced_state import ReducedStateLayout


ROOT = Path(__file__).resolve().parents[3]


def _id(model, typ, name):
    value=int(mujoco.mj_name2id(model, typ, name))
    if value<0: raise KeyError(name)
    return value


def _read_csv(path):
    with Path(path).open("r",encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    return {k:np.asarray([float(r[k]) for r in rows],dtype=float) for k in ("time","x_ref","vx_ref","ax_ref","y_ref","z_ref","yaw_ref")}, rows


def _ref(ref, i): return ReferenceState(*(float(ref[k][i]) for k in ("x_ref","vx_ref","ax_ref","y_ref","z_ref","yaw_ref")))


def _rpy(R): return (float(np.arctan2(R[2,1],R[2,2])),float(np.arcsin(np.clip(-R[2,0],-1,1))),float(np.arctan2(R[1,0],R[0,0])))


def _write(path, rows, columns):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=columns,lineterminator="\n"); w.writeheader()
        for row in rows:
            out={}
            for c in columns:
                v=row[c]
                if isinstance(v,(bool,np.bool_)): out[c]="true" if v else "false"
                elif isinstance(v,str): out[c]=v
                else: out[c]=format(float(v),".17g")
            w.writerow(out)


def schema_columns(n_links=5):
    base=["time","scenario","protocol_mode","wind_x","x_ref","vx_ref","ax_ref","y_ref","z_ref","yaw_ref","uav_x","uav_y","uav_z","uav_vx","uav_vy","uav_vz","tip_x","tip_y","tip_z","tip_displacement"]
    joints=[x for i in range(1,n_links+1) for x in (f"joint_{i}_angle",f"joint_{i}_velocity")]
    return base+joints+["ax_cmd_raw","ax_cmd_limited","ax_saturated","ax_slew_limited","anchor_active","roll_rad","pitch_rad","yaw_rad","thrust_cmd_raw_N","thrust_cmd_limited_N","mx_cmd_raw_Nm","my_cmd_raw_Nm","mz_cmd_raw_Nm","mx_cmd_limited_Nm","my_cmd_limited_Nm","mz_cmd_limited_Nm","rotor_motor_max_abs_cmd","solve_time_ms","controller","position_error_x","velocity_error_x","disturbance_hat","qp_limiter_mismatch","qp_status_code","qp_iterations","preview_horizon_steps","tip_weight","position_scale"]


def run_scene(model_config_path, da_config, scene, wind_path, reference_path, output_csv, temperature=None, duration_s=12.0):
    del temperature
    root=ROOT; model=mujoco.MjModel.from_xml_path(str(root/"artifacts/s3/runtime/model_5link_controlled.xml")); cfg=load_model_config(model_config_path); aero=load_aerodynamic_config(root/"configs/aerodynamics.yaml")
    ref,_=_read_csv(reference_path); wind=read_wind_csv(wind_path); data=mujoco.MjData(model); data.qpos[:7]=[0,0,3.2,1,0,0,0]; data.qvel[:]=0; data.ctrl[:]=0; data.eq_active[:]=0; mujoco.mj_forward(model,data)
    quad=_id(model,mujoco.mjtObj.mjOBJ_BODY,"quadrotor"); tip=_id(model,mujoco.mjtObj.mjOBJ_SITE,"cutter_tip"); relative=float(data.site_xpos[tip,0]-data.xpos[quad,0]); reader=StateReader(model,cfg.n_links,relative); layout=ReducedStateLayout(model)
    q=np.load(root/"artifacts/s4/lqr/Q.npy"); B=np.load(root/"artifacts/s4/linearization/B.npy"); A=np.load(root/"artifacts/s4/linearization/A.npy"); K=np.load(root/"artifacts/s4/lqr/K.npy"); R=float(np.load(root/"artifacts/s4/lqr/R.npy").reshape(-1)[0])
    from scipy.linalg import solve_discrete_are
    P=solve_discrete_are(A,B,q,np.load(root/"artifacts/s4/lqr/R.npy"))
    C=np.load(root/"artifacts/s5a/model/C_tip.npy") if (root/"artifacts/s5a/model/C_tip.npy").exists() else np.zeros((1,16)); total=float(np.sum(model.body_mass)); s3=yaml.safe_load((root/"configs/s3_pid.yaml").read_text(encoding="utf-8")); inner=GeometricInnerLoop(total,np.asarray(model.body_inertia[quad]),s3["attitude_natural_frequency_rad_s"],s3["attitude_damping_ratio"],*s3["position_gains_y"],*s3["position_gains_z"])
    from uav_sway.mpc.osqp_solver import OSQPPreviewSolver
    from uav_sway.control.disturbance_observer import MatchedDisturbanceObserver
    mpc=DAPMPC(A,B,q,P,C,float(da_config.get("selected_position_scale",1.0)),float(da_config.get("selected_tip_weight",40.0)),OSQPPreviewSolver(da_config["osqp_eps_abs"],da_config["osqp_eps_rel"],da_config["osqp_max_iter"],da_config["osqp_warm_start"]),MatchedDisturbanceObserver(A,B,da_config["disturbance_observer_gain"],da_config["disturbance_limit_m_s2"]),da_config["horizon_steps"],da_config["ax_min_m_s2"],da_config["ax_max_m_s2"],da_config["ax_slew_limit_m_s2_per_update"],R); mpc.reset()
    aid={n:_id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ("rotor_motor_0","rotor_motor_1","rotor_motor_2","rotor_motor_3","thrust_motor","mx_motor","my_motor","mz_motor")}; pdt=float(model.opt.timestep); inner_steps=5; outer_steps=50; steps=int(round(duration_s/pdt)); rows=[]; last_ax=0.; outer=logs=winds=0; last_solve=0.; force={"quadrotor_x":0.,"cutter_x":0.,"total_x":0.,**{f"link_{i}_x":0. for i in range(1,6)}}; last_result=None
    for step in range(steps+1):
        idx=min(step//inner_steps,len(wind["time"])-1); force=clear_and_apply_wind(model,data,cfg,aero,float(wind["wind_x"][idx])); winds+=1; reference=_ref(ref,idx)
        if step%outer_steps==0:
            h=make_reference_horizon(ref,idx,da_config["horizon_steps"]); state=layout.extract(model,data,reference); t0=time.perf_counter_ns(); last_ax=mpc.command(state,h); last_solve=(time.perf_counter_ns()-t0)/1e6; outer+=1
        if step%inner_steps==0:
            state=reader.read(model,data); result=inner.compute(state,reference,last_ax); tr=float(result["thrust_raw_N"]); tq=np.asarray(result["torque_raw_Nm"]); tl=np.asarray([np.clip(tq[i],*model.actuator_ctrlrange[aid[n]]) for i,n in enumerate(("mx_motor","my_motor","mz_motor"))]); thr=float(np.clip(tr,*model.actuator_ctrlrange[aid["thrust_motor"]])); data.ctrl[:]=0; data.ctrl[aid["thrust_motor"]]=thr; [data.ctrl.__setitem__(aid[n],tl[i]) for i,n in enumerate(("mx_motor","my_motor","mz_motor"))]; roll,pitch,yaw=_rpy(np.asarray(data.xmat[quad]).reshape(3,3)); d=mpc.diagnostics
            rows.append({"time":float(wind["time"][idx]),"scenario":scene,"protocol_mode":"free_flight_controlled","wind_x":float(wind["wind_x"][idx]),"x_ref":reference.x_ref,"vx_ref":reference.vx_ref,"ax_ref":reference.ax_ref,"y_ref":reference.y_ref,"z_ref":reference.z_ref,"yaw_ref":reference.yaw_ref,"uav_x":state.position[0],"uav_y":state.position[1],"uav_z":state.position[2],"uav_vx":state.velocity[0],"uav_vy":state.velocity[1],"uav_vz":state.velocity[2],"tip_x":data.site_xpos[tip,0],"tip_y":data.site_xpos[tip,1],"tip_z":data.site_xpos[tip,2],"tip_displacement":state.tip_displacement,**{f"joint_{i}_angle":state.joint_angles[i-1] for i in range(1,6)},**{f"joint_{i}_velocity":state.joint_velocities[i-1] for i in range(1,6)},"ax_cmd_raw":d.ax_cmd_raw,"ax_cmd_limited":d.ax_cmd_limited,"ax_saturated":bool(d.ax_cmd_raw!=d.ax_cmd_limited),"ax_slew_limited":bool(mpc.limiter.diagnostics.slew_limited),"anchor_active":False,"roll_rad":roll,"pitch_rad":pitch,"yaw_rad":yaw,"thrust_cmd_raw_N":tr,"thrust_cmd_limited_N":thr,"mx_cmd_raw_Nm":tq[0],"my_cmd_raw_Nm":tq[1],"mz_cmd_raw_Nm":tq[2],"mx_cmd_limited_Nm":tl[0],"my_cmd_limited_Nm":tl[1],"mz_cmd_limited_Nm":tl[2],"rotor_motor_max_abs_cmd":0.,"solve_time_ms":last_solve,"controller":"da_pmpc","position_error_x":state.position[0]-reference.x_ref,"velocity_error_x":state.velocity[0]-reference.vx_ref,"disturbance_hat":d.disturbance_hat,"qp_limiter_mismatch":d.qp_limiter_mismatch,"qp_status_code":1.0 if d.status.startswith("solved") else 0.0,"qp_iterations":d.iterations,"preview_horizon_steps":da_config["horizon_steps"],"tip_weight":mpc.tip_weight,"position_scale":mpc.position_scale}); logs+=1
        if step<steps: mujoco.mj_step(model,data)
    _write(output_csv,rows,schema_columns(5)); metric=compute_controlled_metrics(output_csv,float(da_config["settling_start_s"][scene])); metric.update({"controller":"da_pmpc","formal_log_samples":logs,"outer_control_updates":outer,"wind_force_calls":winds,"anchor_active":False,"physics_intervals":steps,"final_d_hat":float(rows[-1]["disturbance_hat"]),"max_abs_d_hat":float(max(abs(float(r["disturbance_hat"])) for r in rows)),"mean_abs_raw_ax":float(np.mean([abs(float(r["ax_cmd_raw"])) for r in rows])),"mean_abs_limited_ax":float(np.mean([abs(float(r["ax_cmd_limited"])) for r in rows])),"qp_limiter_mismatch_max":float(max(float(r["qp_limiter_mismatch"]) for r in rows))}); return metric


__all__=["run_scene","raw_da_pmpc_gate","schema_columns"]
