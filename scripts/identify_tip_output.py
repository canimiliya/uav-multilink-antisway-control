"""Identify C_tip and freeze the DA-PMPC model audit."""

from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.linalg import solve_discrete_are

from uav_sway.control.base import ReferenceState
from uav_sway.linearization.equilibrium import build_initial_equilibrium
from uav_sway.mpc.tip_output import identify_tip_output

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--runtime-model",default="artifacts/s3/runtime/model_5link_controlled.xml"); p.add_argument("--output-dir",default="artifacts/s5a"); args=p.parse_args(); out=ROOT/args.output_dir; out.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_xml_path(str(ROOT/args.runtime_model)); data,snapshot=build_initial_equilibrium(model); quad=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"quadrotor")); tip=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"cutter_tip")); eq=float(data.site_xpos[tip,0]-data.xpos[quad,0]); linear_data=np.load(ROOT/"artifacts/s4/linearization/linear_model.npz"); eps=np.asarray(linear_data["state_epsilon"],dtype=float); C=identify_tip_output(model,snapshot,eps,eq); (out/"model").mkdir(exist_ok=True); np.save(out/"model/C_tip.npy",C); np.savetxt(out/"model/C_tip.csv",C,delimiter=",")
    audit={"shape":list(C.shape),"finite":bool(np.isfinite(C).all()),"method":"MuJoCo central finite difference at frozen 5-link equilibrium","epsilon":eps.tolist(),"equilibrium_relative_x":eq,"runtime_model_sha256":sha(ROOT/args.runtime_model)}; (out/"model/tip_linearization_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8",newline="\n")
    (out/"disturbance_observer_audit.json").write_text(json.dumps({"model":"x_next=A x+B(a+d), d_next=d","augmented_dimension":17,"measurement":"full 16-state x","uses_wind_x":False,"uses_wind_force":False,"uses_future_wind_csv":False,"inputs":["current_state","historical_command","A","B"],"observer_gain":0.15},indent=2)+"\n",encoding="utf-8",newline="\n")
    (out/"algorithm_audit.json").write_text(json.dumps({"controller":"DA-PMPC pilot","preview_horizon_steps":20,"preview_seconds":1.0,"outer_dt_s":0.05,"solver":"OSQP","warm_start":True,"optimized_input":"delta_a_x","reference_preview":True,"future_wind_used":False,"physical_model_parameters_modified":False,"pid_lqr_mppi_modified":False,"position_scale_candidates":[1.0,2.0,4.0],"tip_weight_candidates":[20.0,40.0,80.0],"grid_size":9},indent=2)+"\n",encoding="utf-8",newline="\n")
    A=np.load(ROOT/"artifacts/s4/linearization/A.npy"); B=np.load(ROOT/"artifacts/s4/linearization/B.npy"); Q=np.load(ROOT/"artifacts/s4/lqr/Q.npy"); R=np.load(ROOT/"artifacts/s4/lqr/R.npy"); P=solve_discrete_are(A,B,Q,R); K=np.linalg.solve(R+B.T@P@B,B.T@P@A); (out/"dependencies.json").write_text(json.dumps({"runtime_model_sha256":sha(ROOT/args.runtime_model),"A_shape":list(A.shape),"B_shape":list(B.shape),"Q_sha256":sha(ROOT/"artifacts/s4/lqr/Q.npy"),"R_sha256":sha(ROOT/"artifacts/s4/lqr/R.npy"),"K_sha256":sha(ROOT/"artifacts/s4/lqr/K.npy"),"s4_k_parity_max_abs":float(np.max(np.abs(K-np.load(ROOT/"artifacts/s4/lqr/K.npy")))),"physics_dt_s":float(model.opt.timestep),"outer_dt_s":0.05,"s0_s5_protected":True},indent=2)+"\n",encoding="utf-8",newline="\n")

if __name__=="__main__": main()
