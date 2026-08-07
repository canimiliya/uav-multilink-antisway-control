"""Run the frozen 3x3 DA-PMPC Pilot development grid."""

from __future__ import annotations

import argparse,csv,json
from pathlib import Path
import numpy as np,yaml
from uav_sway.evaluation.da_pmpc_runner import run_scene,ROOT
from uav_sway.evaluation.da_pmpc_gate import gate_scene
from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics,load_controlled_csv

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-config",required=True); p.add_argument("--da-config",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--headless",action="store_true"); a=p.parse_args();
    if not a.headless: raise SystemExit("S5A requires --headless")
    cfg=yaml.safe_load(Path(a.da_config).read_text(encoding="utf-8")); out=ROOT/a.output_dir; out.mkdir(parents=True,exist_ok=True); (out/"candidates").mkdir(exist_ok=True)
    lqr={s:ROOT/"artifacts/s4/runs"/s/"run.csv" for s in cfg["development_scenarios"]}; winds={"approach_stop":ROOT/"artifacts/s4/inputs/calm.csv","crosswind_hover":ROOT/"artifacts/s2/wind_bank/constant_crosswind.csv"}; rows=[]
    index=0
    for scale in cfg["position_scale_candidates"]:
        for tip_weight in cfg["tip_weight_candidates"]:
            index+=1; scene_results={}; reasons=[]
            for scene in cfg["development_scenarios"]:
                local=dict(cfg); local["selected_position_scale"]=float(scale); local["selected_tip_weight"]=float(tip_weight); path=out/f"candidates/candidate_{index:02d}_{scene}.csv"; run_scene(a.model_config,local,scene,winds[scene],ROOT/"artifacts/s2/references"/f"{scene}.csv",path)
                scene_results[scene]=gate_scene(path,lqr[scene],scene,local); reasons.extend(f"{scene}:{x}" for x in scene_results[scene]["failure_reasons"])
            ratios=[scene_results[s]["metric"]["tip_rms_m"]/scene_results[s]["lqr_metric"]["tip_rms_m"] for s in scene_results]; pos=[scene_results[s]["metric"]["x_position_rmse_m"]/scene_results[s]["lqr_metric"]["x_position_rmse_m"] for s in scene_results]; rate=[scene_results[s]["metric"]["control_rate_proxy"]/max(scene_results[s]["lqr_metric"]["control_rate_proxy"],1e-9) for s in scene_results]; score=float(np.mean(ratios)-.25*np.mean(pos)) if not reasons else float("inf"); row={"candidate_index":index,"position_scale":scale,"tip_weight":tip_weight,"safe":not reasons,"score":score,"approach_x_rmse":scene_results["approach_stop"]["metric"]["x_position_rmse_m"],"crosswind_x_rmse":scene_results["crosswind_hover"]["metric"]["x_position_rmse_m"],"approach_tip_rms":scene_results["approach_stop"]["metric"]["tip_rms_m"],"crosswind_tip_rms":scene_results["crosswind_hover"]["metric"]["tip_rms_m"],"failure_reasons":";".join(reasons),"development_scenarios":"approach_stop,crosswind_hover","gust_used_for_selection":False}; rows.append(row)
            for s in cfg["development_scenarios"]:
                try:(out/f"candidates/candidate_{index:02d}_{s}.csv").unlink()
                except FileNotFoundError: pass
    with (out/"da_pmpc_grid.csv").open("w",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    safe=[r for r in rows if r["safe"]]; selection={"grid_size":len(rows),"safe_candidate_count":len(safe),"development_scenarios":cfg["development_scenarios"],"gust_used_for_selection":False,"selected":None}
    if safe:
        best=min(safe,key=lambda r:r["score"]); selection.update({"result":"selected","selected":best,"selected_position_scale":best["position_scale"],"selected_tip_weight":best["tip_weight"]}); cfg["selected_position_scale"]=best["position_scale"]; cfg["selected_tip_weight"]=best["tip_weight"]; Path(a.da_config).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8",newline="\n")
    else: selection["result"]="BLOCKED_DA_PMPC_PILOT"
    (out/"da_pmpc_selection.json").write_text(json.dumps(selection,indent=2)+"\n",encoding="utf-8",newline="\n"); raise SystemExit(0 if safe else 2)
if __name__=="__main__": main()
