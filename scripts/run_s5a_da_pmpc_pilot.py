"""Run only the two authorized S5A development scenes."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import yaml
from uav_sway.evaluation.da_pmpc_runner import run_scene,ROOT
from uav_sway.evaluation.da_pmpc_gate import raw_da_pmpc_gate

def main():
 p=argparse.ArgumentParser(); p.add_argument("--model-config",required=True); p.add_argument("--da-config",required=True); p.add_argument("--scenarios",nargs="+",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--headless",action="store_true"); a=p.parse_args();
 if set(a.scenarios)-{"approach_stop","crosswind_hover"}: raise SystemExit("S5A pilot forbids gust and holdout scenarios")
 cfg=yaml.safe_load(Path(a.da_config).read_text(encoding="utf-8")); out=ROOT/a.output_dir; paths={}; winds={"approach_stop":ROOT/"artifacts/s4/inputs/calm.csv","crosswind_hover":ROOT/"artifacts/s2/wind_bank/constant_crosswind.csv"};
 for s in a.scenarios:
  pth=out/"runs"/s/"run.csv"; paths[s]=pth; metric=run_scene(a.model_config,cfg,s,winds[s],ROOT/"artifacts/s2/references"/f"{s}.csv",pth); (pth.parent/"metrics.json").write_text(json.dumps(metric,indent=2)+"\n",encoding="utf-8",newline="\n")
 gate=raw_da_pmpc_gate(paths,{s:ROOT/"artifacts/s4/runs"/s/"run.csv" for s in paths},cfg); (out/"raw_gate.json").write_text(json.dumps(gate,indent=2)+"\n",encoding="utf-8",newline="\n"); raise SystemExit(0 if gate["pass"] else 2)
if __name__=="__main__": main()
