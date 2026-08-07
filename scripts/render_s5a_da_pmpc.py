"""Render compact S5A development evidence."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def main():
 p=argparse.ArgumentParser(); p.add_argument("--input-dir",required=True); p.add_argument("--output",required=True); a=p.parse_args(); fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True)
 for s in ("approach_stop","crosswind_hover"):
  d=pd.read_csv(Path(a.input_dir)/s/"run.csv"); ax[0].plot(d.time,d.uav_x,label=f"{s} uav_x"); ax[0].plot(d.time,d.x_ref,"--",label=f"{s} ref"); ax[1].plot(d.time,d.tip_displacement,label=s)
 ax[0].set_ylabel("x (m)"); ax[1].set_ylabel("tip displacement (m)"); ax[1].set_xlabel("time (s)"); ax[0].legend(); ax[1].legend(); fig.tight_layout(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output,dpi=160); plt.close(fig)
if __name__=="__main__": main()
