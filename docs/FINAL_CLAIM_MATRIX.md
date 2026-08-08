# Final claim matrix

冻结依据：S6T6 final evidence freeze。所有数字均来自已冻结 artifact；本文件没有运行新的 physics simulation，也没有使用未执行的 gust/random task-space holdout。

## SAFE TO CLAIM

- The project built and validated a reproducible MuJoCo six-DoF UAV plus multi-link suspended-device simulation and a shared PID/LQR/LS-PMPC evaluation workflow. The task-space protocol and metric contract are frozen in `artifacts/s6_taskspace/t0/`.
- Under the S6T2 setpoint protocol, development-selected Task-LQR `lqr_011` reduced calm task-position RMSE by **17.407892%** and cutter-orientation RMSE by **51.721920%** versus old LQR; it acquired the calm target at **2.455 s elapsed acquisition time**. Source: `artifacts/s6_taskspace/t2/development_comparison.json`.
- Under the same S6T2 protocol, Task-LQR reduced crosswind cutter-orientation RMSE by **41.985099%**, while crosswind task-position RMSE increased by **6.328867%** and acquisition was not achieved. Source: `artifacts/s6_taskspace/t2/development_comparison.json`.
- LS-PMPC was frozen before the formal paired wind holdout and achieved **13.060586%** overall paired mean tip-RMS improvement versus LQR, **19.859454%** overall UAV x-RMSE improvement, **59/60** tip wins, **0** primary safety failures, and **1.0731 ms** primary solve-time P95. These are the frozen S5B results, not S6 results. Source: `artifacts/s5b/final_status.json` and `artifacts/s5b/statistics/primary_summary.json`.
- S6 completed a task-space protocol, metric contract, traditional baselines, Task-LQR development selection, and documented negative studies for Task-PID, Task-LQI/ITS-RMPC, DOB, and AE-TSLQR.
- The AE-TSLQR acquisition lock behavior was audited: locked bias drift was zero and locked bias-rate was zero in the retained 24-run audit. This validates the lock state machine, not the method's task competence. Source: `artifacts/s6_taskspace/t5r1/lock_behavior_audit.json`.

## NOT SAFE TO CLAIM

- Do not claim that S6 established a robust, formally validated task-space anti-wind controller.
- Do not claim that Task-LQR is a final task-space method; it was selected for development evidence only and failed crosswind acquisition.
- Do not claim ITS-RMPC, DOB-TS-RMPC, DOB-Task-LQR, AE-TSLQR, or AE-TSLQR with acquisition lock as superior to traditional methods or as final methods.
- Do not claim that any S6 task-space method passed the complete calm-plus-crosswind final competence contract.
- Do not claim that gust recovery or random task-space holdout was executed or validated. These holdouts remain unused and preserved.
- Do not transfer the S5B LS-PMPC percentages to task-space performance; they are from the frozen legacy tip/x paired wind holdout.
- Do not describe a near-miss acquisition as a stable final equilibrium or as proof of general robustness.
