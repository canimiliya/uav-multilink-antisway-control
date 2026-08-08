# Resume-ready project summary

## Version A — conservative, directly usable

Built a reproducible MuJoCo simulation of a six-DoF M400-class UAV carrying a five-link suspended cutting device under distributed wind disturbances. Implemented and compared PID, full-state LQR, and an LQR-stabilized preview MPC controller through a common geometric inner loop and acceleration-limited interface. In a frozen 20-seed paired wind holdout, LS-PMPC achieved 13.060586% lower mean tip RMS and 19.859454% lower UAV x-RMSE than LQR, with 59/60 tip wins, zero primary safety failures, and 1.0731 ms solve-time P95. Also established and audited a cutter task-space protocol, metrics contract, and Task-LQR baseline.

Evidence: `artifacts/s5b/final_status.json`, `artifacts/s5b/statistics/primary_summary.json`, `artifacts/s6_taskspace/t0/`, and `artifacts/s6_taskspace/t2/`.

## Version B — research-development framing

Developed a staged task-space extension for a wind-disturbed UAV suspended multi-link cutting system: formalized cutter position/orientation/acquisition metrics, built a setpoint protocol, selected a task-space LQR development baseline, and performed controlled negative studies of Task-PID, Task-LQI/ITS-RMPC, DOB-based control, and adaptive-equilibrium TSLQR. The studies exposed distinct failure mechanisms, including crosswind acquisition failure and mismatch between scalar disturbance assumptions and distributed-body wind forces. The task-space line was closed without promoting a novel robust controller; the positive LS-PMPC legacy holdout and all negative evidence were preserved.

Do not describe the closed negative studies as final algorithms or claim task-space gust/random validation.
