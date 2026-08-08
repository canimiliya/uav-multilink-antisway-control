# S6 task-space final report

## Final status

- Task-space protocol: **FROZEN**
- Task-space metric contract: **FROZEN**
- Method development: **CLOSED**
- Final task-space method: **NONE**
- Project method status: **CLOSED_WITH_PARTIAL_SCIENTIFIC_SUCCESS**

## What is complete

Engineering workflow is **COMPLETE**. The project has a reproducible five-link MuJoCo plant, frozen task-space state/reference/instrumentation, a formal metric contract, valid PID/LQR legacy baselines, a development-selected Task-LQR baseline, and retained audits for several advanced task-space controller attempts. The final lineage and hashes are in `artifacts/s6_taskspace/final/`.

The task-space protocol/metrics/baselines/negative studies are also **COMPLETE** as an evidence package. The S6T0 contract explicitly separates elapsed acquisition time from absolute simulation timestamp and defines task metrics as primary and UAV/control metrics as secondary.

## Positive evidence

The development-selected Task-LQR `lqr_011` improved calm task-position RMSE by 17.407892% and calm cutter-orientation RMSE by 51.721920% versus old LQR, with 2.455 s elapsed acquisition. In crosswind, orientation RMSE improved by 41.985099%, but position RMSE worsened by 6.328867% and the target was not acquired.

The independently audited S5B LS-PMPC holdout remains the project's formal robust-control success: 13.060586% overall paired mean tip-RMS improvement, 19.859454% overall x-RMSE improvement, 59/60 tip wins, zero primary safety failures, and 1.0731 ms solve-time P95. This evidence is legacy tip/x evaluation evidence and is not a task-space claim.

## Negative evidence and scientific boundary

Task-PID had no usable candidate. Task-LQI and ITS-RMPC did not pass the re-audited competence gate. DOB studies retained evidence that a scalar matched-disturbance assumption is not a sufficient representation of distributed-body wind. AE-TSLQR and the corrected acquisition-lock variant showed partial acquisition and some crosswind improvement, but no candidate satisfied the complete final calm/crosswind competence contract. The lock state machine itself behaved correctly, so the final AE result is a method-level failure rather than an unimplemented-lock failure.

Therefore, a robust task-space novel method is **NOT ESTABLISHED**. No near-miss is promoted as a final method, and no claim is made about unexecuted task-space gust recovery or random holdout.

## Holdout policy

`unused_holdout_preserved = true`. Since no task-space method passed the development gate, task-space gust recovery and random holdout were deliberately not executed. This preserves the unused test material from result-driven method selection.
