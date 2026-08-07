# S6T0 Cutter Task-Space Control Contract

This document freezes the first cutter task-space measurement, reference, and
evaluation contract. It is instrumentation and evaluation infrastructure; it
does not introduce a task-space controller.

## State

The formal state is

\[
  (p_{tip}^W, v_{tip}^W, d_c^W, R_{cutter}^W),
\]

where `cutter_tip` is the MuJoCo site, `cutter` is the MuJoCo body, and the
local cutter-axis convention is `e_c=[1,0,0]^T`:

\[
 d_c^W = R_{cutter}^W e_c.
\]

The axis is normalized before use. Tip velocity is always obtained from

\[
 v_{tip}^W=J_{site}(q)\dot q,
\]

using `mujoco.mj_jacSite`; finite differences of tip positions are not a
formal implementation.

## Reference

The first reference is derived from the measured frozen-plant equilibrium:

\[
 r_{tip,eq}=p_{tip,eq}-p_{uav,eq},\qquad
 d_{ref}=d_{c,eq}.
\]

For an existing UAV reference, the task reference is

\[
 p_{tip,ref}(t)=p_{uav,ref}(t)+r_{tip,eq},\qquad d_{ref}(t)=d_{c,eq}.
\]

No unreachable pose is hand-entered. The pose and runtime-model SHA-256 are
stored in `artifacts/s6_taskspace/t0/equilibrium_task_pose.json`.

## Errors and acquisition

\[
 e_p=p_{tip}-p_{tip,ref},\quad
 e_{p,xz}=\sqrt{e_{p,x}^2+e_{p,z}^2},\quad
 e_\theta=\cos^{-1}(\operatorname{clip}(d_c^T d_{ref},-1,1)).
\]

The contract freezes `position <= 0.05 m`, `orientation <= 5 deg`, and
`tip speed <= 0.10 m/s`, continuously for `1.0 s`. The internal
`task_acquisition_timestamp_s` is the absolute simulation timestamp at which
that continuous interval begins. The formal
`task_acquisition_time_s` is elapsed time from `task_start_time_s`, where the
task starts at the first non-hover reference event:

`task_acquisition_time_s = task_acquisition_timestamp_s - task_start_time_s`.

If it never occurs, both fields are `null`; the simulation end time is never
substituted.

## Metrics

Task-space metrics are primary for this audit. UAV x/z metrics remain recorded
with `uav_metrics_secondary=true` and are not task-success definitions. The raw
baseline CSVs also retain the legacy `tip_displacement` and x/z error columns
so that parity can be independently recomputed.

The secondary `control_rate_proxy` is always computed by the frozen production
function `uav_sway.evaluation.metrics.control_rate_proxy`, using
`sum((diff(u) / diff(t))^2 * diff(t))`.

For gust recovery, the gust window is the nonzero `wind_x` interval. The peak
is the maximum task error during that interval. Recovery is the elapsed time
from the last nonzero-wind sample to the first complete 1-second acquisition
interval after it. For calm and constant wind, the same fields are `null` when
there is no finite disturbance end from which recovery can be defined.

## Scope freeze

The audit uses the existing 5-link plant, existing PID/LQR parameters,
1000 Hz physics, 200 Hz inner/logging schedule, 20 Hz outer schedule, 12 s
duration, existing references and winds, and the existing geometric inner
loop. PID, LQR, LS-PMPC, physics, wind files, random seeds, dt, solver, and
input limits are not modified. S6T0 does not implement Task-PID, Task-LQR,
TS-PMPC, or any new controller.
