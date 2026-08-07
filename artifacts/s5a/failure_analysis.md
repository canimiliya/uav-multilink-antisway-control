# S5A DA-PMPC Pilot result

Status: `BLOCKED_DA_PMPC_PILOT`.

The authorized 3x3 grid was run exactly once over the two development scenes:
`approach_stop + calm` and `crosswind_hover + constant_crosswind`. No gust,
random-seed holdout, or S6 experiment was run.

All nine candidates were finite and remained within the dynamics, input,
actuator, height, attitude, joint-range, anchor, and rotor-motor gates. The
hard pilot gates were not met:

- `approach_stop` x-RMSE ranged from 1.01698 to 1.05392 m, versus the frozen
  0.110654 m limit;
- `crosswind_hover` x-RMSE ranged from 0.87473 to 0.87473 m, versus the
  strict frozen LQR comparison;
- all nine candidates failed position tracking, so no candidate was selected
  and no formal three-scene run was authorized.

The tip RMS values were small, but that does not compensate for failed
position tracking. This is a pilot negative result, not a claim that DA-PMPC
has beaten LQR. The S5A grid is not expanded and the S0-S5 evidence remains
unchanged.
