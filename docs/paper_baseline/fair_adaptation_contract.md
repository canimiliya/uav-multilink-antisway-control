# Frozen fair-adaptation contract for future SEP-NMPC implementation

This contract is frozen by S5C before any controller implementation or performance experiment. It is a boundary document, not an implementation.

## Original method boundary

The paper's core method is the passivity-constrained NMPC around a point-mass, massless-cable internal model, with HOCBF constraints when obstacle data are part of the task. The current project will not claim that the original equations model the five-link chain.

## Allowed adaptations

1. Convert the paper's x-direction outer-loop force to the existing scalar `a_x_cmd` interface using the frozen current total mass.
2. Use a frozen reduced planar predictor for the controller internals. The minimum declared sway output is
   `s_tip = x_cutter_tip - x_uav` and `s_dot_tip = v_x,cutter_tip - v_x,uav`; an equivalent angle may be defined as `gamma_eq = atan2(s_tip, z_uav-z_cutter_tip)` for a downward configuration. This mapping is an adapter, not a claim of native multi-link equations.
3. Use measured/current five-link state to construct the declared tip or generalized-sway output when the paper's two-angle measurement is unavailable. No unmeasured future link state may be reconstructed from future wind.
4. Use current project nominal masses, geometry, and reference timing in the internal predictor, provided the exact values and source are frozen before benchmark execution.
5. Retain the paper's passivity storage/cost/inequality structure. For the current no-obstacle scenarios, instantiate the HOCBF set as empty; do not claim an obstacle-safety benefit. If obstacles are later introduced, use the paper's HOCBF equations without changing the current plant.
6. Run at the current project's fixed 20 Hz outer loop and use the paper's reported 2 s / 40-node horizon as the initial faithful discretization, with all numerical choices frozen before evaluation.
7. Use the current project's measured/current state, historical input, and historical output for feedback and disturbance rejection. A nominal predictor may be used without future wind.
8. Compute the same current metrics: tip RMS, x-RMSE, safety failures, acceleration magnitude, slew, solver timing, and finiteness.

## Forbidden adaptations

- Do not modify LS-PMPC, LQR, PID, MPPI, their parameters, or their evidence.
- Do not modify the five-link MuJoCo physics, link masses, hinge damping/friction, cutter geometry, UAV parameters, wind force model, reference bank, initial conditions, metrics, or shared GeometricInnerLoop.
- Do not replace the current inner loop with the paper's own attitude/thrust controller.
- Do not output direct rotor thrusts, torques, full SE(3) commands, cable tension commands, winch commands, or payload actuator commands to gain extra authority.
- Do not exceed `a_x in [-2,2] m/s^2` or `|Delta a_x|<=0.25 m/s^2/update`.
- Do not read future wind CSV samples, future random wind, or any preview unavailable to the other controllers.
- Do not give the paper controller a higher control frequency than the frozen 20 Hz outer loop.
- Do not alter the reference trajectory, initial state, scenario duration, or metric windows to favor the paper controller.
- Do not tune paper parameters on the 20-seed holdout or after observing the holdout outcome.
- Do not call the equivalent tip angle a five-link state from the original paper; label it as `OUR FAIR ADAPTATION` in implementation and reports.

## Input/output contract

The future controller receives the same permitted current state/reference/history contract as the other outer controllers and returns one scalar `a_x_cmd`. The shared GeometricInnerLoop remains the only force-to-attitude/thrust layer. The adapter must log both the raw paper-equivalent x force and the final limited acceleration command.

## Fairness gate before future implementation

The future implementation may proceed only if all are true:

| Gate | Required result |
|---|---|
| Same plant | unchanged five-link MuJoCo XML and current wind application |
| Same reference | current frozen reference bank |
| Same wind | same current wind profiles/seeds; no future wind truth |
| Same interface | one x acceleration output into GeometricInnerLoop |
| Same rate | 20 Hz outer loop |
| Same limits | same acceleration and slew limits |
| Same initial state | current evaluation initial conditions |
| Same metrics | current raw CSV schema and metrics |
| S5B unchanged | protected-file diff empty |
| Selection order | method selection remains frozen before implementation |

If any gate is false, classify the implementation as blocked or reject the baseline; do not silently weaken the contract.
