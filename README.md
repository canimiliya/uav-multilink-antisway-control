# UAV Multi-Link Anti-Sway Control

Project nature: 2026 personal simulation reproduction.

Current stage: S2 reproducible wind/scenario/metrics protocol on the S1 passive
six-DoF DJI Matrice 400-class UAV plus generated 4/5/6-link planar rigid-chain
model.

The upstream Udaan baseline remains isolated at
`https://github.com/vkotaru/udaan.git`, commit
`9eb1a2dcfe438ce7b4c4cd119072e4f3d8a6a816`. Its baseline command is
`udaan run quad-payload -t 10 -c links`.

This stage generates project-owned MuJoCo XML from YAML configuration. The
quadrotor retains a free joint and Udaan-style geometry/site actuators, with
9.74 kg M400-class mass and an explicitly labeled equivalent-box inertia
estimate. The chain uses serial rigid bodies with one y-axis hinge per link;
the total link mass is 1.0 kg. The independent box cutter mass is 2.5 kg.
A model-level weld is activated only for passive validation. Parameter source
classes and limits are recorded in `artifacts/s1/m400_parameter_basis.md` and
`artifacts/s1/model_summary.json`.

The passive S1 acceptance rule is corrected RMS decrease plus energy decrease,
finite bounded simulation, and no penetration. The historical 0.60 threshold
is retained only as a candidate future controlled-study target; see
`artifacts/s1/acceptance_policy_change.md`.

S2 adds three world-x wind profiles, three reference-only pilot scenarios,
independent quadratic force proxies for the quadrotor, each link, and the
cutter, a byte-stable wind/reference bank, a unified raw-run CSV schema, and
metrics computed directly from raw CSV. Headless smoke uses the model-level
anchor with `controller: none`; it does not claim reference tracking.

S3 adds a free-flight position-PID baseline. The PID uses only UAV x position
and velocity errors plus the S2 x reference derivatives; it does not use joint
or tip state for feedback. A shared three-dimensional position stabilizer and
the Udaan `GeometricAttitudeController` convert the limited acceleration
command into total thrust and body torques. The S3 runtime XML is an
actuator-range-only copy of the frozen S1 XML; its source/runtime fingerprint
comparison is recorded in `artifacts/s3/runtime/runtime_model_diff.json`.

The frozen 27-point grid selects one PID parameter set using position tracking
and control smoothness only. Three free-flight scenarios are recorded in
`artifacts/s3/runs/`, and `artifacts/s3/raw_gate.json` is recomputed directly
from those CSV files. LQR, MPPI, MPC, wind-model changes, electrical-wire or
cutting-contact models, vision systems, and real-hardware deployment remain
outside this stage.

S4 implements the nominal five-link full-state LQR identification path:
name-based 16-state injection/extraction, an automatically checked free-flight
equilibrium, a 50-physics-step nonlinear closed-loop map, central finite
differences, controllability/PBH analysis, discrete DARE, raw CSV comparison,
and independent gates. The original 27-point Q/R evidence is retained, while
the authorized repair uses the frozen 64-point position/velocity/joint/input
grid. The LQR reuses the S3 runtime model, geometric inner loop, wind bank,
references, and acceleration limits without changing S1-S3 files.

The repair separates the original wide operating-region linearization result
from a 200-sample mirrored local validation at 2x, 5x, and 10x finite-
difference epsilon. The 10x local validation passes. Corrected all-positive
candidate scoring selects a safe Q/R candidate from only `approach_stop` and
`crosswind_hover`; the three raw-CSV LQR scenarios pass the position-fairness
and approach-stop sway gates. Evidence is retained under `artifacts/s4/`.

S5 adds the third-controller implementation path: MPPI optimizes only the
x-direction acceleration correction using independent nonlinear MuJoCo
rollouts, a 12-step horizon, 64 rollouts, the frozen S4 Q/R weights, the
shared S3 geometric inner loop, and the shared acceleration limiter. The
correctness repair freezes the positive terminal tip penalty, the
`ax_ref + delta_ax` command contract, and zero external-wind forecast with
static-air drag retained in every rollout. The timing repair uses 13 reference
boundary samples for 12 actions, evaluating each post-action state against the
next boundary while the real plant still uses the current boundary feedforward.
The same six-point tuning grid was rerun with raw candidate safety and
actuator gates; all six candidates failed LQR position fairness, so S5 remains
`CLOSED_WITH_NEGATIVE_RESULT`; no formal MPPI scenario is reported as passed.
The six-candidate grid is frozen, with 6/6 tip gates passing, 0/6 position
gates passing, no sampler collapse, and corrected reference timing. Pre-fix,
corrected, timing-repair, and closure analyses are retained under
`artifacts/s5/`. S6 final benchmark/evidence is not started by this closure.
