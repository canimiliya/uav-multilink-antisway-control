# Future implementation plan: SEP-NMPC external baseline

S5C does not implement this plan. It only freezes the work required after the paper selection.

## Planned files and dependencies

Planned project-owned files, subject to a later implementation task:

- `src/uav_sway/paper_baseline/sep_nmpc_controller.py`: controller wrapper and x-force-to-acceleration adapter.
- `src/uav_sway/paper_baseline/sep_nmpc_model.py`: reduced planar internal model and equivalent tip-sway mapping.
- `src/uav_sway/paper_baseline/sep_nmpc_config.py`: frozen paper/adaptation parameters and contract checks.
- `tests/test_sep_nmpc_contract.py`: dimensions, no-future-wind, limit, slew, and shared-inner-loop checks.
- Future evidence under a later stage's explicitly authorized artifact directory; no S5C evidence is to be overwritten.

Required dependencies are CasADi and acados, matching the original paper's stated stack. RK4 discretization and SQP-RTI/warm starts are required for the faithful path. OSQP is not required by the paper's implementation; it may not be substituted without recording a method change. No dependency is installed in S5C.

## Equations to implement later

1. Paper equations (1)-(4): reduced point-mass/cable translational and attitude dynamics.
2. Paper equation (7): finite-horizon NMPC objective, dynamics, input/swing constraints, passivity inequality, and HOCBF constraints.
3. Paper equations (8)-(11): shaped energy storage and strict passivity relation.
4. Paper equations (16)-(21): control-affine acceleration channel and relative-degree-two HOCBF recursion.
5. `OUR FAIR ADAPTATION`: map five-link current state to `s_tip`, `s_dot_tip`, and optionally `gamma_eq`; keep the full plant outside the predictor.
6. Interface conversion: `F_x -> a_x_cmd`, followed by the shared acceleration and slew limiter.

## Controller dimensions and timing

| Quantity | Minimum viable | Full faithful |
|---|---:|---:|
| Internal generalized position | `[x, gamma_eq]` (2) | paper 3D `xi` + two angles, with five-link output embedding |
| Internal state | 4: position/sway and two velocities | paper full state plus attitude/velocity state required by the selected transcription |
| Control decision | one x-force/residual scalar after reduction | paper 3D translational force `F` with HOCBF/passivity constraints |
| Output | one `a_x_cmd` | one `a_x_cmd` after the same reduction |
| Outer-loop rate | 20 Hz | 20 Hz, even though the paper reports 100 Hz |
| Horizon | 2 s / 40 nodes | 2 s / 40 nodes initially; freeze any later change |
| Plant | unchanged five-link MuJoCo | unchanged five-link MuJoCo |

## Minimum viable implementation

- Use the reduced planar model and the current cutter-tip sway output.
- Use CasADi/acados with RK4 and the paper's 2 s/40-node discretization.
- Retain the energy storage and strict passivity constraint; use an empty HOCBF obstacle set because current benchmark scenarios have no obstacles.
- Feed only current/history state, reference, and previous command; no future wind.
- Return `a_x_cmd` through the existing 20 Hz limiter and GeometricInnerLoop.
- Validate dimensions, finite outputs, solver status, common limits, no future-wind access, and unchanged protected files before any performance run.

This version is suitable only as a clearly labeled reduced external baseline. It must not be described as a full reproduction of the original 3D SEP-NMPC safety experiments.

## Full faithful implementation

- Implement the paper's 3D translational and SO(3)-compatible outer model while retaining the current inner loop as the plant interface.
- Embed the five-link measured state into the paper's swing-energy channel and document the exact output map.
- Retain the HOCBF machinery when an obstacle scenario exists; otherwise keep the set empty and report that no active HOCBF test was performed.
- Preserve the paper's passivity inequality, QP-compatible affine HOCBF constraints, and solver timing logs.
- Evaluate the exact same current five-link plant, wind bank, references, limits, initial conditions, and metrics.

## Parameter freeze and implementation risks

The paper does not expose every numeric `Q`, `R`, `K`, `rho`, `epsilon`, `kappa`, actuator bound, or solver tolerance in the official text audited here. A later task must pre-register those values before holdout evaluation, using paper values where available and a development-only rule for values that remain unspecified. The missing public code increases reimplementation cost. The main risk is that the reduced five-link sway mapping changes the meaning of the paper's passivity proof; the implementation report must call this an adaptation and must not transfer the original proof automatically.
