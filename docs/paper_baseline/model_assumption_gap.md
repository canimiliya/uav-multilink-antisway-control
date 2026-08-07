# Model-assumption gap: SEP-NMPC vs current project

The selected paper is not a native five-link controller. The table separates the original paper model from the current evaluation plant and records the adaptation risk.

| Item | Original paper model | Current project | Gap | Fairness consequence |
|---|---|---|---|---|
| Vehicle DOF | 3D translation plus SO(3) attitude; six physical UAV DOF | 6-DOF free UAV body | NONE | Keep the current UAV plant unchanged |
| Suspension model | One fixed-length massless rigid cable | Five serial massive rigid links | MAJOR | Reduced internal predictor only; evaluation remains five-link MuJoCo |
| Number of links | One cable / two projected angles | Five rigid links and five hinges | MAJOR | `adaptation_required=YES`; never claim native five-link support |
| Link mass | Cable massless | Total link mass 1.0 kg, distributed across links | MAJOR | Use full plant mass in evaluation and a declared reduced model internally |
| Hinge damping | Not represented | 0.05 hinge damping plus friction loss in current YAML | MAJOR | Do not remove or retune physical dissipation |
| Payload type | Point mass | Rigid 2.5 kg cutter box with geometry and tip site | MODERATE | Use cutter-tip output only as a feedback/output mapping |
| Payload mass model | Scalar `m_L` | Rigid body inertia plus 2.5 kg mass | MODERATE | No plant change; reduced model parameters frozen before benchmark |
| Sway DOF | `gamma=[alpha,beta]` | Five joint angles, link velocities, cutter-tip sway | MAJOR | Equivalent planar sway output is an adaptation, not the original state |
| Wind | External disturbances are discussed; no explicit distributed wind model/observer is specified | Wind forces on UAV, every link, and cutter; no future wind truth allowed | MAJOR | Prediction cannot read future wind CSV; feedback/history only |
| Aerodynamic drag | Not specified in the paper equations | Quadratic body-wise wind-force proxy | MAJOR | Current wind/drag stays frozen and unchanged |
| Actuator dynamics | Attitude-thrust inner loop assumed exponentially accurate/passive | Existing GeometricInnerLoop and MuJoCo actuators | MODERATE | Reuse current inner loop; do not insert paper attitude controller |
| Control input | 3D translational force `F`, with zero swing generalized-force entries | Scalar x desired acceleration to GeometricInnerLoop | MODERATE | Convert only x force to acceleration and apply shared limits |
| State measurement | Full translational/swing/velocity state; obstacle data for HOCBF | Full simulation state is available to the controller contract | MODERATE | Full link state may be read if it replaces unavailable paper swing measurements without future truth |
| Constraints | Symbolic `U`, `A`, strict passivity, HOCBF | `a_x in [-2,2]`, slew `0.25` per update; no obstacle task | MAJOR | Current limits are mandatory even if paper used different limits |
| Outer-loop rate | Paper reports 100 Hz | Frozen project outer loop 20 Hz | MODERATE | Use 20 Hz; do not grant a faster update rate |
| Reference | `xi_d(t)` in the original transport task | Frozen project reference trajectories | MODERATE | Reuse current references without changing them to fit the paper |
| Stability claim | LaSalle argument for point-mass/cable model | No transfer of proof to five-link plant | MAJOR | Report only as original-paper theory |

## Adaptation verdict

The model gap is substantial but not blocking under the task's explicit rule that a paper controller may use a simplified internal model while the evaluation plant remains unchanged. The selected method is therefore marked `five_link_adaptability=MODERATE`, `adaptation_required=YES`, and `fairness_risk=MODERATE`. A future implementation must stop and reclassify as `HIGH` if preserving the passivity structure requires replacing the current inner loop, changing plant physics, or tuning on the holdout.
