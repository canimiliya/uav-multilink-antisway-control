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

No PID, LQR, MPPI, MPC, outer-loop controller, electrical-wire or cutting
contact model, vision system, three-dimensional stochastic wind, or
real-hardware deployment is included.
