# UAV Multi-Link Anti-Sway Control

Project nature: 2026 personal simulation reproduction.

Current stage: S1 passive six-DoF UAV plus generated 4/5/6-link planar rigid-chain model.

The upstream Udaan baseline remains isolated at
`https://github.com/vkotaru/udaan.git`, commit
`9eb1a2dcfe438ce7b4c4cd119072e4f3d8a6a816`. Its baseline command is
`udaan run quad-payload -t 10 -c links`.

This stage generates project-owned MuJoCo XML from YAML configuration. The
quadrotor retains a free joint and Udaan-style geometry/site actuators. The
chain uses serial rigid bodies with one y-axis hinge per link; a model-level
weld is activated only for passive validation. Cutter mass, link mass, and
other model values are simulation assumptions, recorded in
`artifacts/s1/model_summary.json`.

No wind field, pilot trajectory, PID, LQR, MPPI, MPC, outer-loop controller,
experimental S1/S2/S3 scene, or real-hardware deployment is included.
