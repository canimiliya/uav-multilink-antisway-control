# S2 aerodynamic assumptions

The first S2 protocol uses world-x quadratic drag proxies with air density 1.225 kg/m^3. The airframe dimensions are a conservative dimension-envelope assumption; link and cutter coefficients, capsule/box proxies, and the no-torque choice are simulation assumptions. These values are not official DJI wind-drag parameters. Wind is applied independently at each rigid body's center of mass using its own MuJoCo Jacobian velocity.

S2 uses an anchored validation model and no controller. `ax_cmd_raw`, `ax_cmd_limited`, saturation, and solve-time fields are protocol placeholders, not controller results.
