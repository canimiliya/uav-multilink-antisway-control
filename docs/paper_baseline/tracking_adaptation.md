# Tracking adaptation for `SEP-NMPC-adapted`

The frozen baseline name is `SEP-NMPC-adapted`, full report name: **Tracking-adapted planar SEP-NMPC baseline derived from Rezaei et al., ICRA 2026**. This freezes the moving-reference mathematics before any controller performance evaluation.

## ORIGINAL PAPER

The paper's storage function is organized around a fixed desired equilibrium:

`V_paper = 0.5*q_dot^T*M(q)*q_dot + m_L*g*l*(1-cos(alpha)*cos(beta)) + 0.5*e_zeta^T*K*e_zeta`.

Its shaped input is `u_a = u + col(K*e_zeta,0,0)` and its strict passivity condition is `u_a^T*v <= -rho*||v||^2 - epsilon*||u_a||^2`, with `v=xi_dot`.

## OUR FAIR ADAPTATION

For the moving x reference, use causal tracking coordinates

`e_x = x - x_ref`, `e_v = x_dot - v_ref`,

and internal state `z=[e_x,e_v,alpha,alpha_dot]^T`.

The tracking storage function is

`V_tr = 0.5*[e_v,alpha_dot]^T*M_p(alpha)*[e_v,alpha_dot] + m_L*g*l*(1-cos(alpha)) + 0.5*K_e*e_x^2`.

The physical x force, reference feedforward, and shaped input are

`F_x=m_T*a_x`, `F_ff=m_T*a_ref`,

`u_ae = F_x - F_ff + K_e*e_x = m_T*(a_x-a_ref) + K_e*e_x`.

The inverse mapping is

`a_x = a_ref + (u_ae - K_e*e_x)/m_T`.

That result is always passed through the frozen project limiter: `a_x in [-2,2] m/s^2` and `|Delta a_x| <= 0.25 m/s^2/update` at 20 Hz.

The adapted passivity inequality is

`u_ae*e_v <= -rho*e_v^2 - epsilon*u_ae^2 + s_p`, with `0 <= s_p <= s_p_max`.

The frozen slack values are `s_p_max=5.0` and `w_s=1e5`. Slack is bounded and penalized; it is not an unbounded escape from the constraint.

The nonlinear predictor uses `e_x_dot=e_v`, `e_v_dot=x_ddot-a_ref`, `alpha_dot=omega_alpha`, and `omega_alpha_dot=alpha_ddot`, with accelerations obtained from the nonlinear `M_p` solve in `sep_planar_derivation.md`.

## Moving-reference and stationary audits

For perfect tracking, `e_x=e_v=alpha=alpha_dot=0` and `a_x=a_ref`, so `u_ae=0` and `V_tr=0`. This remains true for `v_ref=0.75 m/s`; no false braking term is introduced by absolute velocity.

For a stationary reference, `v_ref=a_ref=0`, hence `e_v=v_x` and `u_ae=F_x+K_e*e_x`, which is the paper-shaped planar form.

The analytical tracking cases and the 1000-case limiter parity result are in `artifacts/s5d1/tracking_adaptation_audit.json` and `artifacts/s5d1/input_mapping_audit.json`.

## HOCBF boundary

The function `build_hocbf_constraints(state, obstacles)` retains the Eq. (16)--(21) interface. The frozen benchmark has no obstacles, so `obstacles=[]` returns zero CBF rows. No obstacle-safety benefit is claimed. Non-empty obstacle rows remain a separately auditable extension.

## NOT CLAIMED

The original SEP-NMPC stability proof is **not claimed** to directly apply to this moving-reference, equivalent-payload, five-link adaptation. The passivity inequality is a frozen controller constraint, not a new theorem for the current plant.
