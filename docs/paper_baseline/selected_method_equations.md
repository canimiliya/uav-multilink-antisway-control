# Selected method: SEP-NMPC

This document preserves the equations of the selected paper as written in the paper. It does not implement the controller and does not claim any result on the current project plant. Source: Rezaei et al., “SEP-NMPC: Safety Enhanced Passivity-Based Nonlinear Model Predictive Control for a UAV Slung Payload System”, arXiv:2603.08860, accepted at ICRA 2026; official HTML/PDF accessed 2026-08-08.

## 1. Original paper state definition

The paper defines

\[
q = [\xi^\top,\gamma^\top]^\top \in \mathbb{R}^{5},
\qquad \xi=[x,y,z]^\top,
\qquad \gamma=[\alpha,\beta]^\top.
\]

Here `xi` is the quadrotor position, `alpha` and `beta` are the two projected payload swing angles in the `xz` and `yz` planes, and `R in SO(3)` and `omega in R^3` describe the UAV attitude and body angular velocity. The full NMPC state is written as

\[
\mathbf{x}=[q^\top,\dot q^\top]^\top
\]

in the translational model, while the paper also includes the rotational state in the full system state.

## 2. Original paper dynamics

The translational dynamics are

\[
M(q)\ddot q+C(q,\dot q)\dot q+G(q)=u,
\tag{1}
\]

with

\[
M=\begin{bmatrix}(m_Q+m_L)I_3&M_c\\M_c^\top&M_p\end{bmatrix},
\]

\[
M_c=m_Ll\begin{bmatrix}
\cos\alpha\cos\beta&-\sin\alpha\sin\beta\\
0&\cos\beta\\
\sin\alpha\cos\beta&\cos\alpha\sin\beta
\end{bmatrix},
\qquad
M_p=m_Ll^2\begin{bmatrix}\cos^2\beta&0\\0&1\end{bmatrix}.
\]

The nonzero Coriolis/centrifugal entries given by the paper are

\[
\begin{aligned}
c_{14}&=-m_Ll\dot\alpha\sin\alpha\cos\beta-m_Ll\dot\beta\cos\alpha\sin\beta,\\
c_{15}&=-m_Ll\dot\alpha\cos\alpha\sin\beta-m_Ll\dot\beta\sin\alpha\cos\beta,\\
c_{25}&=-m_Ll\dot\beta\sin\beta,\\
c_{34}&=m_Ll\dot\alpha\cos\alpha\cos\beta-m_Ll\dot\beta\sin\alpha\sin\beta,\\
c_{35}&=-m_Ll\dot\alpha\sin\alpha\sin\beta+m_Ll\dot\beta\cos\alpha\cos\beta,\\
c_{44}&=-m_Ll^2\dot\beta\cos\beta\sin\beta,\\
c_{45}&=-m_Ll^2\dot\alpha\cos\beta\sin\beta,
\end{aligned}
\]

with `c_ij=-c_ji` and other entries zero. The gravity vector is

\[
G=\begin{bmatrix}
0\\0\\(m_Q+m_L)g\\m_Lgl\sin\alpha\cos\beta\\m_Lgl\cos\alpha\sin\beta
\end{bmatrix}.
\tag{2}
\]

The attitude model is

\[
\dot R=R\omega^\times,
\qquad
J\dot\omega=\tau-\omega^\times(J\omega).
\tag{3--4}
\]

The translational input is explicitly underactuated:

\[
u=\operatorname{col}(F_x,F_y,F_z,0,0),
\qquad F=\operatorname{col}(F_x,F_y,F_z).
\]

The paper states that an attitude-thrust inner loop tracks the commanded force and attitude. It gives a force-to-attitude realization using the desired yaw and `F=||u||`; the exact attitude mapping is retained in the source paper and is not re-used to replace the current project's GeometricInnerLoop.

## 3. Original control objective

The paper's target is

\[
\lim_{t\to\infty}[\xi(t),\gamma(t),\dot\xi(t),\dot\gamma(t)]
=[\xi_d,0,0,0],
\tag{5}
\]

with the desired equilibrium in the interior of the safe set:

\[
[\xi_d,0,0,0]\in\operatorname{int}(\mathcal C).
\tag{6}
\]

The explicit anti-swing state in the original paper is `gamma=[alpha,beta]`, not a five-link generalized sway vector.

## 4. Core SEP-NMPC control law

At each sampling instant `t_k`, the paper solves

\[
\begin{aligned}
\min_{\mathbf{x}(\cdot),u_a(\cdot)}\quad
J&=\int_{t_k}^{t_k+T}
\left(\|\mathbf{x}(t)-\mathbf{x}_d(t)\|_Q^2+
\|u_a(t)\|_R^2\right)\,dt\\
\text{s.t.}\quad
\dot{\mathbf{x}}(t)&=f(\mathbf{x}(t),u_a(t)),
\quad \mathbf{x}(t_k)=\mathbf{x}_k,\\
(\alpha(t),\beta(t))&\in\mathcal A,
\quad u_a(t)\in\mathcal U,\\
u_a^\top(t)v(t)&\le -\rho\|v(t)\|^2-\varepsilon\|u_a(t)\|^2,\\
A_{\mathrm{CBF}}(\mathbf{x}(t),t)u_a(t)&\ge b_{\mathrm{CBF}}(\mathbf{x}(t),t).
\end{aligned}
\tag{7}
\]

The collocated output is `v(t)=dot(xi(t))`. The paper's numerical setup uses `T=2 s`, `N=40` shooting nodes, fixed-step RK4, CasADi, acados, and SQP-RTI with warm starts; it reports 100 Hz control execution in its experiments.

## 5. Passivity / energy relation

The shaped storage function is

\[
V(\mathbf{x})=
\frac12\dot q^\top M(q)\dot q
+m_Lgl(1-\cos\alpha\cos\beta)
+\frac12e_\zeta^\top K e_\zeta,
\tag{8}
\]

where `e_zeta = xi - xi_d` and `K` is symmetric positive definite. The paper invokes

\[
\dot q^\top\left(\frac12\dot M(q)-C(q,\dot q)\right)\dot q=0.
\tag{9}
\]

The shaped input is

\[
u_a=u+\operatorname{col}(Ke_\zeta,0,0),
\]

which yields the port relation

\[
\dot V(\mathbf{x})=v^\top u_a,
\qquad v=\dot\xi.
\]

The strict passivity constraint is

\[
u_a^\top(t)v(t)\le -\rho\|v(t)\|^2-\varepsilon\|u_a(t)\|^2,
\qquad \rho,\varepsilon>0.
\tag{10}
\]

Therefore,

\[
\dot V(\mathbf{x})\le -\rho\|v\|^2-\varepsilon\|u_a\|^2\le0.
\tag{11}
\]

The source paper uses LaSalle/invariant-set reasoning to establish asymptotic convergence of the original point-mass/cable model. This proof is not claimed to transfer unchanged to the five-link plant.

## 6. HOCBF relation

The paper defines

\[
p_Q=\xi,
\qquad
p_L=\xi+l[\sin\alpha\cos\beta,\ \sin\beta,\ -\cos\alpha\cos\beta]^\top.
\]

For obstacle center `p_o,i`, obstacle radius `R_i`, body radius `r_j`, and margin `Delta`,

\[
d_{\min,i,j}=R_i+r_j+\Delta,
\qquad
h_{i,j}=\|p_j-p_{o,i}\|^2-d_{\min,i,j}^2.
\]

The relative-degree-two recursion is

\[
\psi_{i,j,1}=\dot h_{i,j}+\kappa_1h_{i,j},
\qquad
\psi_{i,j,2}=\dot\psi_{i,j,1}+\kappa_2\psi_{i,j,1}\ge0.
\tag{19}
\]

With

\[
\ddot p_j=f_{v,j}(\mathbf{x},t)+G_v(\mathbf{x})u_a,
\]

the paper gives the affine condition

\[
\psi_{i,j,2}=L_f^2h_{i,j}+\kappa_1L_fh_{i,j}
+\kappa_2\psi_{i,j,1}+2r_{i,j}^\top G_vu_a\ge0,
\tag{20}
\]

and stacks it as

\[
A_{\mathrm{CBF}}(\mathbf{x},t)u_a\ge b_{\mathrm{CBF}}(\mathbf{x},t).
\tag{21}
\]

The current benchmark has no obstacle task. The future implementation must retain the HOCBF interface but uses an empty active obstacle set for the existing non-obstacle scenarios; no safety benefit from HOCBF is to be claimed in those runs.

## 7. Parameters and gains

The following are specified by the paper or its official HTML:

| Item | Paper value/status |
|---|---|
| Prediction horizon | `T=2 s` |
| Shooting nodes | `N=40` |
| Integrator | fixed-step RK4 |
| Solver stack | CasADi + acados, SQP-RTI, warm start |
| Reported control rate | 100 Hz |
| Passivity gains `rho`, `epsilon` | NOT SPECIFIED IN PAPER text available for this audit |
| Storage gain `K` | NOT SPECIFIED IN PAPER text available for this audit |
| Cost matrices `Q`, `R` | NOT SPECIFIED IN PAPER text available for this audit |
| Swing set `A` and actuator set `U` | NOT SPECIFIED IN PAPER text available for this audit |
| HOCBF gains `kappa_1`, `kappa_2` | positive; numerical values NOT SPECIFIED |
| Wind observer | NOT SPECIFIED IN PAPER |
| Future wind input | NOT SPECIFIED and prohibited by this project |
| Solver tolerances/iteration limits | NOT SPECIFIED IN PAPER |

Unspecified quantities must be frozen before any future benchmark. They may not be tuned against the 20-seed holdout after seeing performance.

## 8. Original input/output relation

Original outer-loop input: `u_a`, a shaped three-dimensional force embedded in the five-dimensional generalized-force vector. Original physical force input: `F=[F_x,F_y,F_z]^T`; the final attitude/thrust layer realizes the force.

Future current-project adapter output: a single scalar `a_x_cmd` passed to the already frozen `GeometricInnerLoop`. The adapter will convert the selected x-force component with the frozen total mass, then apply the existing `a_x in [-2,2] m/s^2` and `|Delta a_x|<=0.25 m/s^2/update` contract. This conversion is an interface adapter, not a new plant actuator.

## 9. Original assumptions

- The suspension is a massless rigid cable of fixed length `l`.
- The payload is a point mass `m_L`.
- The swing angles remain in `(-pi/2, pi/2)` and the payload remains below the UAV.
- The translational model has five generalized coordinates and three virtual force inputs.
- The attitude/thrust inner loop tracks commanded force/attitude sufficiently accurately and is treated as exponentially accurate/passive.
- Obstacle states used in HOCBF are available as time-varying positions, velocities, and accelerations when dynamic obstacles are present.
- A wind observer or future-wind truth is not specified by the paper.

## 10. Paper-report boundary

The paper reports simulation and real-world experiments and states real-time execution. Those are original-paper reports only. S5C did not implement SEP-NMPC, did not run a controller performance experiment, and makes no claim about its performance relative to PID, LQR, or LS-PMPC on the current plant.
