# SEP-NMPC-adapted planar derivation

This freezes the internal nonlinear predictor for `SEP-NMPC-adapted`. It is a specialization of Rezaei et al., arXiv:2603.08860, Eq. (1)--(4), not a claim that the original paper models the five-link plant.

## ORIGINAL PAPER

The paper uses `M(q) q_ddot + C(q,q_dot) q_dot + G(q) = u`, with `q=[xi^T,gamma^T]^T`, `xi=[x,y,z]`, `gamma=[alpha,beta]`, a point payload, and a massless rigid cable. For this x-only specialization, `y` is constant, `z` is handled by the shared inner loop, and `beta=0`.

## OUR FAIR ADAPTATION

Define `q_p=[x,alpha]^T` and `m_T=m_Q+m_L`. The frozen mass matrix is

`M_p(alpha) = [[m_T, m_L*l*cos(alpha)], [m_L*l*cos(alpha), m_L*l^2]]`.

The retained nonlinear equations are

`m_T*x_ddot + m_L*l*cos(alpha)*alpha_ddot - m_L*l*sin(alpha)*alpha_dot^2 = F_x`,

`m_L*l*cos(alpha)*x_ddot + m_L*l^2*alpha_ddot + m_L*g*l*sin(alpha) = 0`.

Therefore

`[x_ddot, alpha_ddot] = M_p(alpha)^(-1) * [F_x + m_L*l*sin(alpha)*alpha_dot^2, -m_L*g*l*sin(alpha)]`.

The implementation solves these equations directly and does not use a small-angle linearization.

The frozen parameter mapping is read from `configs/model_5link.yaml` and measured from the generated MuJoCo zero-sway state:

| Quantity | Frozen value |
|---|---:|
| `m_Q` | `9.74 kg` |
| `m_L` | `1.0 + 2.5 = 3.5 kg` |
| `l_eq` | `2.57 m` |
| `m_T` | `13.24 kg` |

`l_eq` is the three-dimensional distance from the generated `link_1` suspension attachment point to the cutter rigid-body COM. It is not hand-entered as the nominal 2.5 m chain length.

## Mathematical evidence

`artifacts/s5d1/paper_equation_audit.json` checks the two scalar equations against the matrix form over 1000 deterministic finite samples. `artifacts/s5d1/equivalent_parameters.json` records the measured geometry and configuration source.

## NOT CLAIMED

- This reduced model is not the five-link plant model.
- The original point-payload/cable stability proof is not claimed for the five-link MuJoCo plant.
- The equivalent length is an adapter parameter, not a statement that the five rigid links behave as a massless cable.
