# SEP-NMPC-adapted parameter pre-registration

These values are frozen before any future SEP performance evaluation. They are development-only choices because the paper does not expose complete numerical `Q`, `R`, `K`, `rho`, or `epsilon` values in the source used for this audit.

## ORIGINAL PAPER

The paper reports `T=2 s`, `N=40`, fixed-step RK4, CasADi/acados, SQP-RTI, and warm starts. Numerical cost and passivity values not specified in the paper remain **NOT SPECIFIED IN PAPER** and are not presented as paper parameters.

## OUR FAIR ADAPTATION

The fixed horizon is `T=2.0 s`, `N=40`, giving `dt=0.05 s`, matching the project's 20 Hz outer loop. The fixed stage cost is

`J=sum_k(w_x*e_x^2 + w_v*e_v^2 + w_alpha*alpha^2 + w_omega*alpha_dot^2 + w_u*u_ae^2 + w_s*s_p^2)`.

| Quantity | Frozen candidates/value |
|---|---:|
| `K_e` | `[10, 40]` |
| `rho` | `[0.05, 0.20]` |
| `epsilon` | `[0.005, 0.020]` |
| grid size | `8` |
| `w_x` | `20` |
| `w_v` | `4` |
| `w_alpha` | `40` |
| `w_omega` | `6` |
| `w_u` | `1` |
| `w_s` | `1e5` |
| `s_p_max` | `5.0` |
| internal `|alpha|` | `<=60 deg` |

These are project-neutral development weights, selected before SEP performance evaluation and not copied from LS-PMPC optimization. No holdout or random-seed performance result may be used to change them.

Future S5D2 may select one of the eight rows using only the pre-registered development scenarios specified by the task card. Future S5D3 may then run the separately frozen gust and 20-seed comparison.

## Solver smoke boundary

The recorded smoke is one synthetic constant-reference, no-wind OCP with `SQP_RTI`, ERK, `T=2 s`, and `N=40`. It is not a MuJoCo benchmark and generates no tip RMS, x-RMSE, wind, holdout, or controller comparison result.

## NOT CLAIMED

- These values are not claimed to be the paper's hidden tuning values.
- No parameter candidate has been selected using performance.
- No claim is made that any row will outperform or underperform PID, LQR, LS-PMPC, or another controller.
