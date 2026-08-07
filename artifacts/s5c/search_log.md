# S5C-2026-ADVANCED-PAPER-BASELINE-SELECTION-R1 search log

Search date: 2026-08-08 (Asia/Shanghai). Search count: 20 query strings. Search method: Codex Web Search, followed by direct opening of official publisher/arXiv/PubMed pages. The academic-search skill routing was applied: Cross-disciplinary and robotics sources were checked through official publisher/arXiv pages; no secondary page was used as the primary paper source.

## Query log

1. `2026 quadrotor slung load finite-time anti-swing compensation function observer` — found C2, the PLOS One paper; retained for full formula and explicit disturbance-observer audit.
2. `2026 suspended load SEP-NMPC passivity HOCBF quadrotor` — found C1, arXiv:2603.08860; retained and later selected.
3. `2026 quadrotor suspended payload differential flatness ESO sliding mode anti-swing` — found C4; retained for formula/interface audit, later rejected because it requires the paper's full flatness/attitude realization.
4. `2026 UAV suspended payload anti-swing MPC nonlinear control` — found C3, C5, and related recent work; expanded candidate set.
5. `SEP-NMPC github` — no official repository identified; C1 code marked unavailable.
6. `Robust finite-time anti-swing control github` — no official repository identified; C2 code marked unavailable.
7. `Finite-time swing attenuation AFITSM OSMC github` — no official repository identified; C4 code marked unavailable.
8. `Trajectory tracking control for quadrotor slung-load system PDALESO github` — no official repository identified; C5 code marked unavailable.
9. `Robust control of aerial cable-suspended payload transportation via fully actuated system approach DOI authors` — confirmed C3 official ScienceDirect record and DOI.
10. `Finite-time swing attenuation for differentially flat quadrotor slung-load systems via robust AFITSM-OSMC partitioned control DOI` — confirmed C4 official ScienceDirect record, authors, venue, and online date.
11. `Residual-learning-enhanced extended Kalman filter quadrotor suspended payload DOI` — found C7; retained only as a non-controller rejection candidate.
12. `Decentralized Geometric Control for Cable-Suspended Payload Transport with Adaptive Mass Estimation GitHub` — found C8 official arXiv record; no official code repository identified.
13. `site:sciencedirect.com ... Robust control ... authors` — confirmed C3 title, venue, abstract, and external-disturbance/observer claims.
14. `site:sciencedirect.com ... Finite-time swing attenuation ... authors DOI` — confirmed C4 title, authors, highlights, assumptions, and validation scope.
15. `site:journals.sagepub.com ... PDALESO authors publication date` — confirmed C5 official SAGE record, DOI, authors, and first-online 2026 status.
16. `site:arxiv.org/abs/2601.03386 ... authors` — confirmed C6 authors and abstract-level cascade/interface claims.
17. `10.1016/j.conengprac.2026.106837` — confirmed C3 DOI and open-access status through ScienceDirect.
18. `10.1016/j.isatra.2026 AFITSM` — official search result did not expose a DOI; retained the stable ScienceDirect PII.
19. `10.1177/09596518251404227 authors` — confirmed C5 DOI and article page.
20. `10.1016/j.conengprac.2026.106837 authors` — cross-checked C3 author metadata; the official article page and a metadata index identify Junjie Kang and Jinjun Shan.

## Candidate discovery and disposition

### Retained for full audit

- C1 SEP-NMPC: strongest outer-loop compatibility. The official arXiv HTML gives the 5-DOF translational model, SO(3) attitude model, shaped storage function, strict passivity inequality, HOCBF recursion, and 2 s/40-node acados/CasADi setup. It is the selected future baseline.
- C2 compensation-function-observer anti-swing: complete published article and explicit finite-time anti-swing/unknown-disturbance method. It remains shortlisted but is not selected because its constrained implementation contract is less direct and the validation is simulation-only.
- C3 FAS robust controller: relevant and experimentally validated, but the method designs both outer and inner loops and therefore fails the frozen x-acceleration interface gate.
- C4 AFITSM-OSMC: highly sophisticated and directly anti-swing, but the differential-flatness partition and sliding-mode realization require full force/attitude/torque control, not only x acceleration through the frozen inner loop.
- C5 IT2FP-ADRC/PDALESO: relevant to slung-load trajectory tracking and disturbance rejection, but anti-swing is not the explicit control objective and the fuzzy/ADRC contract is not already the frozen outer interface.
- C6 off-center slung-load cascade: relevant, complete enough at abstract level, and experimentally validated by the paper, but its core geometry and inner-loop design do not match the current centered five-link plant.
- C7 residual-learning EKF: useful estimator research, but not a controller and therefore fails the anti-swing-controller hard condition.
- C8 GPAC: advanced and disturbance-aware, but it is a multi-UAV cooperative full-geometric architecture with flexible cables and cannot be fairly reduced to this single-UAV x-acceleration task.

## Search limitations

No paper PDF was committed. No code was downloaded. Citation counts were not used for ranking because all main candidates are 2026 publications/preprints and the task prioritizes completeness, evidence, and fairness. The original paper's reported performance is not treated as a result of this project; S5C ran no controller performance experiment.
