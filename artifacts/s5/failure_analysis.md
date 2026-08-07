# S5 MPPI failure analysis and closure

## Closure decision

S5 is `CLOSED_WITH_NEGATIVE_RESULT`. The same frozen six-candidate pure-MPPI protocol was rerun after the rollout timing repair. No further temperature, noise, horizon, rollout-count, or iteration tuning is authorized in S5.

The negative result is a control-performance limitation, not an implementation-correctness failure: the controller reduces the suspended-load motion, but its position tracking is not fair against the frozen LQR baseline under the fixed protocol.

## Gate accounting

| Gate | Result | Evidence |
|---|---:|---|
| Tip Gate | 6/6 pass | `artifacts/s5/tuning/mppi_grid.csv` |
| Position Gate | 0/6 pass | `artifacts/s5/tuning/mppi_grid.csv` |
| Dynamics safety | 6/6 pass | `dynamics_safe=true` for every candidate |
| Input safety | 6/6 pass | `input_safe=true` for every candidate |
| Actuator safety | 6/6 pass | `actuator_safe=true` for every candidate |
| Invalid rollouts | 0.0 rate | candidate diagnostics |

The position-fairness thresholds were not changed. From the frozen LQR raw baselines, the approach threshold is `0.11592356732 m` and the crosswind threshold is `0.10111583757 m` (110%). Every candidate exceeded both thresholds. The candidate ranges were approach `0.1671474975828215–0.2557662830712725 m` and crosswind `0.16084446217998602–0.5410858795785404 m`.

All six candidates passed the tip gates. Approach tip RMS ranged from `0.14022864804573712` to `0.15397447167520906 m`, below the frozen 0.95 LQR threshold `0.160416122295 m`. Crosswind tip RMS ranged from `0.05018908051477651` to `0.05289448669572056 m`, below the frozen 1.10 LQR threshold `0.05319887485 m`.

## Sampler diagnosis

The sampler did not collapse under the frozen temperature scale. Across the six candidate summaries, the candidate-median ESS remained between `26.68869538089245` and `60.98291779048144`; candidate-median maximum weight remained between `0.023434692599177482` and `0.09007127823354258`. The invalid-rollout rate was `0.0` for all candidates. Therefore the 0/6 position result is not explained by a sampler-collapse diagnostic.

## Timing and implementation correctness

The repaired rollout uses 12 actions and 13 reference boundaries. Action `j` uses boundary `r_j`; the state after that action is evaluated against `r_{j+1}`; the terminal state is evaluated against `r_12`. The perfect-tracking audit reports the old alignment error as `0.05 m` and the repaired error as `0.0 m`.

The following correctness evidence remains passing:

- positive terminal tip-cost sign;
- `ax_ref + delta_ax` sign contract;
- zero external forecast wind with static-air drag enabled;
- no future wind preview and future reference preview only;
- independent MuJoCo rollout data without real-plant contamination;
- targeted MPPI tests: 11 passed;
- project tests: 70 passed;
- upstream tests: 226 passed.

## Interpretation and scope

Under the frozen pure-MPPI protocol, MPPI does achieve the tip-motion gates but does not meet the LQR position-fairness gates. This supports the current interpretation: the fixed cost and rollout policy prioritize sway suppression, while the controller has no future crosswind preview and therefore does not recover the LQR position-tracking performance. This is recorded as a limitation of the frozen controller configuration, not as evidence for changing the model, wind, common inner loop, input limits, or MPPI search space.

No formal three-scenario MPPI run was executed because the development grid had zero safe candidates. The original Raw Gate remains explicit: `pass=false`, `status=BLOCKED_NO_SAFE_MPPI`. S6 final benchmark/evidence is not started by this closure task.
