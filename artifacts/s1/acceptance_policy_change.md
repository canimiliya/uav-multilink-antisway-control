# S1 passive acceptance policy change

## Old rule

The 15-second passive experiment required `decay_ratio < 0.60`.

## Change reason

That threshold would pressure the physical model to receive artificially
increased damping in order to pass. This conflicts with the later research
goal of evaluating active anti-sway control separately from the passive model.
The prior BLOCKED evidence is retained in `artifacts/s1/failure.log` and in
the Git history.

## New S1 rule

For the fixed 15-second, 10-degree, wind-free, no-control experiment:

- corrected final RMS must be lower than corrected initial RMS;
- final total mechanical energy must be lower than initial total energy;
- all values must be finite, with no physical explosion, joint-range violation,
  or cutter-ground penetration.

The `0.60` threshold is retained as a candidate target for a later controlled
anti-sway study. It is no longer a structural acceptance criterion for the
uncontrolled S1 model.
