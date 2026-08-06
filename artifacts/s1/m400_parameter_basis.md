# DJI Matrice 400 parameter basis

Date recorded: 2026-08-06

This is a simulation parameter basis, not a measured identification of a
specific aircraft. The model intentionally separates source classes:

- `official`: values stated on the DJI Matrice 400 official specifications
  page.
- `derived_from_official`: direct unit conversions or geometric derivations
  from official values.
- `simulation_assumption`: values required to close the project-owned
  simplified rigid-body model and not measured here.
- `user_provided`: values fixed by the task card for this project.

## Aircraft

The official DJI specifications page reports 9,740 g takeoff weight with
batteries, 15.8 kg maximum takeoff weight, 6 kg maximum payload, unfolded
dimensions 980 x 760 x 480 mm, 1,070 mm diagonal wheelbase, 12 m/s maximum
wind resistance during takeoff and landing, and a 35 degree maximum pitch
angle. The page reports a 25 inch propeller size; this is converted to
0.635 m in the configuration.

Source: <https://enterprise.dji.com/matrice-400/specs>

The 9.74 kg aircraft mass, dimensions, wheelbase, payload limits, wind
resistance, pitch limit, and converted propeller diameter are retained in
`configs/airframes/dji_matrice_400.yaml` with an explicit source label.

The rotor center coordinate is derived from the diagonal wheelbase as
`1.07 / (2 * sqrt(2)) = 0.378302127934803 m` for the four demonstration
rotor positions. The suspension mount `[0, 0, -0.24] m` is a
`simulation_assumption`, not an official mounting measurement.

The diagonal inertia is an engineering estimate using the equivalent uniform
bounding-box method and the official unfolded dimensions:

`Ixx = m*(W^2 + H^2)/12`,
`Iyy = m*(L^2 + H^2)/12`,
`Izz = m*(L^2 + W^2)/12`.

It is explicitly marked `inertia_is_measured: false` and must not be cited as
an official DJI inertia measurement.

## Chain and cutter

The total chain mass of 1.0 kg is a
`provisional_simulation_assumption`, divided equally over 4, 5, or 6 rigid
links. The cutter mass of 2.5 kg is `user_provided`. Its 0.16 x 0.14 x 0.45 m
box geometry and bottom tip location are `simulation_assumption` values for
the first demonstration model, not measured cutter geometry.

The five-link external payload is therefore `1.0 + 2.5 = 3.5 kg`, and the
model total takeoff mass is `9.74 + 3.5 = 13.24 kg`. These remain below the
configuration limits of 6.0 kg payload and 15.8 kg maximum takeoff mass.

No wind field, pilot trajectory, controller, or active force input is added by
this parameter rebase.
