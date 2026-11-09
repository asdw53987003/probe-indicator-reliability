# probe-indicator-reliability

Reference implementation of the **split-half check** of probe-vehicle indicators for crash-frequency models and site
screening (Algorithm A.1 of the paper *Are probe vehicle data enough? Indicator reliability for freeway
crash-frequency models and site screening*).

The check estimates, without reference data, how much sampling error the probe vehicles of one observation period
leave in unit-level behaviour indicators, how strongly that error attenuates the indicators' coefficients in a
negative binomial crash-frequency model, how many vehicle-equivalents (or weeks) a required reliability needs, and how
well a site list ranked by the indicators is expected to keep its units. Companion functions cover the checks that the
split halves cannot make (coarser sampling intervals), a correction of attenuated coefficients, and an execution of the
check against held-out vehicles where full 1 Hz records are available.

## Data in this repository

The original crash, traffic and trajectory data cannot be publicly released under data-sharing agreements. The
datasets in `examples/data/` are synthetically generated for demonstrating the split-half check and are not the
empirical data used in the study.

## Installation

Python 3.9 or later.

```
pip install -r requirements.txt
pip install -e .          # optional: makes `probe_reliability` importable from anywhere
```

## Quick start

```
python examples/run_example.py           # writes examples/example_output.txt
python -m pytest                         # unit tests
```

The examples and tests put `src/` on the path themselves; for your own scripts install the package first
(`pip install -e .`).

With your own files:

```python
import pandas as pd
from probe_reliability import sums_from_points, split_half_check

points = pd.read_csv('points.csv')              # vehicle, unit, pass, t (s), speed (m/s)
units = pd.read_csv('units.csv', index_col=0)   # length_km, group, Q (optional), route (optional)
design = pd.read_csv('design.csv', index_col=0) # columns of Eq. (1) without the probe indicators
crashes = pd.read_csv('crashes.csv', index_col=0)['crashes']

records = sums_from_points(points, tau=1.0)
res = split_half_check(records, units, design, period_weeks=13, crash_counts=crashes, seed=1)
print(res.summary())
```

or, after `pip install -e .`, from the command line:
`python -m probe_reliability points.csv units.csv design.csv 13 --crashes crashes.csv`.

## Algorithm A.1 and where each step is implemented

| Step | What it does | Function |
|---|---|---|
| 1 | Indicators of every unit from all vehicles; fit Eq. (1) and keep its working weights | `sums_from_points`, `indicators_table`, `fit_nb2`, `split_half_check` |
| 2 | J = 20 random partitions of the whole fleet into halves of equal size, the same half in every unit | `split_half_check` |
| 3 | Indicators of both halves in each unit; flag units with fewer than two vehicles in a half | `split_half_check` |
| 4 | Centre the half differences within each route and direction (working-weighted) | `split_half_check` |
| 5 | Eq. (2); flagged units take the largest variance of the network; an indicator is not used if more than 5% of units are flagged | `split_half_check` (`few_rule`, `divisor`) |
| 6 | Eq. (3): attenuation matrix, split-half reliability, predicted coefficient ratios, unit reliabilities | `split_half_check`, `attenuation_matrix` |
| 7 | Eq. (4): volume at reliability 0.5, vehicle-equivalents and weeks needed; correction where more weeks cannot be obtained | `k_hat`, `n_required`, `weeks_needed`; `simex_nb2`; `unit_error_variances` gives the Eq. (6) variances of the unit-specific error model, whose latent-variable fit is not included |
| 8 | Site lists ranked by the risk score, HDR, HDC and EHC: ranking reliability and expected overlap, Eq. (5) | `split_half_check` (`lists`, `lists_by_route`), `predicted_overlap` |
| 9 | Distortions shared by all probes: calibration slope on a 1 Hz validation subset and its variation between units | `calibration_slope`, `slope_heterogeneity` |
| 10 | Gain of an indicator on later crashes | not implemented (depends on the crash data and model of the user) |

`executed_check` and `held_out_reliability` run the check on individual vehicles: part of the vehicles is retained
(and, optionally, thinned to a coarser interval), the split-half check is applied to them, and its readings are
compared with the held-out reliability computed from the remaining vehicles at 1 s.

## Input format

`points` (1 Hz records): `vehicle`, `unit`, `pass` (one traversal of the unit), `t` (s), `speed` (m/s).
`sums_from_points` turns them into one row per vehicle × unit with `km`, `kmv`, `kmv2`, `kma2`, `km_acc`, `hd_events`.

`units` (indexed by unit): `length_km`; `group` (route and direction, the fixed-effect group of Eq. (1)); either
`weight` (working weight μ/(1 + αμ) of a fitted NB2 model) or crash counts passed as `crash_counts`; optionally
`offset` (ln `length_km` when absent), `Q` (daily volume, for EHC) and `route` (for lists within routes).

`design` (indexed by unit): the design matrix of Eq. (1) without the probe indicators, e.g. a constant, route-and-
direction dummies, ln volume and road covariates.

Indicators (Table S1 of the paper): MSP mean speed, VSD between-vehicle speed SD, ASD SD of accelerations between
records (a root mean square, as in Table S1), HDR harsh decelerations per 100 probe vehicle-km, HDC harsh decelerations per km and day (HDR × vehicle-
equivalents / 100 / days), EHC volume-expanded harsh decelerations per km and day (HDR × daily volume / 100).
Accelerations are differences of successive kept records over gaps up to max(2τ, 60) s; a harsh deceleration is an
acceleration of −2.0 m/s² or less, and consecutive exceedances count once. Other indicators can be passed as ratio
estimators (`ratio_indicators`).

## Settings

| Argument | Default | Meaning |
|---|---|---|
| `J` | 20 | partitions of the whole fleet |
| `seed` | None | seed of the partitions (fix it to reproduce a run) |
| `few_rule` | `'F'` | flagged units take the largest variance of the unflagged units; `'V'` variance function; `'O'` no error |
| `divisor` | `'4'` | Eq. (2); `'size'` divides by W²/(W₁W₂) where a unit's halves differ in weight |
| `lam_req` | 0.9 | required reliability in Eq. (4) |
| `top_share` | 0.10 | share of units on a site list |
| `sigma_u` | `'diagonal'` | Σ_U of Eq. (3) with error variances only; `'full'` adds error covariances between indicators from cross-products of the half differences |

## Conditions and limits

- **Vehicle identifiers must stay the same over the observation period.** Splitting passes, trips or daily
  identifiers instead of vehicles misses the error shared by the passes of a vehicle. Data whose identifiers change
  daily do not meet this condition.
- The check registers **sampling error only**. A coarser sampling interval rescales acceleration-based indicators in
  both halves alike (use `calibration_slope` on a 1 Hz validation subset); count lists mix behaviour with probe
  coverage; a fleet mix that varies between units shifts indicators in both halves.
- The reliability and the requirements depend on the working weights of the crash model.
- Eq. (3) uses a diagonal Σ_U. Indicators computed from the same records can have correlated errors (braking episodes,
  for instance, enter both ASD and HDR); the diagonal form then understates the reliability of an indicator given
  the others. `sigma_u='full'` estimates the covariances from the same half differences, and `executed_check` shows
  the size of the difference where held-out vehicles are available (section 3 of the example).
- Eq. (5) treats the errors of units as independent; for lists ranked by raw values the variances come from uncentred
  half differences. Predictions for HDC lists can be optimistic.

## Repository layout

```
src/probe_reliability/   indicators, split-half check, lists, requirements, NB2, calibration, correction, executed check
examples/                example data and an end-to-end run with expected output
tests/                   unit tests (pytest)
```

## Licence

Released under the MIT License (`LICENSE`).
