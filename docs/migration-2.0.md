# Migrating from 1.x to 2.0

Version 2.0 makes corrective breaking changes where preserving 1.0 behavior
would retain statistically unsafe or ambiguous results.

## Exposure scale is explicit

High-level calls with `exposure` must now pass `input_scale="rate"`. Aggregate
counts or losses must omit exposure. This prevents accidental double weighting.

## Point prediction functionals are separated

One `evaluate()` call cannot mix mean-target scores such as RMSE or deviance
with median-target MAE. Evaluate mean and median predictions separately.

## Defaults no longer choose portfolio assumptions

Task defaults no longer include a 95th-percentile tail report or a Tweedie
deviance with power 1.5. Request tail levels and Tweedie power explicitly with
metric aliases or `MetricSpec`.

## Bootstrap inference fails closed

Undefined resamples now raise `InputValidationError`; they are not discarded
and retried. Choose an estimand defined on degenerate samples or a bootstrap
design appropriate to the data.

## Result containers and JSON

Result mappings and nested metadata are immutable. Use `to_dict()` to obtain a
detached mutable copy. Strict JSON export represents non-finite metric values as
the strings `"Infinity"`, `"-Infinity"`, and `"NaN"`.

## Optional Tweedie distribution methods

Install `acteval-insurance[tweedie]` to use numerical CDF, density, or entropy
methods on `TweedieDistribution`. Tweedie deviance does not require this extra.
