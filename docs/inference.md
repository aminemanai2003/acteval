# Bootstrap inference

ActEval separates point estimates from sampling uncertainty. The inference API
uses nonparametric percentile bootstrap intervals and records its sample count,
confidence level, seed, undefined-resample policy, and method.

## Single-model intervals

`bootstrap_evaluate()` resamples complete rows. Observations, predictions,
exposures, and sample weights therefore remain aligned. It returns a
`BootstrapEvaluationResult` containing the ordinary `EvaluationResult` plus one
`ConfidenceInterval` per metric.

If a resample makes an estimand undefined—for example, normalized Gini on a
constant outcome—the run raises `InputValidationError` with the resample index.
Undefined samples are never discarded or retried because doing so would
condition the reported distribution on estimand validity.

The default is 1,000 resamples. At least 100 are required. For final reporting,
2,000 or more is generally preferable when computation permits.

## Paired comparisons

`paired_bootstrap_compare()` uses identical resampled rows for every model.
Each result includes:

- the candidate and reference point estimates;
- their raw metric difference;
- an objective-aware difference and percentile interval;
- the bootstrap standard error;
- whether the interval excludes zero.

Objective differences always use a lower-is-better convention:

- minimized metric: `candidate - reference`;
- maximized metric: `reference - candidate`;
- target metric: candidate absolute target distance minus reference distance.

Negative values therefore favor the candidate under that metric's objective.
This transformation does not combine metrics or claim universal superiority.

Intervals are not multiplicity-adjusted and should not be interpreted as a
substitute for a pre-specified model-governance decision rule.

## Calibration intervals

`bootstrap_calibration_by_quantile()` fixes risk bins from the original
predictions and resamples within each populated bin. This preserves each bin's
interpretation while estimating uncertainty for mean prediction, mean
observation, and A/E. Prediction ties remain together and zero-effective-weight
rows do not enter the bootstrap population.

## Reproducibility

Supply an integer `random_state` in governed or published work. `None` requests
non-deterministic sampling. Bootstrap output describes sampling variability of
the supplied evaluation portfolio; it does not account automatically for model
fitting uncertainty, temporal dependence, clustering, or data leakage.
