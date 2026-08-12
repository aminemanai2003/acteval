# Segment and temporal monitoring

Monitoring functions reuse the same validated evaluation pipeline as portfolio
reports. They do not redefine metrics or create a monitoring score.

## Portfolio segments

`evaluate_by_segment()` returns an `EvaluationResult` for every segment meeting
`min_observations`. `compare_by_segment()` repeats an aligned multi-model
comparison within each segment. Missing or empty labels are rejected; labels
that would collide after string normalization are also rejected.

Small segments are recorded in `skipped_segments`. They are not silently folded
into another group. Statistical credibility remains the caller's responsibility.

## Temporal validation

`evaluate_over_time()` treats labels as ordered strings, sorts them, and reports
each metric's signed change from the first retained period. ISO dates and
lexically ordered labels such as `2026-Q1` satisfy the ordering contract.

Changes are descriptive. A positive change may be favorable, unfavorable, or
neither depending on the metric's direction or target.

## Prediction drift

`prediction_drift()` constructs fixed cut points from weighted reference-score
quantiles. It reports:

- population stability index (PSI);
- each bin's reference/current portfolio share and PSI contribution;
- absolute and relative mean-prediction shifts.

Zero bin shares are clipped only inside the logarithm calculation using the
recorded `epsilon`. Reported portfolio shares remain unmodified.

PSI has no universal significance or action threshold. ActEval deliberately
does not label a value as acceptable, moderate, or severe. Operational limits
should be calibrated to the portfolio, score behavior, governance policy, and
cost of action.
