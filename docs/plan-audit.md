# Implementation plan audit

The supplied plan has a strong scientific direction: it separates evaluation
from training, avoids a universal score, makes calibration and tail behavior
first-class concerns, and defers decision-aware metrics until their definitions
are defensible. The following decisions make the MVP implementable without
silently choosing conventions.

## Decisions applied in the foundation milestone

1. **Use a `src/` layout.** This prevents tests from accidentally importing an
   uninstalled source tree and is safer for a package intended for PyPI.
2. **Separate distribution and import names.** `acteval` is already occupied on
   PyPI by an unrelated project. The distribution is `acteval-insurance`; the
   Python import remains `acteval`.
3. **Define exposure semantics.** The initial `ae_ratio` treats exposure as an
   aggregation weight applied to both observed and predicted values. Inputs
   must therefore use the same scale. `sample_weight` and exposure multiply
   when both are supplied.
4. **Allow RMSE and MAE for all three tasks.** The plan first limits them to
   severity and pure premium, but its frequency comparison milestone requires
   RMSE. They are useful scale-dependent diagnostics for frequency too, so the
   registry supports all MVP tasks while documenting their interpretation.
5. **Enforce deviance domains explicitly.** Poisson deviance requires
   nonnegative observations and strictly positive predictions. Gamma deviance
   requires strictly positive observations and predictions. Validation errors
   name the violated condition before scikit-learn is called.
6. **Reuse scikit-learn deviance implementations.** ActEval owns validation,
   metadata, and actuarial interpretation rather than duplicating mature
   numerical formulas.

## Ambiguities to resolve before their milestones

- **Tweedie power:** require an explicit power or a clearly documented caller
  configuration; never infer it from the task name.
- **Quantile bin ties:** specify deterministic behavior when repeated
  predictions make the requested number of bins impossible.
- **Weighted calibration error:** publish the exact bin weighting and scale
  before naming and registering the metric.
- **Normalized Gini:** state the ordering, weight handling, tie behavior, and
  zero-perfect-Gini behavior because multiple conventions exist.
- **Tail thresholds:** define whether the boundary is strict (`y > t`) and how
  weighted quantiles are computed.
- **Result metadata:** record metric parameters (for example Tweedie power or
  tail quantile) so reports are reproducible.

## Revised delivery slices

1. Foundation: package, validation, registry, first five metrics, tests, CI.
2. Calibration tables and a precisely defined weighted calibration error.
3. Discrimination and tail metrics with known-value tests.
4. `EvaluationResult`, `ComparisonResult`, `evaluate()`, and `compare()`.
5. Optional plotting, synthetic example, and expanded reference docs.

The v0.2 probabilistic and v0.3 decision-aware layers remain out of scope until
the MVP definitions and APIs are stable.
