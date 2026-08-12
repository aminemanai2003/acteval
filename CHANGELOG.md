# Changelog

## 1.0.0 - Unreleased

### Stable API and packaging

- Declared the documented public API stable under Semantic Versioning.
- Centralized package version metadata and moved to the Production/Stable
  classifier.
- Added a formal API stability policy and security reporting policy.
- Reworked the README as a complete Python-library landing page.

### v0.5 reporting and monitoring layer

- Added portfolio-segment evaluation and aligned segment model comparison.
- Added chronological evaluation with signed changes from a baseline period.
- Added weighted reference-quantile prediction drift and PSI contributions.
- Added dependency-light standalone HTML reports and CSV/JSON/HTML exports.
- Added a supported helper for exporting Matplotlib figures.

### v0.4 statistical inference layer

- Added reproducible percentile-bootstrap intervals for evaluation metrics.
- Added objective-aware paired bootstrap comparisons against named references.
- Added fixed-bin stratified bootstrap intervals for calibration tables.
- Added structured inference results and CSV export.

## 0.3.0 - 2026-08-12

- Added generic benchmarked financial decision regret.
- Added premium decisions, asymmetric pricing regret, and loss-ratio impact.
- Added reserve and capital shortfall amount/frequency diagnostics.
- Added distribution quantile decisions.
- Added quoted stop-loss reinsurance option selection under explicit
  premium, retained-loss, and capital-cost assumptions.
- Added realized reinsurance decision regret against a named option.
- Completed numerical Tweedie CDF/log-density evaluation and reproducible
  mixed-measure entropy estimation.
- Defined empirical predictive samples as discrete distributions with exact
  probability masses and Shannon entropy.
- Added immutable metadata modules for every supported actuarial task.

## 0.2.0 - 2026-08-12

- Added vectorized Poisson, Negative Binomial, Gamma, Lognormal, empirical,
  and compound Poisson-Gamma Tweedie predictive distributions.
- Added sample-based CRPS, log score, threshold Brier score, quantile score,
  and central interval score.
- Added prediction-interval coverage and width diagnostics.
- Added predictive variance and entropy diagnostics without universal ranking
  directions.
- Added `evaluate_distribution()` and `compare_distributions()`.

## 0.1.0 - 2026-08-12

- Added validated, exposure-aware actuarial point-prediction inputs.
- Added MAE, RMSE, Poisson, Gamma, and explicit-power Tweedie deviance.
- Added aggregate and quantile calibration diagnostics.
- Added weighted Gini, normalized Gini, and lift diagnostics.
- Added observed-tail MAE, RMSE, A/E, and large-loss bias.
- Added metric registry and parameterized metric specifications.
- Added evaluation and model-comparison result objects.
- Added calibration, lift, residual, and tail plots.
- Added Python 3.11-3.13 CI and package-build validation.
