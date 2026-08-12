# Implementation plan audit

The supplied plan has a strong scientific direction: it separates evaluation
from training, avoids a universal score, makes calibration and observed-tail
behavior first-class concerns, and defers decision-aware metrics until their
definitions are defensible.

## Decisions implemented in v0.1

1. **Use a `src/` layout.** Tests exercise the installed package layout.
2. **Separate distribution and import names.** The distribution is
   `acteval-insurance`; the Python import remains `acteval`.
3. **Use a same-scale target contract.** ActEval does not infer whether a
   frequency vector contains counts or rates. Frequency and pure-premium users
   provide rates when using exposure weighting.
4. **Define effective weight.** Exposure is portfolio volume. When both inputs
   are supplied, effective weight is `sample_weight * exposure`.
5. **Allow MAE and RMSE for all MVP tasks.** They remain scale-dependent
   diagnostics and are never presented as actuarially complete objectives.
6. **Reuse scikit-learn deviance implementations.** ActEval owns validation,
   metadata, and interpretation rather than duplicating formulas.
7. **Make Tweedie power reproducible.** Direct calls require power. The
   pure-premium default is the documented power 1.5 and records it in metadata.
8. **Keep prediction ties together.** Weighted risk-quantile cut points can
   produce fewer effective bins than requested.
9. **Define calibration error explicitly.** It is the bin-weighted absolute
   difference between weighted observed and predicted means and is not called
   ECE.
10. **Fix a Gini convention.** Scores sort low to high; score ties aggregate
    before weighted Lorenz integration. Normalization uses the observation
    ordering.
11. **Define observed tails strictly.** Tail rows satisfy `y_true > cutoff`.
    Weighted quantiles use cumulative-weight linear interpolation.
12. **Keep A/E direction consistent.** Both aggregate and tail A/E are actual
    divided by expected. `large_loss_bias` provides the reciprocal direction.
13. **Record parameter metadata.** `MetricSpec` preserves powers, quantiles,
    thresholds, bin counts, and caller labels in results.

## Implemented v0.1 slices

- Packaging, validation, registry, task defaults, tests, and CI.
- Conventional and actuarial deviance metrics.
- Aggregate and prediction-quantile calibration.
- Gini, normalized Gini, and lift diagnostics.
- Observed-tail MAE, RMSE, A/E, and large-loss bias.
- `EvaluationResult`, `ComparisonResult`, `evaluate()`, and `compare()`.
- Calibration, lift, residual, and tail plots.
- Synthetic trade-off script/notebook and metric/API documentation.
- Wheel/sdist builds and trusted-publishing release workflow.

## Later layers implemented after the MVP

Version 0.2 adds vectorized predictive distributions, explicit sample shapes,
reproducible sample-based scores, proper log/Brier/quantile/interval scores, and
uncertainty diagnostics. Entropy has no optimization direction.

Version 0.3 adds explicit financial decision rules and named benchmarks for
pricing, loss ratios, reserve shortfall, capital shortfall, and quoted
stop-loss reinsurance choices. Regret remains a difference in financial loss
and is never combined across decision objectives.

## Completion matrix

| Plan area | Status | Implementation evidence |
|---|---|---|
| Model-agnostic point API | Complete | `evaluate()`, `compare()`, array-like validation |
| Frequency, severity, pure premium | Complete | immutable task definitions and task defaults |
| Results and comparison ranking | Complete | `EvaluationResult`, `ComparisonResult`, metric-specific ranking |
| Accuracy and deviance | Complete | weighted MAE/RMSE and Poisson/Gamma/Tweedie deviance |
| Calibration | Complete | aggregate A/E, risk-quantile table, weighted calibration error |
| Discrimination | Complete | weighted Gini, normalized Gini, scalar and tabular lift |
| Observed-tail diagnostics | Complete | tail MAE/RMSE/A/E and large-loss bias |
| Registry and validation | Complete | documented metric metadata and domain/shape/weight checks |
| Plotting | Complete | calibration, lift, residual, and tail plots as optional extras |
| Probabilistic v0.2 | Complete | six distribution adapters, five proper scores, interval and uncertainty diagnostics |
| Decision-aware v0.3 | Complete | explicit pricing, loss-ratio, reserve, capital, and reinsurance losses/benchmarks |
| Packaging and automation | Complete | `pyproject.toml`, full Apache-2.0 license, wheel/sdist builds, Python 3.11-3.13 CI, trusted-publishing workflow |
| Documentation and examples | Complete | README, API/metric/decision references, scripts, executable notebook |

The automated gates cover formatting, linting, strict typing, unit and
invariant tests with branch coverage, examples, package construction, dependency
consistency, and installation of the built wheel. There is deliberately no
universal model score.

## Deliberate boundaries

- ActEval evaluates supplied predictions; it does not fit models.
- Point inputs must already share a scale. Raw counts are not silently converted
  to rates, and exposure becomes an effective weight.
- Observed-tail metrics are retrospective diagnostics, not predictive tail
  probabilities.
- Built-in parametric distribution columns are sampled independently. Portfolio
  dependence must be supplied as joint empirical scenario rows.
- Empirical distributions are discrete, not kernel-density estimates. An unseen
  value therefore has zero empirical mass and infinite log-score loss.
- Tweedie quantiles and entropy are seeded numerical estimates; sample count and
  seed control reproducibility and precision.
- Reinsurance selection covers quoted stop-loss options under an explicit cost
  model; it is not a treaty-pricing or dynamic capital model.
- The distribution is ready for PyPI trusted publishing as
  `acteval-insurance`, but registering the publisher and issuing the first
  release require repository/PyPI owner actions.

## Version 1.0 extension audit

The v0.4, v0.5, and v1.0 extension roadmap is implemented as the 1.0 public
surface:

| Roadmap requirement | Evidence |
|---|---|
| Bootstrap confidence intervals | `bootstrap_evaluate()` and structured confidence intervals |
| A/E, Gini, lift, and tail uncertainty | Registry-driven inference supports every point metric selected by the caller |
| Paired model comparison | Shared row resamples and objective-aware candidate/reference deltas |
| Calibration uncertainty | Fixed risk bins with stratified intervals for prediction, observation, and A/E |
| HTML reports | Standalone escaped documents with no remote assets or JavaScript |
| Exportable tables and plots | CSV/JSON/HTML `export_table()` and `save_plot()` |
| Portfolio segments | Single- and multi-model segment result objects |
| Temporal validation | Ordered period metrics and signed baseline changes |
| Prediction drift | Weighted reference-quantile PSI contributions and mean shifts |
| Stable API | Semantic Versioning and documented compatibility/deprecation policy |
| Performance | Cached metric signature validation for repeated bootstrap/segment evaluation |
| Documentation | Dedicated inference, monitoring, reporting, migration, stability, security, and README audit documents |
| Production metadata | Single-source 1.0 version, stable classifier, typed marker, citation metadata |

Completion requires local and remote formatting, lint, strict typing, full tests,
source/wheel builds, clean artifact installation, and public API smoke tests.
