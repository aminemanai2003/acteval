# Migrating from 0.3 to 1.0

ActEval 1.0 is backward compatible with the documented 0.3 public API. Existing
point-prediction, predictive-distribution, plotting, and decision-aware calls do
not require changes.

## Package and import names

The installation and import names are unchanged:

```bash
python -m pip install --upgrade acteval-insurance
```

```python
import acteval as ae
```

## New capabilities

- Use `bootstrap_evaluate()` to add confidence intervals to existing metric
  selections.
- Use `paired_bootstrap_compare()` when a named candidate must be compared with
  a reference on aligned resamples.
- Use `evaluate_by_segment()`, `compare_by_segment()`, and
  `evaluate_over_time()` for portfolio monitoring.
- Use `prediction_drift()` for fixed-reference score-distribution diagnostics.
- Use result `save_html()` methods or `export_table()` for governed artifacts.

## Compatibility notes

- `__version__` remains available from `acteval` and now has a single packaging
  source of truth.
- The documented names in `acteval.__all__` are covered by the 1.x stability
  policy.
- `summary()` remains intended for people. Use `to_dict()` or `to_dataframe()`
  for integrations.
- No composite score, automatic PSI threshold, or universal model ranking was
  introduced.
