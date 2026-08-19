# Command-line interface

The `acteval` command evaluates prediction columns directly from a CSV file.
It uses the same validation, task defaults, metrics, and result metadata as the
Python API.

## Input

The shortest form expects columns named `y_true` and `y_pred`:

```bash
acteval evaluate predictions.csv --task claim_frequency
```

ActEval requires the task because frequency, severity, and pure-premium
evaluations have different valid metrics and defaults. Use custom column names
and select metrics explicitly when needed:

```bash
acteval evaluate predictions.csv \
  --task claim_frequency \
  --observed actual_claim_count \
  --predicted predicted_frequency \
  --metric rmse \
  --metric poisson_deviance \
  --metric ae_ratio
```

For rate inputs weighted by policy exposure, name the exposure column and
declare the scale explicitly:

```bash
acteval evaluate predictions.csv \
  --task claim_frequency \
  --exposure policy_years \
  --input-scale rate
```

Add `--sample-weight COLUMN` for a separate weighting column. When exposure and
sample weight are both present, their product is the effective weight, matching
the Python API.

## Output

Without an output path, the command prints a metric table. The output file
extension selects CSV, structured JSON, or a standalone HTML report:

```bash
acteval evaluate predictions.csv --task pure_premium --output report.json
acteval evaluate predictions.csv --task pure_premium --output report.csv
acteval evaluate predictions.csv --task pure_premium --output report.html
```

Existing outputs are protected by default. Pass `--force` to replace one. The
input CSV itself is never accepted as the output path.

Run `acteval evaluate --help` for every option. `python -m acteval` is an
equivalent entry point.
