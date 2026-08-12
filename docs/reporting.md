# Reporting and export

ActEval results are data objects first. Reporting functions do not recompute or
reinterpret metrics.

## Structured output

Core and extended result objects provide `to_dict()`, `to_dataframe()`, and
`summary()`. `export_table()` writes:

- CSV for the primary table;
- JSON for the full structured result;
- standalone HTML for human review.

The output format defaults to the file extension and can be supplied explicitly.

## HTML reports

`render_html_report()` returns a complete UTF-8 document. `save_html_report()`
writes it to disk. `EvaluationResult` and `ComparisonResult` also expose
convenience `to_html()` and `save_html()` methods.

Reports contain inline CSS, no JavaScript, and no remote assets. Titles, labels,
and serialized metadata are escaped. The report includes a reminder that
metrics represent different actuarial objectives.

## Plots

Plotting functions return a Matplotlib axis. `save_plot()` saves the owning
figure with an explicit DPI and tight bounding box. Plot calculation remains an
optional dependency and is separate from metric computation.
