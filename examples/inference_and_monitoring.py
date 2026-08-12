"""End-to-end inference, monitoring, and reporting example."""

from pathlib import Path

import numpy as np

import acteval as ae

rng = np.random.default_rng(2026)
n_policies = 240
exposure = rng.uniform(0.4, 1.0, n_policies)
risk_score = rng.lognormal(-1.5, 0.55, n_policies)
y_true = rng.poisson(risk_score * exposure) / exposure
current = risk_score * 1.08
candidate = risk_score * 0.98
segments = np.where(np.arange(n_policies) % 2, "Personal", "Commercial")
periods = np.repeat(["2026-Q1", "2026-Q2", "2026-Q3"], n_policies // 3)

intervals = ae.bootstrap_evaluate(
    y_true,
    candidate,
    task="claim_frequency",
    exposure=exposure,
    metrics=["poisson_deviance", "ae_ratio", "normalized_gini"],
    n_resamples=200,
    random_state=42,
)

paired = ae.paired_bootstrap_compare(
    y_true,
    {"Current": current, "Candidate": candidate},
    task="claim_frequency",
    reference="Current",
    exposure=exposure,
    metrics=["poisson_deviance", "ae_ratio"],
    n_resamples=200,
    random_state=42,
)

segment_report = ae.evaluate_by_segment(
    y_true,
    candidate,
    segments,
    task="claim_frequency",
    exposure=exposure,
    metrics=["poisson_deviance", "ae_ratio"],
)

timeline = ae.evaluate_over_time(
    y_true,
    candidate,
    periods,
    task="claim_frequency",
    exposure=exposure,
    metrics=["poisson_deviance", "ae_ratio"],
)

drift = ae.prediction_drift(current[:80], candidate[-80:], n_bins=5)

output = Path("acteval-example-report.html")
ae.save_html_report(paired, output, title="ActEval candidate review")

print(intervals.to_dataframe())
print(segment_report.to_dataframe())
print(timeline.to_dataframe())
print(drift.summary())
print(f"Saved {output}")
