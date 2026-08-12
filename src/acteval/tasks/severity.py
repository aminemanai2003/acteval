"""Claim-severity task definition."""

from typing import Final

from acteval.tasks.defaults import DEFAULT_METRICS
from acteval.tasks.definition import TaskDefinition

CLAIM_SEVERITY: Final = TaskDefinition(
    name="claim_severity",
    target="nonnegative loss conditional on a claim",
    prediction_domain="strictly positive when Gamma deviance is evaluated",
    exposure_interpretation="portfolio volume used as an effective weight",
    default_metrics=DEFAULT_METRICS["claim_severity"],
)
