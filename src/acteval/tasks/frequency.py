"""Claim-frequency task definition."""

from typing import Final

from acteval.tasks.defaults import DEFAULT_METRICS
from acteval.tasks.definition import TaskDefinition

CLAIM_FREQUENCY: Final = TaskDefinition(
    name="claim_frequency",
    target="conditional mean claim count or frequency on the prediction scale",
    prediction_domain="nonnegative",
    exposure_interpretation="portfolio volume used as an effective weight",
    default_metrics=DEFAULT_METRICS["claim_frequency"],
)
