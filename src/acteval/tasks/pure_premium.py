"""Pure-premium task definition."""

from typing import Final

from acteval.tasks.defaults import DEFAULT_METRICS
from acteval.tasks.definition import TaskDefinition

PURE_PREMIUM: Final = TaskDefinition(
    name="pure_premium",
    target="nonnegative aggregate loss cost on the prediction scale",
    prediction_domain="strictly positive when Tweedie deviance is evaluated",
    exposure_interpretation="portfolio volume used as an effective weight",
    default_metrics=DEFAULT_METRICS["pure_premium"],
)
