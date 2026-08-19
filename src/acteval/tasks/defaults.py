"""Initial metric defaults by actuarial task."""

from typing import Final

from acteval.types import MetricSpec, Task

DEFAULT_METRICS: Final[dict[Task, tuple[MetricSpec, ...]]] = {
    "claim_frequency": (
        MetricSpec("rmse"),
        MetricSpec("poisson_deviance"),
        MetricSpec("ae_ratio"),
        MetricSpec("normalized_gini"),
    ),
    "claim_severity": (
        MetricSpec("rmse"),
        MetricSpec("gamma_deviance"),
        MetricSpec("ae_ratio"),
        MetricSpec("normalized_gini"),
    ),
    "pure_premium": (
        MetricSpec("rmse"),
        MetricSpec("ae_ratio"),
        MetricSpec("normalized_gini"),
    ),
}
