"""Initial metric defaults by actuarial task."""

from typing import Final

from acteval.types import Task

DEFAULT_METRICS: Final[dict[Task, tuple[str, ...]]] = {
    "claim_frequency": ("rmse", "poisson_deviance", "ae_ratio"),
    "claim_severity": ("mae", "rmse", "gamma_deviance", "ae_ratio"),
    "pure_premium": ("mae", "rmse", "ae_ratio"),
}
