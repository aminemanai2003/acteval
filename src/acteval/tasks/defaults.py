"""Initial metric defaults by actuarial task."""

from typing import Final

from acteval.types import MetricSpec, Task

_TAIL_MAE_95 = MetricSpec("tail_mae", {"quantile": 0.95}, label="tail_mae_95")
_TAIL_AE_95 = MetricSpec("tail_ae_ratio", {"quantile": 0.95}, label="tail_ae_95")

DEFAULT_METRICS: Final[dict[Task, tuple[MetricSpec, ...]]] = {
    "claim_frequency": (
        MetricSpec("rmse"),
        MetricSpec("poisson_deviance"),
        MetricSpec("ae_ratio"),
        MetricSpec("normalized_gini"),
        _TAIL_MAE_95,
        _TAIL_AE_95,
    ),
    "claim_severity": (
        MetricSpec("rmse"),
        MetricSpec("gamma_deviance"),
        MetricSpec("ae_ratio"),
        MetricSpec("normalized_gini"),
        _TAIL_MAE_95,
        _TAIL_AE_95,
    ),
    "pure_premium": (
        MetricSpec("rmse"),
        MetricSpec(
            "tweedie_deviance",
            {"power": 1.5},
            label="tweedie_deviance_p1_5",
        ),
        MetricSpec("ae_ratio"),
        MetricSpec("normalized_gini"),
        _TAIL_MAE_95,
        _TAIL_AE_95,
    ),
}
