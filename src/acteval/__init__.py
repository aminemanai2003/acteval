"""ActEval: evaluation tools for actuarial predictive models."""

from acteval.metrics import (
    ae_ratio,
    gamma_deviance,
    mae,
    poisson_deviance,
    rmse,
)
from acteval.registry import MetricDefinition, get_metric, list_metrics
from acteval.types import Task

__all__ = [
    "MetricDefinition",
    "Task",
    "ae_ratio",
    "gamma_deviance",
    "get_metric",
    "list_metrics",
    "mae",
    "poisson_deviance",
    "rmse",
]

__version__ = "0.1.0"
