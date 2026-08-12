"""ActEval: evaluation tools for actuarial predictive models."""

from acteval.api import compare, evaluate
from acteval.metrics import (
    ae_ratio,
    calibration_by_quantile,
    gamma_deviance,
    gini,
    large_loss_bias,
    lift,
    lift_by_quantile,
    mae,
    normalized_gini,
    poisson_deviance,
    risk_group_lift,
    rmse,
    tail_ae_ratio,
    tail_mae,
    tail_rmse,
    tweedie_deviance,
    weighted_calibration_error,
)
from acteval.plotting import (
    plot_calibration,
    plot_lift,
    plot_residuals,
    plot_tail_diagnostics,
)
from acteval.registry import MetricDefinition, get_metric, list_metrics
from acteval.reports import (
    CalibrationTable,
    ComparisonResult,
    EvaluationResult,
    LiftTable,
)
from acteval.types import MetricSpec, PredictiveDistribution, Task

__all__ = [
    "CalibrationTable",
    "ComparisonResult",
    "EvaluationResult",
    "LiftTable",
    "MetricDefinition",
    "MetricSpec",
    "PredictiveDistribution",
    "Task",
    "ae_ratio",
    "calibration_by_quantile",
    "compare",
    "evaluate",
    "gamma_deviance",
    "get_metric",
    "gini",
    "large_loss_bias",
    "lift",
    "lift_by_quantile",
    "list_metrics",
    "mae",
    "normalized_gini",
    "plot_calibration",
    "plot_lift",
    "plot_residuals",
    "plot_tail_diagnostics",
    "poisson_deviance",
    "risk_group_lift",
    "rmse",
    "tail_ae_ratio",
    "tail_mae",
    "tail_rmse",
    "tweedie_deviance",
    "weighted_calibration_error",
]

__version__ = "0.1.0"
