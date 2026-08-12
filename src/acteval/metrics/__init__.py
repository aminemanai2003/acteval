"""Point-prediction metrics included in the foundation milestone."""

from acteval.metrics.accuracy import mae, rmse
from acteval.metrics.calibration import (
    ae_ratio,
    calibration_by_quantile,
    weighted_calibration_error,
)
from acteval.metrics.deviance import (
    gamma_deviance,
    poisson_deviance,
    tweedie_deviance,
)
from acteval.metrics.discrimination import (
    gini,
    lift,
    lift_by_quantile,
    normalized_gini,
    risk_group_lift,
)
from acteval.metrics.probabilistic import (
    brier_score,
    crps,
    distribution_interval_score,
    distribution_quantile_score,
    interval_score,
    log_score,
    quantile_score,
)
from acteval.metrics.tails import (
    large_loss_bias,
    tail_ae_ratio,
    tail_mae,
    tail_rmse,
)
from acteval.metrics.uncertainty import (
    central_prediction_interval,
    distribution_interval_coverage,
    distribution_interval_width,
    mean_interval_width,
    prediction_interval_coverage,
    predictive_entropy,
    predictive_variance,
)

__all__ = [
    "ae_ratio",
    "brier_score",
    "calibration_by_quantile",
    "central_prediction_interval",
    "crps",
    "distribution_interval_coverage",
    "distribution_interval_score",
    "distribution_interval_width",
    "distribution_quantile_score",
    "gamma_deviance",
    "gini",
    "interval_score",
    "large_loss_bias",
    "lift",
    "lift_by_quantile",
    "log_score",
    "mae",
    "mean_interval_width",
    "normalized_gini",
    "poisson_deviance",
    "prediction_interval_coverage",
    "predictive_entropy",
    "predictive_variance",
    "quantile_score",
    "risk_group_lift",
    "rmse",
    "tail_ae_ratio",
    "tail_mae",
    "tail_rmse",
    "tweedie_deviance",
    "weighted_calibration_error",
]
