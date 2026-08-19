"""Conventional point-prediction accuracy metrics."""

import numpy as np
from numpy.typing import ArrayLike

from acteval.registry import register_metric
from acteval.types import Task
from acteval.validation import combine_weights, validate_inputs

_ALL_TASKS: tuple[Task, ...] = (
    "claim_frequency",
    "claim_severity",
    "pure_premium",
)


@register_metric(
    name="mae",
    tasks=_ALL_TASKS,
    category="accuracy",
    higher_is_better=False,
    prediction_functional="median",
    description="Mean absolute error, optionally weighted.",
)
def mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute mean absolute error for median predictions.

    MAE is the (optionally weighted) arithmetic mean of
    ``abs(y_true - y_pred)``. It has the same unit as the target and is less
    sensitive to large errors than RMSE. As a model-comparison score, MAE
    elicits a conditional median rather than a conditional mean. It does not
    measure calibration or discrimination.
    """
    inputs = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    absolute_error = np.abs(inputs.y_true - inputs.y_pred)
    return float(np.average(absolute_error, weights=combine_weights(inputs)))


@register_metric(
    name="rmse",
    tasks=_ALL_TASKS,
    category="accuracy",
    higher_is_better=False,
    prediction_functional="mean",
    description="Root mean squared error, optionally weighted.",
)
def rmse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute root mean squared error for mean predictions.

    RMSE is the square root of the (optionally weighted) mean squared error.
    It has the same unit as the target and emphasizes larger errors. It does
    not identify whether error concentration occurs in actuarially important
    portfolio segments.
    """
    inputs = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    squared_error = np.square(inputs.y_true - inputs.y_pred)
    return float(np.sqrt(np.average(squared_error, weights=combine_weights(inputs))))
