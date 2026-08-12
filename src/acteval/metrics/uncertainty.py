"""Predictive interval and distribution-uncertainty diagnostics."""

import numpy as np
from numpy.typing import ArrayLike

from acteval.exceptions import InputValidationError
from acteval.registry import register_metric
from acteval.types import PredictiveDistribution, Task
from acteval.utils import effective_weights
from acteval.validation import (
    combine_weights,
    validate_inputs,
    validate_probability,
)

_ALL_TASKS: tuple[Task, ...] = (
    "claim_frequency",
    "claim_severity",
    "pure_premium",
)


def prediction_interval_coverage(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute weighted empirical coverage of inclusive prediction intervals."""
    inputs = validate_inputs(
        y_true,
        lower,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    upper_values = np.asarray(upper, dtype=np.float64)
    if upper_values.shape != inputs.y_true.shape or not np.all(
        np.isfinite(upper_values)
    ):
        raise InputValidationError("upper must be finite and match y_true length.")
    if np.any(inputs.y_pred > upper_values):
        raise InputValidationError("lower must not exceed upper.")
    covered = (inputs.y_true >= inputs.y_pred) & (inputs.y_true <= upper_values)
    return float(np.average(covered, weights=combine_weights(inputs)))


def mean_interval_width(
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute weighted mean interval width as a sharpness diagnostic.

    Narrower intervals are only desirable conditional on adequate calibration;
    this function intentionally has no universal optimization direction.
    """
    inputs = validate_inputs(
        lower,
        upper,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    if np.any(inputs.y_true > inputs.y_pred):
        raise InputValidationError("lower must not exceed upper.")
    return float(
        np.average(inputs.y_pred - inputs.y_true, weights=combine_weights(inputs))
    )


def central_prediction_interval(
    distribution: PredictiveDistribution,
    *,
    coverage: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds of a central predictive interval."""
    central_coverage = validate_probability(coverage, name="coverage")
    alpha = 1 - central_coverage
    lower = np.asarray(distribution.quantile(alpha / 2), dtype=np.float64)
    upper = np.asarray(distribution.quantile(1 - alpha / 2), dtype=np.float64)
    expected_shape = (distribution.n_observations,)
    if lower.shape != expected_shape or upper.shape != expected_shape:
        raise InputValidationError(
            f"Distribution quantiles must have shape {expected_shape}."
        )
    return lower, upper


@register_metric(
    name="interval_coverage",
    tasks=_ALL_TASKS,
    category="uncertainty",
    higher_is_better=None,
    requires_distribution=True,
    description="Empirical coverage of a central predictive interval.",
)
def distribution_interval_coverage(
    y_true: ArrayLike,
    distribution: PredictiveDistribution,
    *,
    coverage: float = 0.9,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute empirical coverage of a central distribution interval."""
    lower, upper = central_prediction_interval(distribution, coverage=coverage)
    return prediction_interval_coverage(
        y_true,
        lower,
        upper,
        sample_weight=sample_weight,
        exposure=exposure,
    )


@register_metric(
    name="interval_width",
    tasks=_ALL_TASKS,
    category="uncertainty",
    higher_is_better=None,
    requires_distribution=True,
    description="Mean central prediction-interval width; no universal direction.",
)
def distribution_interval_width(
    y_true: ArrayLike,
    distribution: PredictiveDistribution,
    *,
    coverage: float = 0.9,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute central interval width as a conditional sharpness diagnostic."""
    lower, upper = central_prediction_interval(distribution, coverage=coverage)
    inputs = validate_inputs(
        y_true,
        y_true,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    if distribution.n_observations != len(inputs.y_true):
        raise InputValidationError("Distribution length does not match y_true.")
    return mean_interval_width(
        lower,
        upper,
        sample_weight=sample_weight,
        exposure=exposure,
    )


@register_metric(
    name="predictive_variance",
    tasks=_ALL_TASKS,
    category="uncertainty",
    higher_is_better=None,
    requires_distribution=True,
    description="Weighted mean predictive variance; no universal direction.",
)
def predictive_variance(
    y_true: ArrayLike,
    distribution: PredictiveDistribution,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute weighted mean predictive variance as an uncertainty diagnostic."""
    inputs = validate_inputs(
        y_true,
        y_true,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    if distribution.n_observations != len(inputs.y_true):
        raise InputValidationError("Distribution length does not match y_true.")
    values = np.asarray(distribution.variance(), dtype=np.float64)
    if (
        values.shape != inputs.y_true.shape
        or np.any(values < 0)
        or not np.all(np.isfinite(values))
    ):
        raise InputValidationError(
            "Predictive variance must be finite and nonnegative."
        )
    weights = effective_weights(len(values), combine_weights(inputs))
    return float(np.average(values, weights=weights))


@register_metric(
    name="predictive_entropy",
    tasks=_ALL_TASKS,
    category="uncertainty",
    higher_is_better=None,
    requires_distribution=True,
    description="Weighted mean predictive entropy; no universal direction.",
)
def predictive_entropy(
    y_true: ArrayLike,
    distribution: PredictiveDistribution,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute weighted mean entropy without treating it as model quality."""
    inputs = validate_inputs(
        y_true,
        y_true,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    if distribution.n_observations != len(inputs.y_true):
        raise InputValidationError("Distribution length does not match y_true.")
    values = np.asarray(distribution.entropy(), dtype=np.float64)
    if values.shape != inputs.y_true.shape or not np.all(np.isfinite(values)):
        raise InputValidationError("Predictive entropy must be finite.")
    weights = effective_weights(len(values), combine_weights(inputs))
    return float(np.average(values, weights=weights))
