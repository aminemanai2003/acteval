"""Diagnostics focused on large observed insurance outcomes."""

import numpy as np
from numpy.typing import ArrayLike

from acteval.exceptions import InputValidationError
from acteval.registry import register_metric
from acteval.utils import effective_weights, weighted_quantile
from acteval.validation import combine_weights, validate_inputs


def _tail_inputs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    threshold: float | None,
    quantile: float | None,
    sample_weight: ArrayLike | None,
    exposure: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if threshold is not None and quantile is not None:
        raise InputValidationError("Specify threshold or quantile, not both.")
    inputs = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
        y_pred_domain="nonnegative",
    )
    weights = effective_weights(len(inputs.y_true), combine_weights(inputs))
    if threshold is None:
        selected_quantile = 0.95 if quantile is None else quantile
        cutoff = weighted_quantile(inputs.y_true, selected_quantile, weights)
        mask = (inputs.y_true >= cutoff) & (weights > 0)
    else:
        if not np.isfinite(threshold) or threshold < 0:
            raise InputValidationError("threshold must be finite and nonnegative.")
        cutoff = float(threshold)
        mask = (inputs.y_true > cutoff) & (weights > 0)
    if not np.any(mask):
        raise InputValidationError(
            f"No positive-weight observations satisfy the tail cutoff {cutoff}."
        )
    return inputs.y_true[mask], inputs.y_pred[mask], weights[mask], cutoff


@register_metric(
    name="tail_mae",
    tasks=("claim_frequency", "claim_severity", "pure_premium"),
    category="tail_risk",
    higher_is_better=False,
    description="Mean absolute error above an observed-loss threshold.",
)
def tail_mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    threshold: float | None = None,
    quantile: float | None = None,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute weighted MAE in the selected observed tail.

    Absolute thresholds use a strict boundary. Quantile cutoffs include the
    boundary atom so discrete claim outcomes always produce a populated tail.
    """
    observed, predicted, weights, _ = _tail_inputs(
        y_true,
        y_pred,
        threshold=threshold,
        quantile=quantile,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    return float(np.average(np.abs(observed - predicted), weights=weights))


@register_metric(
    name="tail_rmse",
    tasks=("claim_frequency", "claim_severity", "pure_premium"),
    category="tail_risk",
    higher_is_better=False,
    description="Root mean squared error above an observed-loss threshold.",
)
def tail_rmse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    threshold: float | None = None,
    quantile: float | None = None,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute weighted RMSE in the selected observed tail."""
    observed, predicted, weights, _ = _tail_inputs(
        y_true,
        y_pred,
        threshold=threshold,
        quantile=quantile,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    return float(np.sqrt(np.average(np.square(observed - predicted), weights=weights)))


@register_metric(
    name="tail_ae_ratio",
    tasks=("claim_frequency", "claim_severity", "pure_premium"),
    category="tail_risk",
    higher_is_better=None,
    target=1.0,
    description="Actual-to-expected ratio above an observed-loss threshold.",
)
def tail_ae_ratio(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    threshold: float | None = None,
    quantile: float | None = None,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute actual-to-expected ratio in the observed tail.

    Absolute thresholds use ``y_true > cutoff``; quantile cutoffs include the
    boundary atom. Values above one indicate tail underprediction. This
    direction is consistent with aggregate ``ae_ratio``.
    """
    observed, predicted, weights, _ = _tail_inputs(
        y_true,
        y_pred,
        threshold=threshold,
        quantile=quantile,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    expected = float(np.sum(weights * predicted))
    if expected <= 0:
        raise InputValidationError(
            "Tail A/E is undefined because aggregate tail expectation is zero."
        )
    return float(np.sum(weights * observed)) / expected


def large_loss_bias(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    threshold: float | None = None,
    quantile: float | None = None,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute expected-to-actual aggregate ratio in the observed tail."""
    ratio = tail_ae_ratio(
        y_true,
        y_pred,
        threshold=threshold,
        quantile=quantile,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    return 1.0 / ratio
