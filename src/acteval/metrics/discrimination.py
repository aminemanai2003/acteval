"""Prediction-ranking and actuarial discrimination diagnostics."""

import numpy as np
from numpy.typing import ArrayLike

from acteval.exceptions import InputValidationError
from acteval.registry import register_metric
from acteval.reports import LiftBin, LiftTable
from acteval.utils import effective_weights, risk_bin_indices, weighted_quantile
from acteval.validation import (
    combine_weights,
    validate_inputs,
    validate_probability,
)


def _grouped_lorenz_gini(
    y_true: np.ndarray,
    score: np.ndarray,
    weights: np.ndarray,
) -> float:
    order = np.argsort(score, kind="stable")
    ordered_score = score[order]
    ordered_true = y_true[order]
    ordered_weight = weights[order]
    unique_score, starts = np.unique(ordered_score, return_index=True)
    del unique_score
    group_weight = np.add.reduceat(ordered_weight, starts)
    group_loss = np.add.reduceat(ordered_weight * ordered_true, starts)
    total_weight = float(np.sum(group_weight))
    total_loss = float(np.sum(group_loss))
    if total_loss <= 0:
        raise InputValidationError(
            "Gini is undefined when weighted observed loss is zero."
        )
    population = np.concatenate(([0.0], np.cumsum(group_weight) / total_weight))
    loss = np.concatenate(([0.0], np.cumsum(group_loss) / total_loss))
    area = float(np.trapezoid(loss, population))
    return 1.0 - 2.0 * area


@register_metric(
    name="gini",
    tasks=("claim_frequency", "claim_severity", "pure_premium"),
    category="discrimination",
    higher_is_better=True,
    description="Weighted Gini of observed outcomes ordered by predictions.",
)
def gini(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute weighted Gini discrimination.

    Predictions are ordered from low to high. Equal predictions are aggregated
    before Lorenz integration, making the result invariant to row order within
    ties. Positive values indicate useful risk ordering; reversing a ranking
    can produce negative values.
    """
    inputs = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
    )
    weights = effective_weights(len(inputs.y_true), combine_weights(inputs))
    positive = weights > 0
    return _grouped_lorenz_gini(
        inputs.y_true[positive],
        inputs.y_pred[positive],
        weights[positive],
    )


@register_metric(
    name="normalized_gini",
    tasks=("claim_frequency", "claim_severity", "pure_premium"),
    category="discrimination",
    higher_is_better=True,
    description="Prediction Gini divided by perfect-ranking Gini.",
)
def normalized_gini(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute Gini normalized by the ordering induced by observations.

    A perfect ordering is 1 and an uninformative constant score is 0. The
    metric is undefined when the perfect-ordering Gini is zero, including when
    all positive-weight observations have the same outcome.
    """
    numerator = gini(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    denominator = gini(
        y_true,
        y_true,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    if np.isclose(denominator, 0.0):
        raise InputValidationError(
            "Normalized Gini is undefined because perfect-ranking Gini is zero."
        )
    return numerator / denominator


def lift_by_quantile(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    n_bins: int = 10,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> LiftTable:
    """Return observed risk and lift in ascending prediction quantiles."""
    inputs = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
    )
    weights = effective_weights(len(inputs.y_true), combine_weights(inputs))
    positive = weights > 0
    overall = float(np.average(inputs.y_true[positive], weights=weights[positive]))
    if overall <= 0:
        raise InputValidationError(
            "Lift is undefined when weighted observed mean is zero."
        )
    bins = risk_bin_indices(inputs.y_pred, n_bins=n_bins, weights=weights)
    rows: list[LiftBin] = []
    for bin_number in sorted(int(value) for value in np.unique(bins) if value > 0):
        mask = (bins == bin_number) & positive
        bin_weights = weights[mask]
        observed = float(np.average(inputs.y_true[mask], weights=bin_weights))
        rows.append(
            LiftBin(
                bin=bin_number,
                count=int(np.count_nonzero(mask)),
                weight=float(np.sum(bin_weights)),
                mean_prediction=float(
                    np.average(inputs.y_pred[mask], weights=bin_weights)
                ),
                mean_observed=observed,
                lift=observed / overall,
            )
        )
    return LiftTable(tuple(rows), requested_bins=n_bins)


@register_metric(
    name="lift",
    tasks=("claim_frequency", "claim_severity", "pure_premium"),
    category="discrimination",
    higher_is_better=True,
    description=(
        "Observed risk in the highest predicted group relative to portfolio risk."
    ),
)
def lift(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    top_fraction: float = 0.1,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute top-predicted-group observed risk divided by portfolio risk.

    The top group contains observations at or above the weighted
    ``1 - top_fraction`` prediction quantile. Ties remain together, so its
    realized weight can exceed the requested fraction.
    """
    fraction = validate_probability(top_fraction, name="top_fraction")
    inputs = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
    )
    weights = effective_weights(len(inputs.y_true), combine_weights(inputs))
    positive = weights > 0
    cutoff = weighted_quantile(inputs.y_pred, 1.0 - fraction, weights)
    selected = positive & (inputs.y_pred >= cutoff)
    portfolio_mean = float(
        np.average(inputs.y_true[positive], weights=weights[positive])
    )
    if portfolio_mean <= 0:
        raise InputValidationError(
            "Lift is undefined when weighted observed mean is zero."
        )
    top_mean = float(np.average(inputs.y_true[selected], weights=weights[selected]))
    return top_mean / portfolio_mean


def risk_group_lift(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    fraction: float = 0.1,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compare observed risk in highest and lowest predicted groups."""
    group_fraction = validate_probability(fraction, name="fraction")
    inputs = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
    )
    weights = effective_weights(len(inputs.y_true), combine_weights(inputs))
    positive = weights > 0
    low_cutoff = weighted_quantile(inputs.y_pred, group_fraction, weights)
    high_cutoff = weighted_quantile(inputs.y_pred, 1.0 - group_fraction, weights)
    low = positive & (inputs.y_pred <= low_cutoff)
    high = positive & (inputs.y_pred >= high_cutoff)
    low_mean = float(np.average(inputs.y_true[low], weights=weights[low]))
    if low_mean <= 0:
        raise InputValidationError(
            "High-to-low risk lift is undefined when low-group observed mean is zero."
        )
    high_mean = float(np.average(inputs.y_true[high], weights=weights[high]))
    return high_mean / low_mean
