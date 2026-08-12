"""Numerical helpers shared by actuarial diagnostics."""

import numpy as np

from acteval.exceptions import InputValidationError
from acteval.types import NumericArray
from acteval.validation import validate_n_bins, validate_probability


def effective_weights(
    length: int,
    weights: NumericArray | None,
) -> NumericArray:
    """Return explicit positive-mass weights for aggregation."""
    if weights is None:
        return np.ones(length, dtype=np.float64)
    return weights


def weighted_quantile(
    values: NumericArray,
    quantile: float,
    weights: NumericArray | None = None,
) -> float:
    """Compute a linearly interpolated weighted quantile.

    Positive-weight observations are sorted stably. Linear interpolation is
    performed against cumulative weight, with the minimum anchored at zero.
    This convention keeps high quantiles below the maximum when the requested
    probability lies inside the final observation's weight interval.
    """
    probability = validate_probability(quantile, name="quantile")
    effective = effective_weights(len(values), weights)
    positive = effective > 0
    if not np.any(positive):
        raise InputValidationError("Weighted quantile requires positive weight.")
    selected_values = values[positive]
    selected_weights = effective[positive]
    order = np.argsort(selected_values, kind="stable")
    sorted_values = selected_values[order]
    sorted_weights = selected_weights[order]
    cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    positions = np.concatenate(([0.0], cumulative))
    interpolation_values = np.concatenate(([sorted_values[0]], sorted_values))
    return float(
        np.interp(
            probability,
            positions,
            interpolation_values,
        )
    )


def risk_bin_indices(
    predictions: NumericArray,
    *,
    n_bins: int,
    weights: NumericArray | None = None,
) -> NumericArray:
    """Assign ascending risk bins while keeping tied predictions together.

    Duplicate cut points are collapsed, so the effective number of bins can be
    lower than requested. Bin identifiers are one-based and contiguous.
    """
    requested = validate_n_bins(n_bins)
    effective = effective_weights(len(predictions), weights)
    probabilities = np.arange(1, requested, dtype=np.float64) / requested
    cut_points = np.unique(
        [weighted_quantile(predictions, float(q), effective) for q in probabilities]
    )
    raw_bins = np.searchsorted(cut_points, predictions, side="left")
    occupied = np.unique(raw_bins[effective > 0])
    remap = {int(old): new + 1 for new, old in enumerate(occupied)}
    return np.asarray(
        [remap.get(int(value), 0) for value in raw_bins], dtype=np.float64
    )
