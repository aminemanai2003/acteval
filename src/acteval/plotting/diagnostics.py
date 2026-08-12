"""Lift, residual, and observed-tail diagnostic plots."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from acteval.exceptions import InputValidationError
from acteval.metrics.discrimination import lift_by_quantile
from acteval.utils import effective_weights, weighted_quantile
from acteval.validation import combine_weights, validate_inputs


def plot_lift(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    n_bins: int = 10,
    exposure: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    ax: Any = None,
) -> Any:
    """Plot observed lift from low to high predicted-risk quantiles."""
    import matplotlib.pyplot as plt

    table = lift_by_quantile(
        y_true,
        y_pred,
        n_bins=n_bins,
        exposure=exposure,
        sample_weight=sample_weight,
    )
    if ax is None:
        _, ax = plt.subplots()
    bins = [row.bin for row in table.bins]
    values = [row.lift for row in table.bins]
    ax.plot(bins, values, marker="o")
    ax.axhline(1.0, linestyle="--", color="black")
    ax.set_xlabel("Predicted-risk bin (low to high)")
    ax.set_ylabel("Observed lift versus portfolio mean")
    ax.set_title("Lift by predicted-risk quantile")
    return ax


def plot_residuals(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    ax: Any = None,
) -> Any:
    """Plot raw residuals ``observed - predicted`` against predictions."""
    import matplotlib.pyplot as plt

    inputs = validate_inputs(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots()
    residuals = inputs.y_true - inputs.y_pred
    ax.scatter(inputs.y_pred, residuals, alpha=0.7)
    ax.axhline(0.0, linestyle="--", color="black")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Observed - predicted")
    ax.set_title("Residuals versus predictions")
    return ax


def plot_tail_diagnostics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    quantile: float = 0.95,
    exposure: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    ax: Any = None,
) -> Any:
    """Plot observed versus predicted values above an observed quantile."""
    import matplotlib.pyplot as plt

    inputs = validate_inputs(
        y_true,
        y_pred,
        exposure=exposure,
        sample_weight=sample_weight,
        y_true_domain="nonnegative",
        y_pred_domain="nonnegative",
    )
    weights = effective_weights(len(inputs.y_true), combine_weights(inputs))
    cutoff = weighted_quantile(inputs.y_true, quantile, weights)
    mask = (inputs.y_true > cutoff) & (weights > 0)
    if not np.any(mask):
        raise InputValidationError(
            f"No positive-weight observations are above quantile {quantile}."
        )
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(inputs.y_true[mask], inputs.y_pred[mask], s=20 + weights[mask])
    lower = float(min(np.min(inputs.y_true[mask]), np.min(inputs.y_pred[mask])))
    upper = float(max(np.max(inputs.y_true[mask]), np.max(inputs.y_pred[mask])))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black")
    ax.set_xlabel("Observed tail value")
    ax.set_ylabel("Predicted value")
    ax.set_title(f"Observed-tail diagnostics (q={quantile:g})")
    return ax
