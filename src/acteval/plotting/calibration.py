"""Calibration plotting functions."""

from typing import Any

from numpy.typing import ArrayLike

from acteval.metrics.calibration import calibration_by_quantile


def plot_calibration(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    n_bins: int = 10,
    exposure: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    ax: Any = None,
) -> Any:
    """Plot observed against predicted means by ascending risk quantile."""
    import matplotlib.pyplot as plt

    table = calibration_by_quantile(
        y_true,
        y_pred,
        n_bins=n_bins,
        exposure=exposure,
        sample_weight=sample_weight,
    )
    if ax is None:
        _, ax = plt.subplots()
    predicted = [row.mean_prediction for row in table.bins]
    observed = [row.mean_observed for row in table.bins]
    ax.plot(predicted, observed, marker="o", label="Risk bins")
    lower = min([*predicted, *observed])
    upper = max([*predicted, *observed])
    ax.plot([lower, upper], [lower, upper], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean prediction")
    ax.set_ylabel("Mean observed")
    ax.set_title("Calibration by predicted-risk quantile")
    ax.legend()
    return ax
