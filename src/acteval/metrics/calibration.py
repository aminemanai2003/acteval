"""Calibration metrics for actuarial point predictions."""

import numpy as np
from numpy.typing import ArrayLike

from acteval.exceptions import InputValidationError
from acteval.registry import register_metric
from acteval.reports import CalibrationBin, CalibrationTable
from acteval.utils import effective_weights, risk_bin_indices
from acteval.validation import combine_weights, validate_inputs


@register_metric(
    name="ae_ratio",
    tasks=("claim_frequency", "claim_severity", "pure_premium"),
    category="calibration",
    higher_is_better=None,
    target=1.0,
    description="Aggregate actual-to-expected ratio; the calibration target is 1.",
)
def ae_ratio(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    exposure: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Compute the aggregate actual-to-expected (A/E) ratio.

    The definition is ``sum(w * y_true) / sum(w * y_pred)``. Here ``w`` is 1
    by default, exposure when supplied, sample weight when supplied, or their
    product when both are supplied. Inputs must be nonnegative and on the same
    scale. A value near 1 indicates aggregate calibration, above 1 indicates
    aggregate underprediction, and below 1 indicates aggregate overprediction.

    Aggregate A/E can hide offsetting miscalibration across risk segments; it
    should later be paired with calibration-by-quantile diagnostics.
    """
    inputs = validate_inputs(
        y_true,
        y_pred,
        exposure=exposure,
        sample_weight=sample_weight,
        y_true_domain="nonnegative",
        y_pred_domain="nonnegative",
    )
    weights = combine_weights(inputs)
    if weights is None:
        actual = float(np.sum(inputs.y_true))
        expected = float(np.sum(inputs.y_pred))
    else:
        actual = float(np.sum(weights * inputs.y_true))
        expected = float(np.sum(weights * inputs.y_pred))
    if expected <= 0:
        raise InputValidationError(
            "A/E ratio is undefined because aggregate expected value is zero."
        )
    return actual / expected


def calibration_by_quantile(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    n_bins: int = 10,
    exposure: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> CalibrationTable:
    """Aggregate calibration from low to high predicted-risk quantiles.

    Bins are formed using weighted prediction quantiles. Identical prediction
    values are never split, so ``effective_bins`` can be lower than
    ``n_bins``. Means use ``sample_weight * exposure`` when both are supplied.
    A bin with zero aggregate expected value receives ``NaN`` A/E.
    """
    inputs = validate_inputs(
        y_true,
        y_pred,
        exposure=exposure,
        sample_weight=sample_weight,
        y_true_domain="nonnegative",
        y_pred_domain="nonnegative",
    )
    combined = combine_weights(inputs)
    weights = effective_weights(len(inputs.y_true), combined)
    bins = risk_bin_indices(
        inputs.y_pred,
        n_bins=n_bins,
        weights=weights,
    )
    rows: list[CalibrationBin] = []
    for bin_number in sorted(int(value) for value in np.unique(bins) if value > 0):
        mask = (bins == bin_number) & (weights > 0)
        bin_weights = weights[mask]
        weight_sum = float(np.sum(bin_weights))
        observed = float(np.average(inputs.y_true[mask], weights=bin_weights))
        predicted = float(np.average(inputs.y_pred[mask], weights=bin_weights))
        if inputs.exposure is None:
            exposure_sum = None
        elif inputs.sample_weight is None:
            exposure_sum = float(np.sum(inputs.exposure[mask]))
        else:
            exposure_sum = float(
                np.sum(inputs.exposure[mask] * inputs.sample_weight[mask])
            )
        rows.append(
            CalibrationBin(
                bin=bin_number,
                count=int(np.count_nonzero(mask)),
                weight=weight_sum,
                exposure=exposure_sum,
                mean_prediction=predicted,
                mean_observed=observed,
                ae_ratio=observed / predicted if predicted > 0 else float("nan"),
            )
        )
    return CalibrationTable(tuple(rows), requested_bins=n_bins)


@register_metric(
    name="weighted_calibration_error",
    tasks=("claim_frequency", "claim_severity", "pure_premium"),
    category="calibration",
    higher_is_better=False,
    description="Risk-bin weighted absolute observed-versus-expected difference.",
)
def weighted_calibration_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    n_bins: int = 10,
    exposure: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Compute weighted absolute calibration error across prediction bins.

    For bin ``b`` with total effective weight ``W_b``, observed mean ``O_b``
    and predicted mean ``P_b``, the metric is
    ``sum_b W_b * abs(O_b - P_b) / sum_b W_b``. It has the target's unit and
    is deliberately not called ECE because it is a regression calibration
    diagnostic rather than classification expected calibration error.
    """
    table = calibration_by_quantile(
        y_true,
        y_pred,
        n_bins=n_bins,
        exposure=exposure,
        sample_weight=sample_weight,
    )
    total_weight = sum(row.weight for row in table.bins)
    return float(
        sum(
            row.weight * abs(row.mean_observed - row.mean_prediction)
            for row in table.bins
        )
        / total_weight
    )
