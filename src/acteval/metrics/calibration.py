"""Calibration metrics for actuarial point predictions."""

import numpy as np
from numpy.typing import ArrayLike

from acteval.exceptions import InputValidationError
from acteval.registry import register_metric
from acteval.validation import combine_weights, validate_inputs


@register_metric(
    name="ae_ratio",
    tasks=("claim_frequency", "claim_severity", "pure_premium"),
    category="calibration",
    higher_is_better=None,
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
