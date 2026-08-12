"""Validation and normalization of array-like metric inputs."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from acteval.exceptions import InputValidationError
from acteval.types import NumericArray

Domain = Literal["real", "nonnegative", "positive"]


@dataclass(frozen=True, slots=True)
class ValidatedInputs:
    """One-dimensional, finite arrays ready for metric computation."""

    y_true: NumericArray
    y_pred: NumericArray
    sample_weight: NumericArray | None
    exposure: NumericArray | None


def _as_1d_float_array(values: ArrayLike, *, name: str) -> NumericArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InputValidationError(f"{name} must be numeric.") from error
    if array.ndim != 1:
        raise InputValidationError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise InputValidationError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise InputValidationError(f"{name} must contain only finite values.")
    return array


def _validate_domain(array: NumericArray, *, name: str, domain: Domain) -> None:
    if domain == "nonnegative" and np.any(array < 0):
        raise InputValidationError(f"{name} must contain only nonnegative values.")
    if domain == "positive" and np.any(array <= 0):
        raise InputValidationError(
            f"{name} must contain only strictly positive values."
        )


def _validate_weight(
    values: ArrayLike | None,
    *,
    name: str,
    expected_length: int,
) -> NumericArray | None:
    if values is None:
        return None
    array = _as_1d_float_array(values, name=name)
    if len(array) != expected_length:
        raise InputValidationError(
            f"{name} has length {len(array)}; expected {expected_length}."
        )
    if np.any(array < 0):
        raise InputValidationError(f"{name} must contain only nonnegative values.")
    if not np.any(array > 0):
        raise InputValidationError(f"{name} must contain at least one positive value.")
    return array


def validate_inputs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
    y_true_domain: Domain = "real",
    y_pred_domain: Domain = "real",
) -> ValidatedInputs:
    """Validate common point-prediction metric inputs.

    Arrays must be non-empty, one-dimensional, finite, and equally sized.
    Weights and exposures must be nonnegative and have positive total mass.
    """
    observed = _as_1d_float_array(y_true, name="y_true")
    predicted = _as_1d_float_array(y_pred, name="y_pred")
    if len(observed) != len(predicted):
        raise InputValidationError(
            f"y_true has length {len(observed)}; y_pred has length {len(predicted)}."
        )
    _validate_domain(observed, name="y_true", domain=y_true_domain)
    _validate_domain(predicted, name="y_pred", domain=y_pred_domain)
    weights = _validate_weight(
        sample_weight, name="sample_weight", expected_length=len(observed)
    )
    exposures = _validate_weight(
        exposure, name="exposure", expected_length=len(observed)
    )
    return ValidatedInputs(observed, predicted, weights, exposures)


def combine_weights(inputs: ValidatedInputs) -> NumericArray | None:
    """Combine sample weights and exposure using elementwise multiplication."""
    if inputs.sample_weight is None:
        return inputs.exposure
    if inputs.exposure is None:
        return inputs.sample_weight
    combined = inputs.sample_weight * inputs.exposure
    if not np.any(combined > 0):
        raise InputValidationError(
            "sample_weight and exposure have no observations with positive "
            "combined weight."
        )
    return combined
