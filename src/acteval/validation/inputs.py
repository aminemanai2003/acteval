"""Validation and normalization of array-like metric inputs."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from acteval.exceptions import InputValidationError
from acteval.types import NumericArray, Task

Domain = Literal["real", "nonnegative", "positive"]
VALID_TASKS: tuple[Task, ...] = (
    "claim_frequency",
    "claim_severity",
    "pure_premium",
)


@dataclass(frozen=True, slots=True)
class ValidatedInputs:
    """One-dimensional, finite arrays ready for metric computation."""

    y_true: NumericArray
    y_pred: NumericArray
    sample_weight: NumericArray | None
    exposure: NumericArray | None


def as_1d_float_array(values: ArrayLike, *, name: str) -> NumericArray:
    """Convert numeric array-like input into a finite float64 vector."""
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
    array = as_1d_float_array(values, name=name)
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
    observed = as_1d_float_array(y_true, name="y_true")
    predicted = as_1d_float_array(y_pred, name="y_pred")
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


def validate_task(task: str) -> Task:
    """Validate and normalize an actuarial task name."""
    normalized = task.strip().lower()
    if normalized not in VALID_TASKS:
        choices = ", ".join(VALID_TASKS)
        raise InputValidationError(
            f"Unknown task {task!r}. Expected one of: {choices}."
        )
    return normalized


def validate_probability(value: float, *, name: str) -> float:
    """Validate a finite probability strictly between zero and one."""
    if not np.isfinite(value) or not 0 < value < 1:
        raise InputValidationError(f"{name} must be strictly between 0 and 1.")
    return float(value)


def validate_n_bins(n_bins: int) -> int:
    """Validate a requested number of risk bins."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise InputValidationError(
            "n_bins must be an integer greater than or equal to 2."
        )
    return n_bins
