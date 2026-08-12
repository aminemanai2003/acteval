import numpy as np
import pytest

from acteval.exceptions import InputValidationError
from acteval.validation import combine_weights, validate_inputs


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([[1, 2]], "one-dimensional"),
        ([], "must not be empty"),
        ([1, np.inf], "finite"),
        ([1, np.nan], "finite"),
    ],
)
def test_invalid_y_true(values: object, message: str) -> None:
    with pytest.raises(InputValidationError, match=message):
        validate_inputs(values, [1, 2])


def test_negative_sample_weight_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="nonnegative"):
        validate_inputs([1, 2], [1, 2], sample_weight=[1, -1])


def test_zero_exposure_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="positive value"):
        validate_inputs([1, 2], [1, 2], exposure=[0, 0])


def test_disjoint_positive_weights_are_rejected() -> None:
    inputs = validate_inputs(
        [1, 2],
        [1, 2],
        sample_weight=[1, 0],
        exposure=[0, 1],
    )
    with pytest.raises(InputValidationError, match="combined weight"):
        combine_weights(inputs)
