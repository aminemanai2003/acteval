import numpy as np
import pytest

from acteval.exceptions import InputValidationError
from acteval.validation import (
    combine_weights,
    validate_input_scale,
    validate_inputs,
    validate_n_bins,
    validate_probability,
    validate_task,
)


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


def test_task_probability_and_bin_validation() -> None:
    assert validate_task(" Claim_Frequency ") == "claim_frequency"
    assert validate_probability(0.25, name="q") == pytest.approx(0.25)
    assert validate_n_bins(2) == 2
    with pytest.raises(InputValidationError, match="Unknown task"):
        validate_task("frequency")
    with pytest.raises(InputValidationError, match="between 0 and 1"):
        validate_probability(1, name="q")
    with pytest.raises(InputValidationError, match="greater than or equal"):
        validate_n_bins(1)


def test_input_scale_contract_rejects_ambiguous_exposure() -> None:
    assert validate_input_scale(None, exposure=None) == "aggregate"
    assert validate_input_scale("rate", exposure=[1, 2]) == "rate"
    with pytest.raises(InputValidationError, match="input_scale='rate'"):
        validate_input_scale(None, exposure=[1, 2])
    with pytest.raises(InputValidationError, match="must not be multiplied"):
        validate_input_scale("aggregate", exposure=[1, 2])
    with pytest.raises(InputValidationError, match="'aggregate' or 'rate'"):
        validate_input_scale("unknown", exposure=None)
