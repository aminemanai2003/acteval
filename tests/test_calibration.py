import pytest

from acteval import ae_ratio
from acteval.exceptions import InputValidationError


def test_ae_ratio_perfect_prediction() -> None:
    assert ae_ratio([0, 1, 3], [0, 1, 3]) == pytest.approx(1)


def test_ae_ratio_known_weighted_value() -> None:
    assert ae_ratio([1, 3], [2, 2], exposure=[1, 3]) == pytest.approx(1.25)


def test_ae_ratio_combines_exposure_and_sample_weight() -> None:
    result = ae_ratio(
        [1, 3],
        [2, 2],
        exposure=[1, 3],
        sample_weight=[2, 1],
    )
    assert result == pytest.approx(11 / 10)


def test_ae_ratio_is_invariant_to_exposure_scaling() -> None:
    base = ae_ratio([1, 3], [2, 2], exposure=[1, 3])
    scaled = ae_ratio([1, 3], [2, 2], exposure=[10, 30])
    assert scaled == pytest.approx(base)


def test_ae_ratio_rejects_zero_expected_total() -> None:
    with pytest.raises(InputValidationError, match="expected value is zero"):
        ae_ratio([1, 2], [0, 0])


def test_ae_ratio_rejects_negative_values() -> None:
    with pytest.raises(InputValidationError, match="nonnegative"):
        ae_ratio([1, -1], [1, 1])
