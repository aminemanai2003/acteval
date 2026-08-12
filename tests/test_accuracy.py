import math

import pytest

from acteval import mae, rmse
from acteval.exceptions import InputValidationError


def test_mae_known_value() -> None:
    assert mae([1, 2, 4], [0, 2, 7]) == pytest.approx(4 / 3)


def test_mae_weighted() -> None:
    assert mae([1, 2, 4], [0, 2, 7], sample_weight=[1, 2, 1]) == pytest.approx(1)


def test_rmse_known_value() -> None:
    assert rmse([1, 2, 4], [0, 2, 7]) == pytest.approx(math.sqrt(10 / 3))


def test_rmse_weighted() -> None:
    assert rmse([1, 2, 4], [0, 2, 7], sample_weight=[1, 2, 1]) == pytest.approx(
        math.sqrt(2.5)
    )


@pytest.mark.parametrize("metric", [mae, rmse])
def test_perfect_prediction_is_zero(metric: object) -> None:
    assert metric([1, 2, 3], [1, 2, 3]) == pytest.approx(0)  # type: ignore[operator]


def test_accuracy_rejects_mismatched_lengths() -> None:
    with pytest.raises(InputValidationError, match="length"):
        mae([1, 2], [1])
