import pytest

from acteval import large_loss_bias, tail_ae_ratio, tail_mae, tail_rmse
from acteval.exceptions import InputValidationError


def test_tail_metrics_known_threshold() -> None:
    observed = [1, 3, 5]
    predicted = [1, 2, 4]
    assert tail_mae(observed, predicted, threshold=2) == pytest.approx(1)
    assert tail_rmse(observed, predicted, threshold=2) == pytest.approx(1)
    assert tail_ae_ratio(observed, predicted, threshold=2) == pytest.approx(4 / 3)
    assert large_loss_bias(observed, predicted, threshold=2) == pytest.approx(0.75)


def test_tail_metrics_weight_selected_observations() -> None:
    result = tail_mae(
        [1, 3, 5],
        [1, 1, 4],
        threshold=2,
        sample_weight=[10, 1, 3],
    )
    assert result == pytest.approx(1.25)


def test_tail_quantile_defaults_to_95_percent() -> None:
    observed = list(range(1, 21))
    predicted = [value - 1 for value in observed]
    assert tail_mae(observed, predicted) == pytest.approx(1)


def test_tail_rejects_threshold_and_quantile_together() -> None:
    with pytest.raises(InputValidationError, match="not both"):
        tail_mae([1, 2, 3], [1, 2, 3], threshold=1, quantile=0.9)


def test_tail_rejects_empty_selection() -> None:
    with pytest.raises(InputValidationError, match="No positive-weight"):
        tail_rmse([1, 2, 3], [1, 2, 3], threshold=3)


def test_tail_ae_rejects_zero_expectation() -> None:
    with pytest.raises(InputValidationError, match="expectation is zero"):
        tail_ae_ratio([1, 2, 3], [0, 0, 0], threshold=2)
