import pytest

from acteval import (
    gini,
    lift,
    lift_by_quantile,
    normalized_gini,
    risk_group_lift,
)
from acteval.exceptions import InputValidationError


def test_normalized_gini_known_orderings() -> None:
    observed = [0, 1, 2, 4]
    assert normalized_gini(observed, observed) == pytest.approx(1)
    assert normalized_gini(observed, [4, 2, 1, 0]) == pytest.approx(-1)
    assert gini(observed, [1, 1, 1, 1]) == pytest.approx(0)


def test_gini_is_invariant_to_order_inside_prediction_ties() -> None:
    first = gini([0, 3, 1, 4], [1, 1, 2, 3], sample_weight=[1, 2, 1, 1])
    second = gini([3, 0, 1, 4], [1, 1, 2, 3], sample_weight=[2, 1, 1, 1])
    assert first == pytest.approx(second)


def test_normalized_gini_rejects_constant_observations() -> None:
    with pytest.raises(InputValidationError, match="perfect-ranking Gini is zero"):
        normalized_gini([1, 1, 1], [1, 2, 3])


def test_lift_by_quantile_known_monotonic_example() -> None:
    table = lift_by_quantile([1, 2, 4, 8], [1, 2, 4, 8], n_bins=2)
    assert table.effective_bins == 2
    assert table.bins[1].lift > 1
    assert table.bins[0].lift < 1
    assert table.to_dataframe().shape == (2, 6)


def test_top_and_high_to_low_lift() -> None:
    observed = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    predicted = list(observed)
    assert lift(observed, predicted, top_fraction=0.2) > 1
    assert risk_group_lift(observed, predicted, fraction=0.2) > 1


def test_risk_group_lift_rejects_zero_low_group() -> None:
    with pytest.raises(InputValidationError, match="low-group"):
        risk_group_lift([0, 0, 2, 4], [1, 2, 3, 4], fraction=0.25)
