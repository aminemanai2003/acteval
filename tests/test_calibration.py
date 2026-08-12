import pytest

from acteval import ae_ratio, calibration_by_quantile, weighted_calibration_error
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


def test_calibration_by_quantile_known_bins() -> None:
    table = calibration_by_quantile(
        [0, 1, 2, 3],
        [0.1, 0.9, 2.1, 2.9],
        n_bins=2,
    )
    assert table.requested_bins == 2
    assert table.effective_bins == 2
    assert table.bins[0].count == 2
    assert table.bins[0].mean_observed == pytest.approx(0.5)
    assert table.bins[0].mean_prediction == pytest.approx(0.5)
    assert table.bins[0].ae_ratio == pytest.approx(1)
    assert table.bins[1].mean_observed == pytest.approx(2.5)


def test_calibration_ties_collapse_bins() -> None:
    table = calibration_by_quantile([0, 1, 2, 3], [1, 1, 1, 1], n_bins=4)
    assert table.effective_bins == 1
    assert table.bins[0].count == 4


def test_calibration_records_weighted_exposure() -> None:
    table = calibration_by_quantile(
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        n_bins=2,
        exposure=[1, 2, 3, 4],
        sample_weight=[2, 1, 1, 1],
    )
    assert sum(row.exposure or 0 for row in table.bins) == pytest.approx(11)
    assert sum(row.weight for row in table.bins) == pytest.approx(11)


def test_weighted_calibration_error_definition() -> None:
    value = weighted_calibration_error(
        [0, 2, 4, 6],
        [1, 1, 3, 3],
        n_bins=2,
    )
    assert value == pytest.approx(1)


def test_calibration_table_converts_to_dataframe() -> None:
    frame = calibration_by_quantile(
        [0, 1, 2, 3], [0.1, 0.9, 2.1, 2.9], n_bins=2
    ).to_dataframe()
    assert list(frame.columns) == [
        "bin",
        "count",
        "weight",
        "exposure",
        "mean_prediction",
        "mean_observed",
        "ae_ratio",
    ]
