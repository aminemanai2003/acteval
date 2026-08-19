import numpy as np
import pytest

import acteval as ae
from acteval.exceptions import InputValidationError


@pytest.fixture
def monitoring_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray([0, 1, 2, 1, 3, 4, 2, 5, 6, 4, 7, 8], dtype=float)
    predicted = observed * 0.9 + 0.2
    segments = np.asarray(["Retail"] * 6 + ["Commercial"] * 6)
    periods = np.asarray(["2025-Q2"] * 4 + ["2025-Q1"] * 4 + ["2025-Q3"] * 4)
    return observed, predicted, segments, periods


def test_evaluate_and_compare_by_segment(
    monitoring_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, segments, _ = monitoring_data
    result = ae.evaluate_by_segment(
        observed,
        predicted,
        segments,
        task="claim_frequency",
        metrics=["rmse", "ae_ratio"],
    )
    assert set(result.segments) == {"Retail", "Commercial"}
    assert result.to_dataframe().shape == (4, 1)
    comparison = ae.compare_by_segment(
        observed,
        {"A": predicted, "B": predicted * 1.1},
        segments,
        task="claim_frequency",
        metrics=["rmse"],
    )
    assert comparison.to_dataframe().shape == (2, 2)
    assert comparison.metadata["no_universal_best_model"] is True
    with pytest.raises(TypeError):
        result.segments["changed"] = result.segments["Retail"]  # type: ignore[index]
    with pytest.raises(TypeError):
        comparison.metadata["changed"] = True  # type: ignore[index]


def test_segment_evaluation_skips_small_groups(
    monitoring_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, _, _ = monitoring_data
    segments = np.asarray(["small"] + ["large"] * 11)
    result = ae.evaluate_by_segment(
        observed,
        predicted,
        segments,
        task="claim_frequency",
        metrics=["rmse"],
        min_observations=2,
    )
    assert set(result.segments) == {"large"}
    assert result.metadata["skipped_segments"] == ("small",)


def test_temporal_evaluation_sorts_periods_and_reports_change(
    monitoring_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, _, periods = monitoring_data
    result = ae.evaluate_over_time(
        observed,
        predicted,
        periods,
        task="claim_frequency",
        metrics=["rmse"],
    )
    assert tuple(result.periods) == ("2025-Q1", "2025-Q2", "2025-Q3")
    assert result.baseline_period == "2025-Q1"
    baseline_row = result.to_dataframe().loc[("2025-Q1", "rmse")]
    assert baseline_row["change_from_baseline"] == pytest.approx(0)


def test_prediction_drift_known_behavior_and_weights() -> None:
    reference = np.linspace(0, 1, 100)
    identical = ae.prediction_drift(reference, reference.copy(), n_bins=5)
    assert identical.population_stability_index == pytest.approx(0)
    shifted = ae.prediction_drift(
        reference,
        reference + 0.5,
        n_bins=5,
        reference_weight=np.ones(100),
        current_weight=np.linspace(1, 2, 100),
    )
    assert shifted.population_stability_index > 0
    assert shifted.mean_shift > 0
    assert sum(row.psi_contribution for row in shifted.bins) == pytest.approx(
        shifted.population_stability_index
    )
    assert "PSI=" in shifted.summary()


def test_monitoring_rejects_invalid_inputs(
    monitoring_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, segments, _ = monitoring_data
    with pytest.raises(InputValidationError, match="expected 12"):
        ae.evaluate_by_segment(
            observed,
            predicted,
            segments[:-1],
            task="claim_frequency",
            metrics=["rmse"],
        )
    with pytest.raises(InputValidationError, match="No segment"):
        ae.evaluate_by_segment(
            observed,
            predicted,
            segments,
            task="claim_frequency",
            metrics=["rmse"],
            min_observations=100,
        )
    with pytest.raises(InputValidationError, match="missing or empty"):
        ae.evaluate_by_segment(
            observed,
            predicted,
            [None, *list(segments[1:])],
            task="claim_frequency",
            metrics=["rmse"],
        )
    with pytest.raises(InputValidationError, match="positive integer"):
        ae.compare_by_segment(
            observed,
            {"A": predicted, "B": predicted},
            segments,
            task="claim_frequency",
            metrics=["rmse"],
            min_observations=0,
        )
    with pytest.raises(InputValidationError, match="epsilon"):
        ae.prediction_drift([1, 2], [1, 2], epsilon=0)
