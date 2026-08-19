from pathlib import Path

import numpy as np
import pytest

import acteval as ae
from acteval.exceptions import InputValidationError


@pytest.fixture
def inference_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    predicted = np.linspace(0.2, 5.0, 80)
    observed = rng.poisson(predicted).astype(float)
    exposure = np.linspace(0.5, 1.5, len(observed))
    return observed, predicted, exposure


def test_bootstrap_evaluate_is_reproducible_and_contains_point_estimate(
    inference_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, exposure = inference_data
    first = ae.bootstrap_evaluate(
        observed,
        predicted,
        task="claim_frequency",
        exposure=exposure,
        input_scale="rate",
        metrics=["rmse", "ae_ratio"],
        n_resamples=100,
        random_state=17,
    )
    second = ae.bootstrap_evaluate(
        observed,
        predicted,
        task="claim_frequency",
        exposure=exposure,
        input_scale="rate",
        metrics=["rmse", "ae_ratio"],
        n_resamples=100,
        random_state=17,
    )
    assert first.to_dict() == second.to_dict()
    assert first.intervals["rmse"].estimate == pytest.approx(
        ae.rmse(observed, predicted, exposure=exposure)
    )
    assert first.intervals["rmse"].lower <= first.intervals["rmse"].upper
    assert first.to_dataframe().shape == (2, 7)
    assert "rmse" in first.summary()


def test_paired_bootstrap_compare_uses_objective_aware_delta(
    inference_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, _ = inference_data
    result = ae.paired_bootstrap_compare(
        observed,
        {"reference": predicted * 1.8, "candidate": predicted},
        task="claim_frequency",
        reference="reference",
        metrics=["rmse", "ae_ratio"],
        n_resamples=100,
        random_state=5,
    )
    rmse = next(item for item in result.comparisons if item.metric == "rmse")
    assert rmse.objective_delta < 0
    assert rmse.raw_delta < 0
    assert result.metadata["negative_objective_delta_favors_candidate"] is True
    assert result.to_dataframe().shape[0] == 2


def test_identical_models_have_zero_paired_interval(
    inference_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, _ = inference_data
    result = ae.paired_bootstrap_compare(
        observed,
        {"A": predicted, "B": predicted.copy()},
        task="claim_frequency",
        metrics=["rmse"],
        n_resamples=100,
        random_state=8,
    )
    comparison = result.comparisons[0]
    assert comparison.objective_delta == pytest.approx(0)
    assert comparison.lower == pytest.approx(0)
    assert comparison.upper == pytest.approx(0)
    assert comparison.confidence_excludes_zero is False


def test_bootstrap_calibration_uses_fixed_bins_and_intervals(
    inference_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, exposure = inference_data
    result = ae.bootstrap_calibration_by_quantile(
        observed,
        predicted,
        n_bins=5,
        exposure=exposure,
        input_scale="rate",
        n_resamples=100,
        random_state=3,
    )
    assert result.requested_bins == 5
    assert result.effective_bins == 5
    assert result.to_dataframe().shape == (5, 10)
    for row in result.bins:
        assert row.mean_observed.lower <= row.mean_observed.upper
        assert row.ae_ratio.n_resamples == 100
    assert result.metadata["method"] == "fixed_bin_stratified_percentile_bootstrap"


def test_bootstrap_calibration_retries_zero_predicted_resamples() -> None:
    observed = [0, 0, 0, 1, 2, 3, 4, 5]
    predicted = [0, 0, 0, 0.1, 1, 2, 3, 4]
    result = ae.bootstrap_calibration_by_quantile(
        observed,
        predicted,
        n_bins=2,
        n_resamples=100,
        random_state=4,
    )
    assert result.metadata["failed_resamples"] >= 0


def test_bootstrap_supports_discrimination_lift_and_tail_metrics(
    inference_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, exposure = inference_data
    result = ae.bootstrap_evaluate(
        observed,
        predicted,
        task="claim_frequency",
        exposure=exposure,
        input_scale="rate",
        metrics=["normalized_gini", "lift", "tail_mae_80", "tail_ae_80"],
        n_resamples=100,
        random_state=19,
    )
    assert set(result.intervals) == {
        "normalized_gini",
        "lift",
        "tail_mae_80",
        "tail_ae_80",
    }
    assert all(interval.n_resamples == 100 for interval in result.intervals.values())


def test_paired_bootstrap_reverses_higher_is_better_objective() -> None:
    observed = np.arange(1, 61, dtype=float)
    candidate = observed.copy()
    reference = observed[::-1].copy()
    result = ae.paired_bootstrap_compare(
        observed,
        {"reference": reference, "candidate": candidate},
        task="claim_frequency",
        reference="reference",
        metrics=["normalized_gini"],
        n_resamples=100,
        random_state=12,
    )
    comparison = result.comparisons[0]
    assert comparison.raw_delta > 0
    assert comparison.objective_delta < 0


def test_save_interval_csv(
    tmp_path: Path,
    inference_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, _ = inference_data
    result = ae.bootstrap_evaluate(
        observed,
        predicted,
        task="claim_frequency",
        metrics=["rmse"],
        n_resamples=100,
    )
    destination = ae.save_interval_csv(result, tmp_path / "nested" / "intervals.csv")
    assert destination.exists()
    assert "standard_error" in destination.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_resamples": 99}, "at least 100"),
        ({"confidence_level": 1.0}, "strictly between"),
    ],
)
def test_bootstrap_rejects_invalid_configuration(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(InputValidationError, match=message):
        ae.bootstrap_evaluate(
            [1, 2, 3],
            [1, 2, 3],
            task="claim_frequency",
            metrics=["rmse"],
            **kwargs,
        )


def test_paired_comparison_rejects_unknown_or_only_reference() -> None:
    with pytest.raises(InputValidationError, match="Unknown reference"):
        ae.paired_bootstrap_compare(
            [1, 2, 3],
            {"A": [1, 2, 3], "B": [1, 2, 3]},
            task="claim_frequency",
            reference="missing",
            metrics=["rmse"],
            n_resamples=100,
        )
    with pytest.raises(InputValidationError, match="non-empty"):
        ae.paired_bootstrap_compare(
            [1, 2, 3],
            {"A": [1, 2, 3], "B": [1, 2, 3]},
            task="claim_frequency",
            reference="",
            metrics=["rmse"],
            n_resamples=100,
        )
    with pytest.raises(InputValidationError, match="At least two"):
        ae.paired_bootstrap_compare(
            [1, 2, 3],
            {"A": [1, 2, 3]},
            task="claim_frequency",
            metrics=["rmse"],
            n_resamples=100,
        )
