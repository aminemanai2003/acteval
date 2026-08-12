import numpy as np
import pytest

import acteval as ae
from acteval.exceptions import InputValidationError


@pytest.fixture
def frequency_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.arange(1, 21, dtype=float)
    predicted = observed * 0.9 + 0.2
    exposure = np.linspace(0.5, 2.0, len(observed))
    return observed, predicted, exposure


def test_evaluate_default_frequency_milestone(
    frequency_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, exposure = frequency_data
    result = ae.evaluate(
        observed,
        predicted,
        exposure=exposure,
        task="claim_frequency",
    )
    assert set(result.metrics) == {
        "rmse",
        "poisson_deviance",
        "ae_ratio",
        "normalized_gini",
        "tail_mae_95",
        "tail_ae_95",
    }
    assert result.metadata["weighting"] == "exposure"
    assert result.metadata["metric_specs"]["tail_mae_95"]["parameters"] == {
        "quantile": 0.95
    }
    assert "rmse" in result.summary()
    assert result.to_dataframe().shape == (6, 5)
    assert result.to_dict()["task"] == "claim_frequency"


def test_evaluate_parameterized_metrics_and_alias(
    frequency_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, _ = frequency_data
    result = ae.evaluate(
        observed,
        predicted,
        task="claim_frequency",
        metrics=[
            "rmse",
            "tail_ae_90",
            ae.MetricSpec("tail_mae", {"quantile": 0.8}, label="tail_mae_80"),
        ],
    )
    assert set(result.metrics) == {"rmse", "tail_ae_90", "tail_mae_80"}


def test_evaluate_pure_premium_records_tweedie_power(
    frequency_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, _ = frequency_data
    result = ae.evaluate(observed, predicted, task="pure_premium")
    details = result.metadata["metric_specs"]["tweedie_deviance_p1_5"]
    assert details["parameters"] == {"power": 1.5}


def test_compare_initial_milestone_and_rank(
    frequency_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, exposure = frequency_data
    comparison = ae.compare(
        observed,
        {"Model A": predicted, "Model B": predicted * 1.1},
        exposure=exposure,
        task="claim_frequency",
    )
    frame = comparison.to_dataframe()
    assert frame.shape == (6, 2)
    expected_best = frame.loc["rmse"].idxmin()
    assert comparison.rank("rmse").iloc[0]["model"] == expected_best
    assert comparison.rank("ae_ratio").iloc[0]["model"] in {
        "Model A",
        "Model B",
    }
    assert "Model A" in comparison.summary()
    assert comparison.to_dict()["metadata"]["no_universal_best_model"] is True


def test_identical_models_produce_identical_metrics(
    frequency_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, _ = frequency_data
    comparison = ae.compare(
        observed,
        {"A": predicted, "B": predicted.copy()},
        task="claim_frequency",
    )
    assert comparison.results["A"].metrics == comparison.results["B"].metrics


@pytest.mark.parametrize("task", ["", "frequency", "unknown"])
def test_evaluate_rejects_invalid_task(task: str) -> None:
    with pytest.raises(InputValidationError, match="Unknown task"):
        ae.evaluate([1, 2], [1, 2], task=task, metrics=["rmse"])


def test_evaluate_rejects_unsupported_and_duplicate_metrics() -> None:
    with pytest.raises(InputValidationError, match="does not support"):
        ae.evaluate([1, 2], [1, 2], task="claim_frequency", metrics=["gamma_deviance"])
    with pytest.raises(InputValidationError, match="Duplicate metric"):
        ae.evaluate([1, 2], [1, 2], task="claim_frequency", metrics=["rmse", "rmse"])


def test_compare_rejects_empty_predictions() -> None:
    with pytest.raises(InputValidationError, match="at least one model"):
        ae.compare([1, 2], {}, task="claim_frequency")
