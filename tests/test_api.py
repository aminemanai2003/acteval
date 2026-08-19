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
        input_scale="rate",
        task="claim_frequency",
    )
    assert set(result.metrics) == {
        "rmse",
        "poisson_deviance",
        "ae_ratio",
        "normalized_gini",
    }
    assert result.metadata["weighting"] == "exposure"
    assert result.metadata["prediction_functional"] == "mean"
    assert "rmse" in result.summary()
    assert result.to_dataframe().shape == (4, 5)
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


def test_explicit_pure_premium_tweedie_records_power(
    frequency_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, _ = frequency_data
    result = ae.evaluate(
        observed,
        predicted,
        task="pure_premium",
        metrics=[
            ae.MetricSpec(
                "tweedie_deviance", {"power": 1.7}, label="tweedie_deviance_p1_7"
            )
        ],
    )
    details = result.metadata["metric_specs"]["tweedie_deviance_p1_7"]
    assert details["parameters"] == {"power": 1.7}


def test_point_evaluation_rejects_mixed_prediction_functionals() -> None:
    with pytest.raises(InputValidationError, match="incompatible prediction"):
        ae.evaluate(
            [1, 1, 1, 101],
            [26, 26, 26, 26],
            task="claim_severity",
            metrics=["mae", "rmse"],
        )

    median_result = ae.evaluate(
        [1, 1, 1, 101],
        [1, 1, 1, 1],
        task="claim_severity",
        metrics=["mae"],
    )
    assert median_result.metadata["prediction_functional"] == "median"


def test_compare_initial_milestone_and_rank(
    frequency_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    observed, predicted, exposure = frequency_data
    comparison = ae.compare(
        observed,
        {"Model A": predicted, "Model B": predicted * 1.1},
        exposure=exposure,
        input_scale="rate",
        task="claim_frequency",
    )
    frame = comparison.to_dataframe()
    assert frame.shape == (4, 2)
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
    ranking = comparison.rank("rmse")
    assert ranking.loc[ranking["model"] == "A", "rank"].item() == 1
    assert ranking.loc[ranking["model"] == "B", "rank"].item() == 1


def test_explicit_tail_evaluation_supports_tied_maximum_counts() -> None:
    result = ae.evaluate(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        task="claim_frequency",
        metrics=["tail_mae_95"],
    )
    assert result.metrics["tail_mae_95"] == pytest.approx(0.15)


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


def test_high_level_api_requires_explicit_rate_scale_with_exposure() -> None:
    with pytest.raises(InputValidationError, match="input_scale='rate'"):
        ae.evaluate(
            [1, 2],
            [1, 2],
            task="claim_frequency",
            exposure=[1, 2],
            metrics=["rmse"],
        )
    with pytest.raises(InputValidationError, match="must not be multiplied"):
        ae.evaluate(
            [1, 2],
            [1, 2],
            task="claim_frequency",
            exposure=[1, 2],
            input_scale="aggregate",
            metrics=["rmse"],
        )
    result = ae.evaluate(
        [1, 2],
        [1, 2],
        task="claim_frequency",
        exposure=[1, 2],
        input_scale="rate",
        metrics=["rmse"],
    )
    assert result.metadata["input_scale"] == "rate"


def test_public_api_exports_are_resolvable_and_version_is_stable() -> None:
    assert ae.__version__ == "2.0.0"
    assert ae.__all__
    assert len(ae.__all__) == len(set(ae.__all__))
    assert all(hasattr(ae, name) for name in ae.__all__)


def test_metric_signature_validation_is_cached(
    frequency_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    from acteval.api import _metric_signature

    observed, predicted, _ = frequency_data
    _metric_signature.cache_clear()
    ae.evaluate(observed, predicted, task="claim_frequency", metrics=["rmse"])
    first = _metric_signature.cache_info()
    ae.evaluate(observed, predicted, task="claim_frequency", metrics=["rmse"])
    second = _metric_signature.cache_info()
    assert first.misses == 1
    assert second.hits == first.hits + 1


def test_results_and_metric_specs_are_deeply_immutable_and_detached() -> None:
    parameters = {"quantile": 0.9}
    specification = ae.MetricSpec("tail_mae", parameters)
    parameters["quantile"] = 0.8
    assert specification.parameters["quantile"] == pytest.approx(0.9)
    with pytest.raises(TypeError):
        specification.parameters["quantile"] = 0.7  # type: ignore[index]

    metrics = {"rmse": 1.0}
    metadata: dict[str, object] = {"metric_specs": {"rmse": {"parameters": {}}}}
    result = ae.EvaluationResult("claim_frequency", metrics, metadata)
    metrics["rmse"] = 2.0
    mutable_specs = metadata["metric_specs"]
    assert isinstance(mutable_specs, dict)
    mutable_specs["rmse"]["parameters"]["changed"] = True
    assert result.metrics["rmse"] == pytest.approx(1.0)
    assert "changed" not in result.metadata["metric_specs"]["rmse"]["parameters"]
    with pytest.raises(TypeError):
        result.metadata["metric_specs"]["rmse"]["parameters"]["changed"] = True

    detached = result.to_dict()
    detached["metadata"]["metric_specs"]["rmse"]["parameters"]["changed"] = True
    assert "changed" not in result.metadata["metric_specs"]["rmse"]["parameters"]


def test_evaluation_records_versions_input_identity_and_caller_context() -> None:
    first = ae.evaluate(
        [1, 2, 3],
        [1, 2, 2.5],
        task="claim_frequency",
        metrics=["rmse"],
        context={"model_id": "glm-v4", "split_seed": 42, "split": "holdout"},
    )
    repeated = ae.evaluate(
        [1, 2, 3],
        [1, 2, 2.5],
        task="claim_frequency",
        metrics=["rmse"],
    )
    changed = ae.evaluate(
        [1, 2, 3],
        [1, 2, 2.6],
        task="claim_frequency",
        metrics=["rmse"],
    )
    assert first.metadata["acteval_version"] == ae.__version__
    assert first.metadata["python_version"]
    assert first.metadata["dependency_versions"]["numpy"]
    assert first.metadata["evaluation_context"]["model_id"] == "glm-v4"
    assert first.metadata["input_fingerprint"] == repeated.metadata["input_fingerprint"]
    assert first.metadata["input_fingerprint"] != changed.metadata["input_fingerprint"]
    with pytest.raises(InputValidationError, match="context keys"):
        ae.evaluate(
            [1, 2],
            [1, 2],
            task="claim_frequency",
            metrics=["rmse"],
            context={"": "invalid"},
        )
