import pytest

import acteval
from acteval.exceptions import UnknownMetricError


def test_mvp_scalar_metrics_are_registered() -> None:
    assert {metric.name for metric in acteval.list_metrics()} == {
        "ae_ratio",
        "brier_score",
        "crps",
        "gamma_deviance",
        "gini",
        "interval_coverage",
        "interval_score",
        "interval_width",
        "lift",
        "log_score",
        "mae",
        "normalized_gini",
        "poisson_deviance",
        "predictive_entropy",
        "predictive_variance",
        "quantile_score",
        "rmse",
        "tail_ae_ratio",
        "tail_mae",
        "tail_rmse",
        "tweedie_deviance",
        "weighted_calibration_error",
    }


def test_registry_filters_by_task() -> None:
    frequency_names = {
        metric.name for metric in acteval.list_metrics(task="claim_frequency")
    }
    assert frequency_names == {
        "ae_ratio",
        "brier_score",
        "crps",
        "gini",
        "interval_coverage",
        "interval_score",
        "interval_width",
        "lift",
        "log_score",
        "mae",
        "normalized_gini",
        "poisson_deviance",
        "predictive_entropy",
        "predictive_variance",
        "quantile_score",
        "rmse",
        "tail_ae_ratio",
        "tail_mae",
        "tail_rmse",
        "weighted_calibration_error",
    }


def test_registered_metric_is_callable() -> None:
    metric = acteval.get_metric("RMSE")
    assert metric([1, 2], [1, 2]) == pytest.approx(0)


def test_unknown_metric_has_clear_error() -> None:
    with pytest.raises(UnknownMetricError, match="Unknown metric"):
        acteval.get_metric("not-a-metric")
