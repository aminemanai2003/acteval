import pytest

import acteval
from acteval.exceptions import UnknownMetricError


def test_foundation_metrics_are_registered() -> None:
    assert {metric.name for metric in acteval.list_metrics()} == {
        "ae_ratio",
        "gamma_deviance",
        "mae",
        "poisson_deviance",
        "rmse",
    }


def test_registry_filters_by_task() -> None:
    frequency_names = {
        metric.name for metric in acteval.list_metrics(task="claim_frequency")
    }
    assert frequency_names == {"ae_ratio", "mae", "poisson_deviance", "rmse"}


def test_registered_metric_is_callable() -> None:
    metric = acteval.get_metric("RMSE")
    assert metric([1, 2], [1, 2]) == pytest.approx(0)


def test_unknown_metric_has_clear_error() -> None:
    with pytest.raises(UnknownMetricError, match="Unknown metric"):
        acteval.get_metric("not-a-metric")
