import pytest

import acteval as ae


@pytest.mark.parametrize(
    "task",
    ["claim_frequency", "claim_severity", "pure_premium"],
)
def test_task_definitions_match_evaluation_defaults(task: ae.Task) -> None:
    definition = ae.get_task_definition(task)
    assert definition.name == task
    assert definition.target
    assert definition.prediction_domain
    assert definition.exposure_interpretation
    assert definition.default_metrics
