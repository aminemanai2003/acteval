"""Metadata describing the supported actuarial evaluation tasks."""

from dataclasses import dataclass

from acteval.types import MetricSpec, Task


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """Document a task's target, prediction domain, and default metrics."""

    name: Task
    target: str
    prediction_domain: str
    exposure_interpretation: str
    default_metrics: tuple[MetricSpec, ...]
