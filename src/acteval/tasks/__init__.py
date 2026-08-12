"""Definitions and defaults for supported actuarial tasks."""

from acteval.tasks.defaults import DEFAULT_METRICS
from acteval.tasks.definition import TaskDefinition
from acteval.tasks.frequency import CLAIM_FREQUENCY
from acteval.tasks.pure_premium import PURE_PREMIUM
from acteval.tasks.severity import CLAIM_SEVERITY
from acteval.types import Task

TASK_DEFINITIONS: dict[Task, TaskDefinition] = {
    definition.name: definition
    for definition in (CLAIM_FREQUENCY, CLAIM_SEVERITY, PURE_PREMIUM)
}


def get_task_definition(task: Task) -> TaskDefinition:
    """Return the immutable definition for a supported task."""

    return TASK_DEFINITIONS[task]


__all__ = [
    "CLAIM_FREQUENCY",
    "CLAIM_SEVERITY",
    "DEFAULT_METRICS",
    "PURE_PREMIUM",
    "TASK_DEFINITIONS",
    "TaskDefinition",
    "get_task_definition",
]
