"""Lightweight registry for metric discovery and metadata."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

from acteval.exceptions import MetricRegistrationError, UnknownMetricError
from acteval.types import Task

MetricCategory = Literal[
    "accuracy",
    "calibration",
    "discrimination",
    "probabilistic",
    "uncertainty",
    "tail_risk",
]
MetricFunction = Callable[..., float]
F = TypeVar("F", bound=MetricFunction)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Metadata and callable associated with a registered metric."""

    name: str
    category: MetricCategory
    supported_tasks: tuple[Task, ...]
    higher_is_better: bool | None
    requires_distribution: bool
    description: str
    reference: str | None
    function: MetricFunction

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        """Evaluate the registered metric."""
        return self.function(*args, **kwargs)


_REGISTRY: dict[str, MetricDefinition] = {}


def register_metric(
    *,
    name: str,
    tasks: Iterable[Task],
    category: MetricCategory,
    higher_is_better: bool | None,
    description: str,
    requires_distribution: bool = False,
    reference: str | None = None,
) -> Callable[[F], F]:
    """Register a metric function and its evaluation metadata."""
    normalized_name = name.strip().lower()
    supported_tasks = tuple(tasks)
    if not normalized_name:
        raise MetricRegistrationError("Metric name must not be empty.")
    if not supported_tasks:
        raise MetricRegistrationError(
            f"Metric {normalized_name!r} must support at least one task."
        )

    def decorator(function: F) -> F:
        if normalized_name in _REGISTRY:
            raise MetricRegistrationError(
                f"Metric {normalized_name!r} is already registered."
            )
        _REGISTRY[normalized_name] = MetricDefinition(
            name=normalized_name,
            category=category,
            supported_tasks=supported_tasks,
            higher_is_better=higher_is_better,
            requires_distribution=requires_distribution,
            description=description,
            reference=reference,
            function=cast(MetricFunction, function),
        )
        return function

    return decorator


def get_metric(name: str) -> MetricDefinition:
    """Return a registered metric by its case-insensitive name."""
    normalized_name = name.strip().lower()
    try:
        return _REGISTRY[normalized_name]
    except KeyError as error:
        raise UnknownMetricError(f"Unknown metric: {name!r}.") from error


def list_metrics(*, task: Task | None = None) -> tuple[MetricDefinition, ...]:
    """List registered metrics, optionally filtered by supported task."""
    metrics = tuple(_REGISTRY.values())
    if task is not None:
        metrics = tuple(metric for metric in metrics if task in metric.supported_tasks)
    return tuple(sorted(metrics, key=lambda metric: metric.name))
