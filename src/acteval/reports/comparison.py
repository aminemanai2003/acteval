"""Multi-model evaluation result."""

from dataclasses import dataclass
from typing import Any

from acteval.exceptions import InputValidationError
from acteval.reports.result import EvaluationResult
from acteval.types import Task


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Aligned evaluation results for several prediction arrays."""

    task: Task
    results: dict[str, EvaluationResult]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly nested representation."""
        return {
            "task": self.task,
            "models": {
                model: result.to_dict() for model, result in self.results.items()
            },
            "metadata": dict(self.metadata),
        }

    def to_dataframe(self) -> Any:
        """Return metrics as rows and model names as columns."""
        import pandas as pd

        frame = pd.DataFrame(
            {model: result.metrics for model, result in self.results.items()}
        )
        frame.index.name = "metric"
        return frame

    def summary(self) -> str:
        """Return a concise printable model-comparison table."""
        return str(self.to_dataframe().to_string())

    def rank(self, metric: str) -> Any:
        """Rank models using the metric's registered direction or target."""
        import pandas as pd

        if not self.results:
            raise InputValidationError("Cannot rank an empty comparison.")
        first = next(iter(self.results.values()))
        if metric not in first.metrics:
            raise InputValidationError(f"Metric {metric!r} is not in this comparison.")
        details = first.metadata["metric_specs"][metric]
        values = {
            model: result.metrics[metric] for model, result in self.results.items()
        }
        ranking = pd.DataFrame({"model": list(values), "value": list(values.values())})
        target = details.get("target")
        direction = details.get("higher_is_better")
        if target is not None:
            ranking["distance_to_target"] = (ranking["value"] - target).abs()
            ranking = ranking.sort_values(
                ["distance_to_target", "model"], kind="stable"
            )
        elif direction is not None:
            ranking = ranking.sort_values(
                ["value", "model"], ascending=[not direction, True], kind="stable"
            )
        else:
            raise InputValidationError(
                f"Metric {metric!r} has neither an optimization direction nor target."
            )
        ranking = ranking.reset_index(drop=True)
        ranking["rank"] = ranking.index + 1
        return ranking
