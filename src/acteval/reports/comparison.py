"""Multi-model evaluation result."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acteval.exceptions import InputValidationError
from acteval.reports.result import EvaluationResult
from acteval.types import Task
from acteval.utils import freeze_mapping, thaw


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Aligned evaluation results for several prediction arrays."""

    task: Task
    results: Mapping[str, EvaluationResult]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Detach and freeze result and metadata mappings."""
        object.__setattr__(self, "results", freeze_mapping(self.results))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly nested representation."""
        return {
            "task": self.task,
            "models": {
                model: result.to_dict() for model, result in self.results.items()
            },
            "metadata": thaw(self.metadata),
        }

    def to_dataframe(self) -> Any:
        """Return metrics as rows and model names as columns."""
        import pandas as pd

        frame = pd.DataFrame(
            {model: dict(result.metrics) for model, result in self.results.items()}
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
            ranking["rank"] = (
                ranking["distance_to_target"]
                .rank(method="min", ascending=True)
                .astype(int)
            )
            ranking = ranking.sort_values(
                ["distance_to_target", "model"], kind="stable"
            )
        elif direction is not None:
            ranking["rank"] = (
                ranking["value"].rank(method="min", ascending=not direction).astype(int)
            )
            ranking = ranking.sort_values(
                ["value", "model"], ascending=[not direction, True], kind="stable"
            )
        else:
            raise InputValidationError(
                f"Metric {metric!r} has neither an optimization direction nor target."
            )
        ranking = ranking.reset_index(drop=True)
        return ranking

    def to_html(self, *, title: str = "ActEval model comparison") -> str:
        """Render this comparison as a standalone HTML report."""

        from acteval.reporting import render_html_report

        return render_html_report(self, title=title)

    def save_html(
        self, path: str | Path, *, title: str = "ActEval model comparison"
    ) -> Path:
        """Write a standalone HTML report and return its resolved path."""

        from acteval.reporting import save_html_report

        return save_html_report(self, path, title=title)
