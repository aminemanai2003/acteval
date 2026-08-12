"""Single-model evaluation result."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from acteval.types import Task


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metrics and reproducibility metadata for one prediction array."""

    task: Task
    metrics: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-friendly representation."""
        return {
            "task": self.task,
            "metrics": dict(self.metrics),
            "metadata": deepcopy(self.metadata),
        }

    def to_dataframe(self) -> Any:
        """Return one row per metric as a pandas DataFrame."""
        import pandas as pd

        specifications = self.metadata.get("metric_specs", {})
        rows: list[dict[str, Any]] = []
        for label, value in self.metrics.items():
            details = specifications.get(label, {})
            rows.append(
                {
                    "metric": label,
                    "value": value,
                    "category": details.get("category"),
                    "higher_is_better": details.get("higher_is_better"),
                    "target": details.get("target"),
                    "parameters": details.get("parameters", {}),
                }
            )
        return pd.DataFrame.from_records(rows).set_index("metric")

    def summary(self) -> str:
        """Return a concise printable metric table."""
        return str(self.to_dataframe().to_string())
