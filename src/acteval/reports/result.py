"""Single-model evaluation result."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
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

    def to_html(self, *, title: str = "ActEval evaluation") -> str:
        """Render this result as a standalone HTML report."""

        from acteval.reporting import render_html_report

        return render_html_report(self, title=title)

    def save_html(self, path: str | Path, *, title: str = "ActEval evaluation") -> Path:
        """Write a standalone HTML report and return its resolved path."""

        from acteval.reporting import save_html_report

        return save_html_report(self, path, title=title)
