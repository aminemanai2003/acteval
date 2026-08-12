"""Dependency-light export and standalone HTML reporting utilities."""

import json
from html import escape
from pathlib import Path
from typing import Any, Protocol


class TabularResult(Protocol):
    """Structural contract used by result export helpers."""

    def to_dataframe(self) -> Any:
        """Return the primary result table."""
        ...

    def to_dict(self) -> Any:
        """Return JSON-friendly result content."""
        ...


_STYLE = """
body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem auto;
       max-width: 1100px; padding: 0 1rem; color: #172033; }
h1 { color: #075985; } h2 { margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
th, td { border: 1px solid #d8dee9; padding: 0.5rem 0.65rem; text-align: right; }
th { background: #f0f7fa; } tbody tr:nth-child(even) { background: #f8fafc; }
pre { background: #f6f8fa; border: 1px solid #d8dee9; border-radius: 6px;
      padding: 1rem; overflow-x: auto; }
.note { border-left: 4px solid #0284c7; padding: 0.75rem 1rem; background: #f0f9ff; }
""".strip()


def render_html_report(
    result: TabularResult,
    *,
    title: str = "ActEval report",
    include_metadata: bool = True,
) -> str:
    """Render any tabular ActEval result as a standalone UTF-8 HTML document.

    The report has no JavaScript or remote assets, so it can be archived and
    reviewed offline. Content generated from labels and metadata is escaped.
    """

    frame = result.to_dataframe()
    table = frame.to_html(border=0, escape=True, classes="acteval-table")
    metadata_html = ""
    payload = result.to_dict()
    if include_metadata:
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        metadata_html = (
            "<h2>Reproducibility metadata</h2><pre>"
            + escape(json.dumps(metadata, indent=2, sort_keys=True, default=str))
            + "</pre>"
        )
    safe_title = escape(title)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width">'
        f"<title>{safe_title}</title>"
        f"<style>{_STYLE}</style></head><body><h1>{safe_title}</h1>"
        '<p class="note">Metrics represent distinct actuarial objectives; '
        "this report does not define a universal best model.</p>"
        f"<h2>Results</h2>{table}{metadata_html}</body></html>"
    )


def save_html_report(
    result: TabularResult,
    path: str | Path,
    *,
    title: str = "ActEval report",
    include_metadata: bool = True,
) -> Path:
    """Write a standalone HTML report and return its resolved path."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_html_report(result, title=title, include_metadata=include_metadata),
        encoding="utf-8",
    )
    return destination


def export_table(
    result: TabularResult,
    path: str | Path,
    *,
    format: str | None = None,
) -> Path:
    """Export a result table to CSV, JSON, or standalone HTML.

    The format defaults to the destination extension. JSON uses the full
    structured result while CSV exports the primary table.
    """

    destination = Path(path).expanduser().resolve()
    resolved_format = (format or destination.suffix.lstrip(".")).strip().lower()
    if resolved_format not in {"csv", "json", "html"}:
        raise ValueError("format must be one of: csv, json, html.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if resolved_format == "csv":
        result.to_dataframe().to_csv(destination)
    elif resolved_format == "json":
        destination.write_text(
            json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
        )
    else:
        destination.write_text(render_html_report(result), encoding="utf-8")
    return destination


def save_plot(axis: Any, path: str | Path, *, dpi: int = 150) -> Path:
    """Save a Matplotlib axis' figure with a tight bounding box."""

    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer.")
    if not hasattr(axis, "figure") or not hasattr(axis.figure, "savefig"):
        raise TypeError("axis must be a Matplotlib axis with a figure.")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    axis.figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    return destination
