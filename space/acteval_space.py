"""Validated data handling for the public ActEval Space.

This module contains no Gradio code so its security and evaluation behavior can
be tested with the ordinary project test suite.
"""

from __future__ import annotations

import csv
import tempfile
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

import acteval as ae

MAX_FILE_SIZE_BYTES: Final = 5 * 1024 * 1024
MAX_ROWS: Final = 50_000
MAX_COLUMNS: Final = 100
MAX_FRAME_MEMORY_BYTES: Final = 64 * 1024 * 1024
REPORT_MAX_AGE_SECONDS: Final = 60 * 60
PREVIEW_ROWS: Final = 8
REPORT_DIRECTORY: Final = Path(tempfile.gettempdir()) / "acteval-space-reports"


class SpaceInputError(ValueError):
    """An upload or selection problem safe to show to a Space visitor."""


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Bounded metadata returned after inspecting an uploaded CSV."""

    rows: int
    columns: tuple[str, ...]
    preview: pd.DataFrame


@dataclass(frozen=True, slots=True)
class EvaluationArtifacts:
    """User-facing outputs produced by one successful evaluation."""

    metrics: pd.DataFrame
    calibration: pd.DataFrame
    summary_html: str
    report_path: Path


def _validated_upload_path(upload: str | Path | None) -> Path:
    if upload is None:
        raise SpaceInputError("Upload a CSV file before running an evaluation.")
    path = Path(upload)
    if path.suffix.lower() != ".csv":
        raise SpaceInputError("Only uncompressed .csv files are accepted.")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise SpaceInputError("The uploaded file is no longer available.") from error
    if size == 0:
        raise SpaceInputError("The uploaded CSV is empty.")
    if size > MAX_FILE_SIZE_BYTES:
        raise SpaceInputError("The CSV exceeds the 5 MiB upload limit.")
    return path


def _read_header(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            raw_header = next(csv.reader(stream))
    except StopIteration as error:
        raise SpaceInputError("The uploaded CSV is empty.") from error
    except (OSError, UnicodeError, csv.Error) as error:
        raise SpaceInputError("The CSV must be valid UTF-8 text.") from error

    columns = tuple(name.strip() for name in raw_header)
    if not columns or any(not name for name in columns):
        raise SpaceInputError("Every CSV column must have a non-empty name.")
    if len(columns) > MAX_COLUMNS:
        raise SpaceInputError(f"The CSV exceeds the {MAX_COLUMNS}-column limit.")
    if len(columns) != len(set(columns)):
        raise SpaceInputError("CSV column names must be unique.")
    return columns


def read_upload(upload: str | Path | None) -> pd.DataFrame:
    """Read one bounded UTF-8 CSV after validating its shape and header."""

    path = _validated_upload_path(upload)
    columns = _read_header(path)
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", nrows=MAX_ROWS + 1)
    except (
        OSError,
        UnicodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        raise SpaceInputError("The file could not be parsed as a CSV.") from error

    if len(frame) > MAX_ROWS:
        raise SpaceInputError(f"The CSV exceeds the {MAX_ROWS:,}-row limit.")
    if frame.empty:
        raise SpaceInputError("The CSV must contain at least one data row.")
    if len(frame.columns) != len(columns):
        raise SpaceInputError("The CSV rows do not match the header width.")
    frame.columns = list(columns)
    memory_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    if memory_bytes > MAX_FRAME_MEMORY_BYTES:
        raise SpaceInputError("The parsed CSV exceeds the in-memory data limit.")
    return frame


def profile_upload(upload: str | Path | None) -> DatasetProfile:
    """Return safe column choices and a small preview for one upload."""

    frame = read_upload(upload)
    return DatasetProfile(
        rows=len(frame),
        columns=tuple(str(column) for column in frame.columns),
        preview=frame.head(PREVIEW_ROWS).copy(),
    )


def suggested_column(columns: tuple[str, ...], role: str) -> str | None:
    """Choose a conventional column for a role without guessing by position."""

    candidates = {
        "observed": ("y_true", "observed", "actual", "outcome"),
        "predicted": ("y_pred", "predicted", "prediction", "estimate"),
        "exposure": ("exposure", "earned_exposure"),
        "sample_weight": ("sample_weight", "weight"),
    }
    normalized = {column.casefold(): column for column in columns}
    return next(
        (normalized[name] for name in candidates[role] if name in normalized),
        None,
    )


def _numeric_column(frame: pd.DataFrame, name: str | None, role: str) -> pd.Series:
    if name is None or not str(name).strip():
        raise SpaceInputError(f"Select the {role} column.")
    if name not in frame.columns:
        raise SpaceInputError(f"The selected {role} column is not in the CSV.")
    try:
        return pd.to_numeric(frame[name], errors="raise")
    except (TypeError, ValueError) as error:
        raise SpaceInputError(
            f"The {role} column must contain only numbers."
        ) from error


def _optional_numeric_column(
    frame: pd.DataFrame, name: str | None, role: str
) -> pd.Series | None:
    if name is None or not str(name).strip():
        return None
    return _numeric_column(frame, name, role)


def _metric_table(result: ae.EvaluationResult) -> pd.DataFrame:
    metrics = result.to_dataframe().reset_index()

    def objective(row: pd.Series) -> str:
        target = row["target"]
        if pd.notna(target):
            return f"target {float(target):g}"
        direction = row["higher_is_better"]
        if direction is True:
            return "higher is better"
        if direction is False:
            return "lower is better"
        return "diagnostic"

    return pd.DataFrame(
        {
            "Metric": metrics["metric"],
            "Value": metrics["value"],
            "Role": metrics["category"].str.replace("_", " ").str.title(),
            "Interpretation": metrics.apply(objective, axis=1),
        }
    )


def _calibration_table(
    observed: pd.Series,
    predicted: pd.Series,
    *,
    exposure: pd.Series | None,
    sample_weight: pd.Series | None,
) -> pd.DataFrame:
    observed_values = observed.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    weights = np.ones(len(observed_values), dtype=float)
    if exposure is not None:
        weights *= exposure.to_numpy(dtype=float)
    if sample_weight is not None:
        weights *= sample_weight.to_numpy(dtype=float)

    band_count = min(10, len(observed_values))
    ranked = pd.Series(predicted_values).rank(method="first")
    bands = pd.qcut(ranked, q=band_count, labels=False, duplicates="drop")
    source = pd.DataFrame(
        {
            "band": bands.to_numpy(),
            "observed": observed_values,
            "predicted": predicted_values,
            "weight": weights,
        }
    )
    rows: list[dict[str, float | int | str]] = []
    for band, group in source.groupby("band", observed=True, sort=True):
        group_weights = group["weight"].to_numpy(dtype=float)
        if float(group_weights.sum()) <= 0:
            continue
        for series, column in (("Observed", "observed"), ("Predicted", "predicted")):
            rows.append(
                {
                    "Prediction band": int(band) + 1,
                    "Series": series,
                    "Mean value": float(
                        np.average(
                            group[column].to_numpy(dtype=float), weights=group_weights
                        )
                    ),
                    "Observations": len(group),
                }
            )
    return pd.DataFrame.from_records(rows)


def _summary_html(result: ae.EvaluationResult) -> str:
    ae_ratio = result.metrics.get("ae_ratio")
    ae_text = "not selected" if ae_ratio is None else f"{float(ae_ratio):.4f}"
    values = (
        ("Observations", f"{int(result.metadata['n_observations']):,}"),
        ("A/E ratio", ae_text),
        ("Input scale", str(result.metadata["input_scale"]).replace("_", " ")),
        ("Weighting", str(result.metadata["weighting"]).replace("_", " ")),
    )
    cells = "".join(
        '<div class="evidence-item">'
        f'<span class="evidence-label">{escape(label)}</span>'
        f"<strong>{escape(value)}</strong></div>"
        for label, value in values
    )
    return f'<div class="evidence-strip" aria-label="Evaluation summary">{cells}</div>'


def _remove_stale_reports() -> None:
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - REPORT_MAX_AGE_SECONDS
    for report in REPORT_DIRECTORY.glob("acteval-report-*.html"):
        try:
            if report.stat().st_mtime < cutoff:
                report.unlink()
        except OSError:
            continue


def _write_report(result: ae.EvaluationResult) -> Path:
    _remove_stale_reports()
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="acteval-report-",
        suffix=".html",
        dir=REPORT_DIRECTORY,
        delete=False,
    ) as stream:
        destination = Path(stream.name)
    return result.save_html(destination, title="ActEval interactive evaluation")


def evaluate_upload(
    upload: str | Path | None,
    *,
    task: str,
    observed_column: str | None,
    predicted_column: str | None,
    exposure_column: str | None = None,
    sample_weight_column: str | None = None,
    input_scale: str = "aggregate",
) -> EvaluationArtifacts:
    """Evaluate one uploaded CSV using task defaults and bounded resources."""

    frame = read_upload(upload)
    observed = _numeric_column(frame, observed_column, "observed")
    predicted = _numeric_column(frame, predicted_column, "predicted")
    exposure = _optional_numeric_column(frame, exposure_column, "exposure")
    sample_weight = _optional_numeric_column(
        frame, sample_weight_column, "sample-weight"
    )
    if observed_column == predicted_column:
        raise SpaceInputError("Observed and predicted columns must be different.")
    if input_scale not in {"aggregate", "rate"}:
        raise SpaceInputError("Input scale must be aggregate or rate.")
    if exposure is not None and input_scale != "rate":
        raise SpaceInputError("Select rate input scale when using exposure.")

    result = ae.evaluate(
        observed,
        predicted,
        task=task,
        exposure=exposure,
        sample_weight=sample_weight,
        input_scale=input_scale,
        context={"source": "hugging-face-space"},
    )
    return EvaluationArtifacts(
        metrics=_metric_table(result),
        calibration=_calibration_table(
            observed,
            predicted,
            exposure=exposure,
            sample_weight=sample_weight,
        ),
        summary_html=_summary_html(result),
        report_path=_write_report(result),
    )
