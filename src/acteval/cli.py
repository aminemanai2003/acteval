"""Command-line interface for evaluating prediction CSV files."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from acteval._version import __version__
from acteval.api import evaluate
from acteval.exceptions import ActEvalError
from acteval.reporting import export_table

_TASKS = ("claim_frequency", "claim_severity", "pure_premium")


class _CLIError(Exception):
    """An input or output problem that should be shown without a traceback."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acteval",
        description="Evaluate actuarial prediction columns from a CSV file.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="evaluate one observed and predicted column",
        description=(
            "Evaluate one prediction column against observed outcomes. "
            "CSV columns default to y_true and y_pred."
        ),
    )
    evaluate_parser.add_argument("csv", type=Path, help="input CSV file")
    evaluate_parser.add_argument(
        "--task",
        required=True,
        choices=_TASKS,
        help="actuarial task that determines metric semantics and defaults",
    )
    evaluate_parser.add_argument(
        "--observed",
        default="y_true",
        metavar="COLUMN",
        help="observed outcome column (default: y_true)",
    )
    evaluate_parser.add_argument(
        "--predicted",
        default="y_pred",
        metavar="COLUMN",
        help="prediction column (default: y_pred)",
    )
    evaluate_parser.add_argument(
        "--exposure",
        metavar="COLUMN",
        help="optional exposure column; requires --input-scale rate",
    )
    evaluate_parser.add_argument(
        "--sample-weight",
        metavar="COLUMN",
        help="optional sample-weight column",
    )
    evaluate_parser.add_argument(
        "--input-scale",
        choices=("aggregate", "rate"),
        help="scale of observed and predicted values",
    )
    evaluate_parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        metavar="NAME",
        help="metric name; repeat to select several (default: task metrics)",
    )
    evaluate_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        metavar="PATH",
        help="write .csv, .json, or .html instead of printing a table",
    )
    evaluate_parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output file",
    )
    evaluate_parser.set_defaults(handler=_evaluate_csv)
    return parser


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (
        OSError,
        UnicodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        raise _CLIError(f"could not read {path}: {error}") from error


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        available = ", ".join(str(column) for column in frame.columns) or "none"
        raise _CLIError(
            f"column {name!r} was not found; available columns: {available}"
        )
    return frame[name]


def _validate_output(input_path: Path, output_path: Path, *, force: bool) -> None:
    if input_path.resolve() == output_path.resolve():
        raise _CLIError("output path must not replace the input CSV")
    if output_path.exists() and not force:
        raise _CLIError(
            f"output already exists: {output_path}; pass --force to replace it"
        )


def _evaluate_csv(arguments: argparse.Namespace) -> int:
    if arguments.output is None and arguments.force:
        raise _CLIError("--force requires --output")
    if arguments.output is not None:
        _validate_output(arguments.csv, arguments.output, force=arguments.force)

    frame = _read_csv(arguments.csv)
    observed = _column(frame, arguments.observed)
    predicted = _column(frame, arguments.predicted)
    exposure = (
        None if arguments.exposure is None else _column(frame, arguments.exposure)
    )
    sample_weight = (
        None
        if arguments.sample_weight is None
        else _column(frame, arguments.sample_weight)
    )
    result = evaluate(
        observed,
        predicted,
        task=arguments.task,
        exposure=exposure,
        input_scale=arguments.input_scale,
        sample_weight=sample_weight,
        metrics=arguments.metrics,
        context={"source": str(arguments.csv)},
    )

    if arguments.output is None:
        print(result.summary())
        return 0

    try:
        destination = export_table(result, arguments.output)
    except ValueError as error:
        raise _CLIError(str(error)) from error
    print(destination)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ActEval command line and return a process exit status."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (_CLIError, ActEvalError) as error:
        print(f"acteval: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through the entry point
    raise SystemExit(main())
