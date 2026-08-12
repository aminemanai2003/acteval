"""Structured results returned by ActEval diagnostics."""

from acteval.reports.comparison import ComparisonResult
from acteval.reports.result import EvaluationResult
from acteval.reports.tables import (
    CalibrationBin,
    CalibrationTable,
    LiftBin,
    LiftTable,
)

__all__ = [
    "CalibrationBin",
    "CalibrationTable",
    "ComparisonResult",
    "EvaluationResult",
    "LiftBin",
    "LiftTable",
]
