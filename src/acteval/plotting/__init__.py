"""Optional plotting separated from metric computation."""

from acteval.plotting.calibration import plot_calibration
from acteval.plotting.diagnostics import (
    plot_lift,
    plot_residuals,
    plot_tail_diagnostics,
)

__all__ = [
    "plot_calibration",
    "plot_lift",
    "plot_residuals",
    "plot_tail_diagnostics",
]
