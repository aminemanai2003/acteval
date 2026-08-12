"""Structured diagnostic tables that do not require plotting."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """Aggregated observed and expected values for one risk bin."""

    bin: int
    count: int
    weight: float
    exposure: float | None
    mean_prediction: float
    mean_observed: float
    ae_ratio: float


@dataclass(frozen=True, slots=True)
class CalibrationTable:
    """Calibration bins ordered from lowest to highest predicted risk."""

    bins: tuple[CalibrationBin, ...]
    requested_bins: int

    @property
    def effective_bins(self) -> int:
        """Number of populated bins after collapsing prediction ties."""
        return len(self.bins)

    def to_dict(self) -> list[dict[str, Any]]:
        """Return records suitable for JSON serialization."""
        return [asdict(row) for row in self.bins]

    def to_dataframe(self) -> Any:
        """Return a pandas DataFrame with one row per bin."""
        import pandas as pd

        return pd.DataFrame.from_records(self.to_dict())


@dataclass(frozen=True, slots=True)
class LiftBin:
    """Observed risk and lift for one prediction-ranked bin."""

    bin: int
    count: int
    weight: float
    mean_prediction: float
    mean_observed: float
    lift: float


@dataclass(frozen=True, slots=True)
class LiftTable:
    """Lift bins ordered from lowest to highest predicted risk."""

    bins: tuple[LiftBin, ...]
    requested_bins: int

    @property
    def effective_bins(self) -> int:
        """Number of populated bins after collapsing prediction ties."""
        return len(self.bins)

    def to_dict(self) -> list[dict[str, Any]]:
        """Return records suitable for JSON serialization."""
        return [asdict(row) for row in self.bins]

    def to_dataframe(self) -> Any:
        """Return a pandas DataFrame with one row per bin."""
        import pandas as pd

        return pd.DataFrame.from_records(self.to_dict())
