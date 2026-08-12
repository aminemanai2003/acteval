"""Shared public types."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

Task = Literal["claim_frequency", "claim_severity", "pure_premium"]
NumericArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Select a metric and record explicit parameter values.

    Parameters are copied on construction so later mutation of a caller's
    dictionary cannot change an evaluation request.
    """

    name: str
    parameters: Mapping[str, int | float] = field(default_factory=dict)
    label: str | None = None

    def __post_init__(self) -> None:
        """Normalize and defensively copy specification fields."""
        normalized_name = self.name.strip().lower()
        if not normalized_name:
            raise ValueError("MetricSpec name must not be empty.")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "parameters", dict(self.parameters))
        if self.label is not None:
            normalized_label = self.label.strip()
            if not normalized_label:
                raise ValueError("MetricSpec label must not be empty.")
            object.__setattr__(self, "label", normalized_label)


class PredictiveDistribution(Protocol):
    """Minimal future-facing interface for probabilistic predictions.

    The protocol is architectural only in v0.1. No probabilistic metrics are
    registered until their numerical contracts are specified and tested.
    """

    def cdf(self, x: ArrayLike) -> NumericArray:
        """Evaluate the cumulative distribution function at ``x``."""
        ...

    def quantile(self, q: ArrayLike) -> NumericArray:
        """Evaluate predictive quantiles at probabilities ``q``."""
        ...

    def sample(self, n: int) -> NumericArray:
        """Draw ``n`` samples from each predictive distribution."""
        ...
