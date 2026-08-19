"""Shared public types."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

Task = Literal["claim_frequency", "claim_severity", "pure_premium"]
InputScale = Literal["aggregate", "rate"]
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
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        if self.label is not None:
            normalized_label = self.label.strip()
            if not normalized_label:
                raise ValueError("MetricSpec label must not be empty.")
            object.__setattr__(self, "label", normalized_label)


class PredictiveDistribution(Protocol):
    """Vectorized predictive distributions, one distribution per observation.

    Scalar quantiles return shape ``(n_observations,)``. A vector of quantiles
    returns ``(n_quantiles, n_observations)``. Samples always return
    ``(n_samples, n_observations)``.
    """

    @property
    def n_observations(self) -> int:
        """Number of per-observation predictive distributions."""
        ...

    def cdf(self, x: ArrayLike) -> NumericArray:
        """Evaluate the cumulative distribution function at ``x``."""
        ...

    def quantile(self, q: ArrayLike) -> NumericArray:
        """Evaluate predictive quantiles at probabilities ``q``."""
        ...

    def sample(
        self,
        n: int,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> NumericArray:
        """Draw ``n`` samples from each predictive distribution."""
        ...

    def log_prob(self, y: ArrayLike) -> NumericArray:
        """Evaluate per-observation log probability or log density."""
        ...

    def mean(self) -> NumericArray:
        """Return predictive means."""
        ...

    def variance(self) -> NumericArray:
        """Return predictive variances."""
        ...

    def entropy(self) -> NumericArray:
        """Return distribution-specific predictive entropy."""
        ...
