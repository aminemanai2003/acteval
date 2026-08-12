"""Shared public types."""

from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

Task = Literal["claim_frequency", "claim_severity", "pure_premium"]
NumericArray = NDArray[np.float64]


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
