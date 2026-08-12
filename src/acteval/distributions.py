"""Vectorized predictive-distribution adapters for actuarial models."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from acteval.exceptions import InputValidationError
from acteval.types import NumericArray
from acteval.validation import as_1d_float_array

RandomState = int | np.random.Generator | None


def _parameter(
    values: ArrayLike,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> NumericArray:
    array = as_1d_float_array(values, name=name)
    if positive and np.any(array <= 0):
        raise InputValidationError(f"{name} must be strictly positive.")
    if nonnegative and np.any(array < 0):
        raise InputValidationError(f"{name} must be nonnegative.")
    return array


def _broadcast_parameters(*parameters: NumericArray) -> tuple[NumericArray, ...]:
    try:
        arrays = np.broadcast_arrays(*parameters)
    except ValueError as error:
        raise InputValidationError(
            "Distribution parameters must have equal lengths or length one."
        ) from error
    return tuple(np.asarray(array, dtype=np.float64) for array in arrays)


def _observations(values: ArrayLike, *, length: int, name: str) -> NumericArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return np.full(length, float(array), dtype=np.float64)
    if array.ndim != 1 or len(array) != length:
        raise InputValidationError(f"{name} must be scalar or have length {length}.")
    if not np.all(np.isfinite(array)):
        raise InputValidationError(f"{name} must contain only finite values.")
    return array


def _quantiles(
    q: ArrayLike,
    *,
    length: int,
    ppf: Callable[[np.ndarray], np.ndarray],
) -> NumericArray:
    probabilities = np.asarray(q, dtype=np.float64)
    if probabilities.ndim > 1 or not np.all(np.isfinite(probabilities)):
        raise InputValidationError(
            "q must be a finite scalar or one-dimensional array."
        )
    if np.any((probabilities <= 0) | (probabilities >= 1)):
        raise InputValidationError("q values must be strictly between 0 and 1.")
    if probabilities.ndim == 0:
        return np.asarray(ppf(probabilities), dtype=np.float64)
    return np.asarray(ppf(probabilities[:, None]), dtype=np.float64).reshape(
        len(probabilities), length
    )


def _sample_size(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise InputValidationError("n must be a positive integer.")
    return n


def _rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


class PoissonDistribution:
    """Independent Poisson predictive distributions parameterized by means."""

    def __init__(self, mu: ArrayLike) -> None:
        self.mu = _parameter(mu, name="mu", nonnegative=True)

    @property
    def n_observations(self) -> int:
        return len(self.mu)

    def cdf(self, x: ArrayLike) -> NumericArray:
        values = _observations(x, length=self.n_observations, name="x")
        return np.asarray(stats.poisson.cdf(values, self.mu), dtype=np.float64)

    def quantile(self, q: ArrayLike) -> NumericArray:
        return _quantiles(
            q,
            length=self.n_observations,
            ppf=lambda probabilities: stats.poisson.ppf(probabilities, self.mu),
        )

    def sample(self, n: int, *, random_state: RandomState = None) -> NumericArray:
        size = _sample_size(n)
        return (
            _rng(random_state)
            .poisson(self.mu, size=(size, self.n_observations))
            .astype(np.float64)
        )

    def log_prob(self, y: ArrayLike) -> NumericArray:
        values = _observations(y, length=self.n_observations, name="y")
        return np.asarray(stats.poisson.logpmf(values, self.mu), dtype=np.float64)

    def mean(self) -> NumericArray:
        return self.mu.copy()

    def variance(self) -> NumericArray:
        return self.mu.copy()

    def entropy(self) -> NumericArray:
        return np.asarray(stats.poisson.entropy(self.mu), dtype=np.float64)


class NegativeBinomialDistribution:
    """Negative Binomial distributions parameterized by mean and dispersion.

    The variance is ``mean + mean**2 / dispersion``. Larger dispersion tends
    toward Poisson variance.
    """

    def __init__(self, mean: ArrayLike, dispersion: ArrayLike) -> None:
        mean_array = _parameter(mean, name="mean", nonnegative=True)
        dispersion_array = _parameter(dispersion, name="dispersion", positive=True)
        self._mean, self.dispersion = _broadcast_parameters(
            mean_array, dispersion_array
        )
        self._probability = self.dispersion / (self.dispersion + self._mean)

    @property
    def n_observations(self) -> int:
        return len(self._mean)

    def cdf(self, x: ArrayLike) -> NumericArray:
        values = _observations(x, length=self.n_observations, name="x")
        return np.asarray(
            stats.nbinom.cdf(values, self.dispersion, self._probability),
            dtype=np.float64,
        )

    def quantile(self, q: ArrayLike) -> NumericArray:
        return _quantiles(
            q,
            length=self.n_observations,
            ppf=lambda probabilities: stats.nbinom.ppf(
                probabilities, self.dispersion, self._probability
            ),
        )

    def sample(self, n: int, *, random_state: RandomState = None) -> NumericArray:
        size = _sample_size(n)
        return (
            _rng(random_state)
            .negative_binomial(
                self.dispersion,
                self._probability,
                size=(size, self.n_observations),
            )
            .astype(np.float64)
        )

    def log_prob(self, y: ArrayLike) -> NumericArray:
        values = _observations(y, length=self.n_observations, name="y")
        return np.asarray(
            stats.nbinom.logpmf(values, self.dispersion, self._probability),
            dtype=np.float64,
        )

    def mean(self) -> NumericArray:
        return self._mean.copy()

    def variance(self) -> NumericArray:
        return self._mean + np.square(self._mean) / self.dispersion

    def entropy(self) -> NumericArray:
        return np.asarray(
            stats.nbinom.entropy(self.dispersion, self._probability),
            dtype=np.float64,
        )


class GammaDistribution:
    """Gamma predictive distributions parameterized by mean and shape."""

    def __init__(self, mean: ArrayLike, shape: ArrayLike) -> None:
        mean_array = _parameter(mean, name="mean", positive=True)
        shape_array = _parameter(shape, name="shape", positive=True)
        self._mean, self.shape = _broadcast_parameters(mean_array, shape_array)
        self.scale = self._mean / self.shape

    @property
    def n_observations(self) -> int:
        return len(self._mean)

    def cdf(self, x: ArrayLike) -> NumericArray:
        values = _observations(x, length=self.n_observations, name="x")
        return np.asarray(
            stats.gamma.cdf(values, self.shape, scale=self.scale), dtype=np.float64
        )

    def quantile(self, q: ArrayLike) -> NumericArray:
        return _quantiles(
            q,
            length=self.n_observations,
            ppf=lambda probabilities: stats.gamma.ppf(
                probabilities, self.shape, scale=self.scale
            ),
        )

    def sample(self, n: int, *, random_state: RandomState = None) -> NumericArray:
        size = _sample_size(n)
        return _rng(random_state).gamma(
            self.shape, self.scale, size=(size, self.n_observations)
        )

    def log_prob(self, y: ArrayLike) -> NumericArray:
        values = _observations(y, length=self.n_observations, name="y")
        return np.asarray(
            stats.gamma.logpdf(values, self.shape, scale=self.scale),
            dtype=np.float64,
        )

    def mean(self) -> NumericArray:
        return self._mean.copy()

    def variance(self) -> NumericArray:
        return np.square(self._mean) / self.shape

    def entropy(self) -> NumericArray:
        return np.asarray(
            stats.gamma.entropy(self.shape, scale=self.scale), dtype=np.float64
        )


class LognormalDistribution:
    """Lognormal distributions parameterized on the log scale."""

    def __init__(self, meanlog: ArrayLike, sdlog: ArrayLike) -> None:
        meanlog_array = _parameter(meanlog, name="meanlog")
        sdlog_array = _parameter(sdlog, name="sdlog", positive=True)
        self.meanlog, self.sdlog = _broadcast_parameters(meanlog_array, sdlog_array)

    @property
    def n_observations(self) -> int:
        return len(self.meanlog)

    def cdf(self, x: ArrayLike) -> NumericArray:
        values = _observations(x, length=self.n_observations, name="x")
        return np.asarray(
            stats.lognorm.cdf(values, self.sdlog, scale=np.exp(self.meanlog)),
            dtype=np.float64,
        )

    def quantile(self, q: ArrayLike) -> NumericArray:
        return _quantiles(
            q,
            length=self.n_observations,
            ppf=lambda probabilities: stats.lognorm.ppf(
                probabilities, self.sdlog, scale=np.exp(self.meanlog)
            ),
        )

    def sample(self, n: int, *, random_state: RandomState = None) -> NumericArray:
        size = _sample_size(n)
        return _rng(random_state).lognormal(
            self.meanlog,
            self.sdlog,
            size=(size, self.n_observations),
        )

    def log_prob(self, y: ArrayLike) -> NumericArray:
        values = _observations(y, length=self.n_observations, name="y")
        return np.asarray(
            stats.lognorm.logpdf(values, self.sdlog, scale=np.exp(self.meanlog)),
            dtype=np.float64,
        )

    def mean(self) -> NumericArray:
        return np.exp(self.meanlog + 0.5 * np.square(self.sdlog))

    def variance(self) -> NumericArray:
        variance_factor = np.expm1(np.square(self.sdlog))
        return variance_factor * np.exp(2 * self.meanlog + np.square(self.sdlog))

    def entropy(self) -> NumericArray:
        return np.asarray(
            stats.lognorm.entropy(self.sdlog, scale=np.exp(self.meanlog)),
            dtype=np.float64,
        )


class EmpiricalDistribution:
    """Per-observation predictive distributions represented by draws."""

    def __init__(self, samples: ArrayLike) -> None:
        array = np.asarray(samples, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
            raise InputValidationError(
                "samples must have shape (at least 2 draws, at least 1 observation)."
            )
        if not np.all(np.isfinite(array)):
            raise InputValidationError("samples must contain only finite values.")
        self.samples: NumericArray = np.asarray(array, dtype=np.float64)

    @property
    def n_observations(self) -> int:
        return int(self.samples.shape[1])

    def cdf(self, x: ArrayLike) -> NumericArray:
        values = _observations(x, length=self.n_observations, name="x")
        return np.asarray(
            np.mean(self.samples <= values, axis=0, dtype=np.float64),
            dtype=np.float64,
        )

    def quantile(self, q: ArrayLike) -> NumericArray:
        probabilities = np.asarray(q, dtype=np.float64)
        if probabilities.ndim > 1 or np.any(
            (probabilities <= 0) | (probabilities >= 1)
        ):
            raise InputValidationError(
                "q must be a scalar or vector strictly between 0 and 1."
            )
        return np.asarray(
            np.quantile(self.samples, probabilities, axis=0), dtype=np.float64
        )

    def sample(self, n: int, *, random_state: RandomState = None) -> NumericArray:
        size = _sample_size(n)
        indices = _rng(random_state).integers(0, len(self.samples), size=size)
        return self.samples[indices].copy()

    def log_prob(self, y: ArrayLike) -> NumericArray:
        del y
        raise NotImplementedError(
            "Log score is undefined for raw empirical draws without a density model."
        )

    def mean(self) -> NumericArray:
        return np.asarray(
            np.mean(self.samples, axis=0, dtype=np.float64), dtype=np.float64
        )

    def variance(self) -> NumericArray:
        return np.asarray(
            np.var(self.samples, axis=0, dtype=np.float64), dtype=np.float64
        )

    def entropy(self) -> NumericArray:
        raise NotImplementedError(
            "Entropy is undefined for raw empirical draws without discretization."
        )


class TweedieDistribution:
    """Compound Poisson-Gamma Tweedie distributions for ``1 < power < 2``.

    Sampling is exact under the compound representation. CDF and quantile are
    deterministic Monte Carlo approximations controlled by
    ``approximation_samples`` and ``approximation_seed``. Log density and
    entropy are intentionally unavailable without a validated series method.
    """

    def __init__(
        self,
        mean: ArrayLike,
        *,
        power: float,
        dispersion: ArrayLike,
        approximation_samples: int = 20_000,
        approximation_seed: int = 0,
    ) -> None:
        if not 1 < power < 2:
            raise InputValidationError(
                "Compound Poisson-Gamma Tweedie power must be between 1 and 2."
            )
        mean_array = _parameter(mean, name="mean", positive=True)
        dispersion_array = _parameter(dispersion, name="dispersion", positive=True)
        self._mean, self.dispersion = _broadcast_parameters(
            mean_array, dispersion_array
        )
        self.power = float(power)
        self.approximation_samples = _sample_size(approximation_samples)
        self.approximation_seed = approximation_seed

    @property
    def n_observations(self) -> int:
        return len(self._mean)

    def _approximation(self) -> NumericArray:
        return self.sample(
            self.approximation_samples,
            random_state=self.approximation_seed,
        )

    def cdf(self, x: ArrayLike) -> NumericArray:
        values = _observations(x, length=self.n_observations, name="x")
        return np.asarray(
            np.mean(
                self._approximation() <= values,
                axis=0,
                dtype=np.float64,
            ),
            dtype=np.float64,
        )

    def quantile(self, q: ArrayLike) -> NumericArray:
        probabilities = np.asarray(q, dtype=np.float64)
        if probabilities.ndim > 1 or np.any(
            (probabilities <= 0) | (probabilities >= 1)
        ):
            raise InputValidationError(
                "q must be a scalar or vector strictly between 0 and 1."
            )
        return np.asarray(
            np.quantile(self._approximation(), probabilities, axis=0),
            dtype=np.float64,
        )

    def sample(self, n: int, *, random_state: RandomState = None) -> NumericArray:
        size = _sample_size(n)
        generator = _rng(random_state)
        poisson_mean = np.power(self._mean, 2 - self.power) / (
            self.dispersion * (2 - self.power)
        )
        gamma_shape = (2 - self.power) / (self.power - 1)
        gamma_scale = (
            self.dispersion * (self.power - 1) * np.power(self._mean, self.power - 1)
        )
        counts = generator.poisson(poisson_mean, size=(size, self.n_observations))
        return np.asarray(
            generator.gamma(
                counts * gamma_shape,
                gamma_scale,
            ),
            dtype=np.float64,
        )

    def log_prob(self, y: ArrayLike) -> NumericArray:
        del y
        raise NotImplementedError(
            "Tweedie log density requires a validated numerical series implementation."
        )

    def mean(self) -> NumericArray:
        return self._mean.copy()

    def variance(self) -> NumericArray:
        return self.dispersion * np.power(self._mean, self.power)

    def entropy(self) -> NumericArray:
        raise NotImplementedError(
            "Tweedie entropy is not available from the compound sampling adapter."
        )
