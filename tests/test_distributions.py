import numpy as np
import pytest

import acteval as ae
from acteval.exceptions import InputValidationError


def test_poisson_distribution_vectorized_contract() -> None:
    distribution = ae.PoissonDistribution([1, 3])
    assert distribution.n_observations == 2
    assert distribution.mean() == pytest.approx([1, 3])
    assert distribution.variance() == pytest.approx([1, 3])
    assert distribution.cdf([1, 3]).shape == (2,)
    assert distribution.quantile(0.5).shape == (2,)
    assert distribution.quantile([0.1, 0.9]).shape == (2, 2)
    assert distribution.log_prob([1, 3]).shape == (2,)
    assert distribution.entropy().shape == (2,)
    first = distribution.sample(20, random_state=42)
    second = distribution.sample(20, random_state=42)
    assert first.shape == (20, 2)
    assert np.array_equal(first, second)


def test_negative_binomial_mean_variance_parameterization() -> None:
    distribution = ae.NegativeBinomialDistribution([2, 4], dispersion=[2])
    assert distribution.mean() == pytest.approx([2, 4])
    assert distribution.variance() == pytest.approx([4, 12])
    assert distribution.sample(10, random_state=1).shape == (10, 2)
    assert np.all(np.isfinite(distribution.log_prob([1, 5])))
    assert distribution.cdf([1, 5]).shape == (2,)
    assert distribution.quantile([0.2, 0.8]).shape == (2, 2)
    assert np.all(np.isfinite(distribution.entropy()))


def test_gamma_distribution_moments() -> None:
    distribution = ae.GammaDistribution([10, 20], shape=[2])
    assert distribution.mean() == pytest.approx([10, 20])
    assert distribution.variance() == pytest.approx([50, 200])
    assert np.all(distribution.quantile(0.9) > distribution.mean())
    assert distribution.cdf([10, 20]).shape == (2,)
    assert distribution.sample(5, random_state=2).shape == (5, 2)
    assert np.all(np.isfinite(distribution.log_prob([10, 20])))
    assert np.all(np.isfinite(distribution.entropy()))


def test_lognormal_distribution_moments_and_shapes() -> None:
    distribution = ae.LognormalDistribution([0, 1], sdlog=[0.5])
    expected_mean = np.exp(np.asarray([0, 1]) + 0.125)
    assert distribution.mean() == pytest.approx(expected_mean)
    assert distribution.cdf(distribution.mean()).shape == (2,)
    assert distribution.sample(4, random_state=3).shape == (4, 2)
    assert distribution.quantile([0.2, 0.8]).shape == (2, 2)
    assert np.all(np.isfinite(distribution.log_prob([1, 3])))
    assert np.all(distribution.variance() > 0)
    assert np.all(np.isfinite(distribution.entropy()))


def test_empirical_distribution_contract() -> None:
    samples = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=float)
    distribution = ae.EmpiricalDistribution(samples)
    assert distribution.mean() == pytest.approx([1.5, 2.5])
    assert distribution.variance() == pytest.approx([1.25, 1.25])
    assert distribution.cdf([1, 3]) == pytest.approx([0.5, 0.75])
    assert distribution.quantile([0.25, 0.75]).shape == (2, 2)
    assert distribution.sample(5, random_state=2).shape == (5, 2)
    assert distribution.log_prob([1, 2]) == pytest.approx(np.log([0.25, 0.25]))
    assert distribution.entropy() == pytest.approx(np.log([4, 4]))


def test_tweedie_compound_sampling_moments_and_approximations() -> None:
    distribution = ae.TweedieDistribution(
        [2, 5],
        power=1.5,
        dispersion=[0.5],
        approximation_samples=2_000,
        approximation_seed=7,
    )
    samples = distribution.sample(20_000, random_state=11)
    assert np.mean(samples, axis=0) == pytest.approx([2, 5], rel=0.04)
    assert distribution.variance() == pytest.approx(0.5 * np.power([2, 5], 1.5))
    assert distribution.cdf([2, 5]).shape == (2,)
    assert distribution.quantile([0.1, 0.9]).shape == (2, 2)
    assert np.all(np.isfinite(distribution.log_prob([2, 5])))
    assert np.all(np.isfinite(distribution.entropy()))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ae.PoissonDistribution([-1]),
        lambda: ae.NegativeBinomialDistribution([1], dispersion=[0]),
        lambda: ae.GammaDistribution([0], shape=[1]),
        lambda: ae.LognormalDistribution([0], sdlog=[0]),
        lambda: ae.EmpiricalDistribution([[1, 2]]),
        lambda: ae.TweedieDistribution([1], power=2, dispersion=[1]),
    ],
)
def test_distribution_invalid_parameters(factory: object) -> None:
    with pytest.raises(InputValidationError):
        factory()  # type: ignore[operator]


def test_distribution_rejects_invalid_shapes_and_quantiles() -> None:
    with pytest.raises(InputValidationError, match="equal lengths"):
        ae.GammaDistribution([1, 2], shape=[1, 2, 3])
    distribution = ae.PoissonDistribution([1, 2])
    with pytest.raises(InputValidationError, match="length 2"):
        distribution.cdf([1, 2, 3])
    with pytest.raises(InputValidationError, match="between 0 and 1"):
        distribution.quantile(1)
    with pytest.raises(InputValidationError, match="positive integer"):
        distribution.sample(0)
    empirical = ae.EmpiricalDistribution([[1, 2], [2, 3]])
    with pytest.raises(InputValidationError, match="between 0 and 1"):
        empirical.quantile(0)
