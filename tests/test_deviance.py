import math

import pytest

from acteval import gamma_deviance, poisson_deviance, tweedie_deviance
from acteval.exceptions import InputValidationError


def test_poisson_deviance_known_value() -> None:
    assert poisson_deviance([1, 2], [1, 1]) == pytest.approx(2 * math.log(2) - 1)


def test_poisson_deviance_weighted() -> None:
    value = poisson_deviance([1, 2], [1, 1], sample_weight=[3, 1])
    assert value == pytest.approx((2 * math.log(2) - 1) / 2)


def test_poisson_deviance_rejects_zero_prediction() -> None:
    with pytest.raises(InputValidationError, match="strictly positive"):
        poisson_deviance([0, 1], [0, 1])


def test_gamma_deviance_known_value() -> None:
    assert gamma_deviance([1, 2], [1, 1]) == pytest.approx(1 - math.log(2))


def test_gamma_deviance_weighted() -> None:
    value = gamma_deviance([1, 2], [1, 1], sample_weight=[3, 1])
    assert value == pytest.approx((1 - math.log(2)) / 2)


def test_gamma_deviance_rejects_zero_observation() -> None:
    with pytest.raises(InputValidationError, match="strictly positive"):
        gamma_deviance([0, 1], [1, 1])


def test_tweedie_deviance_perfect_prediction() -> None:
    assert tweedie_deviance([0, 1, 3], [0.1, 1, 3], power=1.5) > 0
    assert tweedie_deviance([1, 2, 3], [1, 2, 3], power=1.5) == pytest.approx(0)


def test_tweedie_deviance_supports_exposure_weighting() -> None:
    weighted = tweedie_deviance([0, 1, 4], [0.5, 1.5, 3], power=1.5, exposure=[1, 2, 5])
    repeated = tweedie_deviance(
        [0, 1, 1, 4, 4, 4, 4, 4],
        [0.5, 1.5, 1.5, 3, 3, 3, 3, 3],
        power=1.5,
    )
    assert weighted == pytest.approx(repeated)


@pytest.mark.parametrize("power", [0.1, 0.5, 0.99])
def test_tweedie_deviance_rejects_undefined_power(power: float) -> None:
    with pytest.raises(InputValidationError, match="undefined"):
        tweedie_deviance([1, 2], [1, 2], power=power)
