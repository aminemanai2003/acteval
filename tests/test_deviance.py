import math

import pytest

from acteval import gamma_deviance, poisson_deviance
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
