import numpy as np
import pytest

import acteval as ae
from acteval.exceptions import InputValidationError


def test_asymmetric_pricing_regret_known_value() -> None:
    result = ae.pricing_regret(
        [10, 20],
        premium=[8, 24],
        benchmark_premium=[9, 22],
        underpricing_cost=2,
        overpricing_cost=1,
    )
    assert result.model_loss == pytest.approx(4)
    assert result.benchmark_loss == pytest.approx(2)
    assert result.regret == pytest.approx(2)
    assert result.relative_regret == pytest.approx(1)
    assert result.to_dict()["decision"] == "pricing"


def test_decision_regret_can_be_negative_and_weighted() -> None:
    result = ae.pricing_regret(
        [10, 20],
        premium=[10, 20],
        benchmark_premium=[8, 24],
        exposure=[1, 3],
    )
    assert result.model_loss == pytest.approx(0)
    assert result.regret < 0
    assert result.metadata["regret_can_be_negative"] is True
    with pytest.raises(TypeError):
        result.metadata["changed"] = True  # type: ignore[index]


def test_zero_benchmark_loss_omits_relative_regret() -> None:
    result = ae.pricing_regret(
        [10, 20],
        premium=[8, 20],
        benchmark_premium=[10, 20],
    )
    assert result.relative_regret is None


def test_premium_from_distribution_loadings() -> None:
    distribution = ae.PoissonDistribution([10, 20])
    premiums = ae.premium_from_distribution(
        distribution,
        profit_loading=0.1,
        expense_ratio=0.2,
    )
    assert premiums == pytest.approx([13.75, 27.5])


def test_loss_ratio_impact_known_weighted_value() -> None:
    result = ae.loss_ratio_impact(
        [80, 120],
        premium=[100, 100],
        target_loss_ratio=0.9,
        exposure=[1, 2],
    )
    assert result.loss_ratio == pytest.approx(320 / 300)
    assert result.signed_impact == pytest.approx(320 / 300 - 0.9)
    assert result.absolute_impact == pytest.approx(abs(result.signed_impact))
    assert result.to_dict()["target_loss_ratio"] == pytest.approx(0.9)


def test_reserve_and_capital_shortfall_known_values() -> None:
    reserve = ae.reserve_shortfall([10, 20, 30], [12, 15, 20])
    assert reserve.aggregate_shortfall == pytest.approx(15)
    assert reserve.mean_shortfall == pytest.approx(5)
    assert reserve.shortfall_frequency == pytest.approx(2 / 3)
    assert reserve.conditional_mean_shortfall == pytest.approx(7.5)
    assert reserve.to_dict()["decision"] == "reserve"
    capital = ae.capital_shortfall([10, 20], [20, 20])
    assert capital.aggregate_shortfall == pytest.approx(0)
    assert capital.conditional_mean_shortfall == pytest.approx(0)


def test_quantile_decision() -> None:
    distribution = ae.GammaDistribution([10, 20], shape=[5])
    decision = ae.quantile_decision(distribution, quantile=0.9)
    assert decision.shape == (2,)
    assert np.all(decision > distribution.mean())


def test_expected_shortfall_handles_discrete_boundary_mass() -> None:
    samples = np.asarray([[0.0]] * 990 + [[100.0]] * 10)
    distribution = ae.EmpiricalDistribution(samples)
    no_cover = ae.ReinsuranceOption("no cover", retention=100, premium=0)
    cover = ae.ReinsuranceOption("cover", retention=0, premium=5)

    selection = ae.select_reinsurance_option(
        distribution,
        [no_cover, cover],
        risk_measure="expected_shortfall",
        risk_quantile=0.95,
        capital_cost_rate=1,
        n_samples=100_000,
        random_state=7,
    )

    assert selection.selected == cover
    assert selection.projected_costs["no cover"] == pytest.approx(21, rel=0.08)
    assert selection.projected_costs["cover"] == pytest.approx(5)
    with pytest.raises(TypeError):
        selection.projected_costs["changed"] = 1.0  # type: ignore[index]


def test_reinsurance_selection_known_tradeoff() -> None:
    distribution = ae.EmpiricalDistribution(
        np.asarray([[0], [20], [40], [60], [80], [100]], dtype=float)
    )
    options = [
        ae.ReinsuranceOption("No cover", retention=1_000, premium=0),
        ae.ReinsuranceOption("Low retention", retention=30, premium=15),
        ae.ReinsuranceOption("High retention", retention=70, premium=4),
    ]
    selection = ae.select_reinsurance_option(
        distribution,
        options,
        risk_measure="expected_shortfall",
        risk_quantile=0.8,
        capital_cost_rate=0.5,
        n_samples=10_000,
        random_state=4,
    )
    assert selection.selected.name == min(
        selection.projected_costs, key=selection.projected_costs.__getitem__
    )
    assert selection.risk_measure == "expected_shortfall"
    assert selection.to_dict()["selected"]["name"] == selection.selected.name


def test_reinsurance_var_selection_and_realized_regret() -> None:
    distribution = ae.EmpiricalDistribution(
        np.asarray([[0], [10], [50], [100]], dtype=float)
    )
    no_cover = ae.ReinsuranceOption("No cover", retention=1_000, premium=0)
    cover = ae.ReinsuranceOption("Cover", retention=30, premium=10)
    selection = ae.select_reinsurance_option(
        distribution,
        [no_cover, cover],
        risk_measure="var",
        risk_quantile=0.75,
        capital_cost_rate=1,
        n_samples=5_000,
        random_state=2,
    )
    assert selection.selected in {no_cover, cover}
    regret = ae.reinsurance_decision_regret(
        [20, 100], selected=cover, benchmark=no_cover
    )
    assert regret.model_loss == pytest.approx((30 + 40) / 2)
    assert regret.benchmark_loss == pytest.approx((20 + 100) / 2)
    assert regret.regret < 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: ae.premium_from_distribution(
            ae.PoissonDistribution([1]), expense_ratio=1
        ),
        lambda: ae.loss_ratio_impact([1], [0]),
        lambda: ae.ReinsuranceOption("", retention=1, premium=1),
        lambda: ae.ReinsuranceOption("bad", retention=-1, premium=1),
        lambda: ae.select_reinsurance_option(
            ae.PoissonDistribution([1]), [], n_samples=10
        ),
        lambda: ae.select_reinsurance_option(
            ae.PoissonDistribution([1]),
            [ae.ReinsuranceOption("a", 1, 0)],
            risk_measure="other",  # type: ignore[arg-type]
        ),
    ],
)
def test_invalid_decision_inputs(call: object) -> None:
    with pytest.raises(InputValidationError):
        call()  # type: ignore[operator]


def test_generic_decision_rejects_invalid_loss_function() -> None:
    with pytest.raises(InputValidationError, match="loss_function"):
        ae.decision_regret(
            [1, 2],
            [1, 2],
            [1, 2],
            loss_function=lambda y, decision: np.asarray([-1, -1]),
            decision_name="invalid",
        )
