"""Evaluate pricing, capital, and reinsurance financial consequences."""

import numpy as np

import acteval as ae

rng = np.random.default_rng(7)
mean_loss = np.linspace(100, 1_000, 60)
distribution = ae.GammaDistribution(mean_loss, shape=[4])
realized_loss = rng.gamma(4, mean_loss / 4)

model_premium = ae.premium_from_distribution(
    distribution,
    profit_loading=0.08,
    expense_ratio=0.20,
)
benchmark_premium = mean_loss * 1.25

pricing = ae.pricing_regret(
    realized_loss,
    model_premium,
    benchmark_premium,
    underpricing_cost=2,
    overpricing_cost=1,
    benchmark_name="current tariff",
)
print("Pricing:", pricing.to_dict())
print(
    "Loss ratio:",
    ae.loss_ratio_impact(realized_loss, model_premium, target_loss_ratio=0.7),
)

capital = ae.quantile_decision(distribution, quantile=0.995)
print("Capital shortfall:", ae.capital_shortfall(realized_loss, capital))

aggregate_distribution = ae.EmpiricalDistribution(
    distribution.sample(5_000, random_state=9)
)
options = [
    ae.ReinsuranceOption("No cover", retention=1_000_000, premium=0),
    ae.ReinsuranceOption("10k retention", retention=10_000, premium=8_000),
    ae.ReinsuranceOption("20k retention", retention=20_000, premium=4_000),
]
selection = ae.select_reinsurance_option(
    aggregate_distribution,
    options,
    risk_measure="expected_shortfall",
    risk_quantile=0.995,
    capital_cost_rate=0.10,
    random_state=11,
)
print("Reinsurance selection:", selection.to_dict())
