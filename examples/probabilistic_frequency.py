"""Compare Poisson and Negative Binomial predictive claim distributions."""

import numpy as np

import acteval as ae

rng = np.random.default_rng(42)
mean = np.linspace(0.2, 4.0, 80)
observed = rng.negative_binomial(2.5, 2.5 / (2.5 + mean))

poisson = ae.PoissonDistribution(mean)
negative_binomial = ae.NegativeBinomialDistribution(mean, dispersion=[2.5])

comparison = ae.compare_distributions(
    observed,
    {"Poisson": poisson, "Negative Binomial": negative_binomial},
    task="claim_frequency",
    metrics=[
        ae.MetricSpec("crps", {"n_samples": 2_000, "random_state": 42}),
        "log_score",
        ae.MetricSpec("brier_score", {"threshold": 0}),
        ae.MetricSpec("interval_score", {"coverage": 0.9}),
        "predictive_variance",
        "predictive_entropy",
    ],
)

print(comparison.to_dataframe())
print("\nCRPS ranking")
print(comparison.rank("crps[n_samples=2000,random_state=42]"))
