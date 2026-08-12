import numpy as np
import pytest

import acteval as ae
from acteval.exceptions import InputValidationError


def test_crps_is_zero_for_perfect_degenerate_empirical_distribution() -> None:
    observed = np.asarray([1, 2, 3], dtype=float)
    distribution = ae.EmpiricalDistribution(np.tile(observed, (10, 1)))
    assert ae.crps(observed, distribution, n_samples=100) == pytest.approx(0)


def test_crps_is_reproducible_and_rewards_better_distribution() -> None:
    observed = np.asarray([1, 2, 4, 6], dtype=float)
    good = ae.PoissonDistribution([1, 2, 4, 6])
    bad = ae.PoissonDistribution([8, 8, 8, 8])
    good_score = ae.crps(observed, good, n_samples=5_000, random_state=4)
    repeated = ae.crps(observed, good, n_samples=5_000, random_state=4)
    bad_score = ae.crps(observed, bad, n_samples=5_000, random_state=4)
    assert repeated == pytest.approx(good_score)
    assert good_score < bad_score


def test_log_and_brier_scores_known_behavior() -> None:
    observed = [0, 1, 2]
    good = ae.PoissonDistribution([0.2, 1, 2])
    bad = ae.PoissonDistribution([5, 5, 5])
    assert ae.log_score(observed, good) < ae.log_score(observed, bad)
    score = ae.brier_score(observed, good, threshold=1)
    assert 0 <= score <= 1


def test_quantile_and_interval_scores_known_values() -> None:
    assert ae.quantile_score([0, 2], [1, 1], quantile=0.5) == pytest.approx(0.5)
    assert ae.interval_score([1, 2], [0, 1], [2, 3], coverage=0.9) == pytest.approx(2)


def test_interval_diagnostics() -> None:
    observed = [1, 2, 4]
    lower = [0, 1, 3]
    upper = [2, 3, 5]
    assert ae.prediction_interval_coverage(observed, lower, upper) == pytest.approx(1)
    assert ae.mean_interval_width(lower, upper) == pytest.approx(2)
    distribution = ae.GammaDistribution([1, 2, 4], shape=[10])
    central_lower, central_upper = ae.central_prediction_interval(
        distribution, coverage=0.8
    )
    assert central_lower.shape == (3,)
    assert np.all(central_lower < central_upper)
    assert (
        0
        <= ae.distribution_interval_coverage(observed, distribution, coverage=0.8)
        <= 1
    )
    assert ae.distribution_interval_width(observed, distribution, coverage=0.8) > 0
    with pytest.raises(InputValidationError, match="upper"):
        ae.prediction_interval_coverage(observed, lower, [1, 2])
    with pytest.raises(InputValidationError, match="lower"):
        ae.mean_interval_width([2, 3], [1, 2])


def test_variance_and_entropy_are_uncertainty_diagnostics() -> None:
    distribution = ae.PoissonDistribution([1, 2, 3])
    assert ae.predictive_variance([1, 2, 3], distribution) == pytest.approx(2)
    assert np.isfinite(ae.predictive_entropy([1, 2, 3], distribution))


def test_evaluate_distribution_defaults_and_comparison() -> None:
    observed = [1, 2, 3, 4]
    good = ae.PoissonDistribution([1, 2, 3, 4])
    bad = ae.PoissonDistribution([7, 7, 7, 7])
    result = ae.evaluate_distribution(observed, good, task="claim_frequency")
    assert set(result.metrics) == {
        "crps[n_samples=2000,random_state=0]",
        "predictive_variance",
        "coverage_90",
        "interval_width_90",
    }
    assert result.metadata["entropy_is_not_universal_quality"] is True
    comparison = ae.compare_distributions(
        observed,
        {"Good": good, "Bad": bad},
        task="claim_frequency",
        metrics=[ae.MetricSpec("crps", {"n_samples": 5_000, "random_state": 5})],
    )
    metric = "crps[n_samples=5000,random_state=5]"
    assert comparison.rank(metric).iloc[0]["model"] == "Good"


def test_distribution_evaluation_with_explicit_scores() -> None:
    observed = [1, 2, 3]
    distribution = ae.PoissonDistribution([1, 2, 3])
    result = ae.evaluate_distribution(
        observed,
        distribution,
        task="claim_frequency",
        metrics=[
            "log_score",
            ae.MetricSpec("brier_score", {"threshold": 2}),
            ae.MetricSpec("quantile_score", {"quantile": 0.75}),
            ae.MetricSpec("interval_score", {"coverage": 0.8}),
            "predictive_entropy",
        ],
    )
    assert len(result.metrics) == 5


def test_distribution_evaluation_rejects_point_and_length_mismatch() -> None:
    distribution = ae.PoissonDistribution([1, 2])
    with pytest.raises(InputValidationError, match="length"):
        ae.evaluate_distribution([1], distribution, task="claim_frequency")
    with pytest.raises(InputValidationError, match="not a predictive-distribution"):
        ae.evaluate_distribution(
            [1, 2], distribution, task="claim_frequency", metrics=["rmse"]
        )
    with pytest.raises(InputValidationError, match="at least one model"):
        ae.compare_distributions([1, 2], {}, task="claim_frequency")


def test_invalid_probabilistic_inputs() -> None:
    distribution = ae.PoissonDistribution([1, 2])
    with pytest.raises(InputValidationError, match="at least 2"):
        ae.crps([1, 2], distribution, n_samples=1)
    with pytest.raises(InputValidationError, match="finite"):
        ae.brier_score([1, 2], distribution, threshold=float("nan"))
    with pytest.raises(InputValidationError, match="lower"):
        ae.interval_score([1, 2], [2, 2], [1, 1], coverage=0.9)
    with pytest.raises(InputValidationError, match="expected 1"):
        ae.crps([1], distribution, n_samples=10)
    empirical = ae.EmpiricalDistribution([[1, 2], [2, 3]])
    assert ae.predictive_entropy([1, 2], empirical) == pytest.approx(np.log(2))
