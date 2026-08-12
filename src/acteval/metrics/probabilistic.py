"""Proper scoring rules for predictive distributions and quantiles."""

import numpy as np
from numpy.typing import ArrayLike

from acteval.exceptions import InputValidationError
from acteval.registry import register_metric
from acteval.types import NumericArray, PredictiveDistribution, Task
from acteval.utils import effective_weights
from acteval.validation import (
    combine_weights,
    validate_inputs,
    validate_probability,
)

_ALL_TASKS: tuple[Task, ...] = (
    "claim_frequency",
    "claim_severity",
    "pure_premium",
)


def _distribution_inputs(
    y_true: ArrayLike,
    distribution: PredictiveDistribution,
    *,
    sample_weight: ArrayLike | None,
    exposure: ArrayLike | None,
) -> tuple[NumericArray, NumericArray]:
    inputs = validate_inputs(
        y_true,
        y_true,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
    )
    if distribution.n_observations != len(inputs.y_true):
        raise InputValidationError(
            "Predictive distribution has "
            f"{distribution.n_observations} observations; expected "
            f"{len(inputs.y_true)}."
        )
    weights = effective_weights(len(inputs.y_true), combine_weights(inputs))
    return inputs.y_true, weights


def _distribution_values(
    values: ArrayLike,
    *,
    length: int,
    name: str,
    allow_infinite: bool = False,
) -> NumericArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,):
        raise InputValidationError(f"{name} must have shape ({length},).")
    invalid_infinite = not allow_infinite and not np.all(np.isfinite(array))
    if np.any(np.isnan(array)) or invalid_infinite:
        raise InputValidationError(f"{name} contains invalid numerical values.")
    return array


@register_metric(
    name="crps",
    tasks=_ALL_TASKS,
    category="probabilistic",
    higher_is_better=False,
    requires_distribution=True,
    description="Sample-approximated continuous ranked probability score.",
)
def crps(
    y_true: ArrayLike,
    distribution: PredictiveDistribution,
    *,
    n_samples: int = 2_000,
    random_state: int | np.random.Generator | None = 0,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute CRPS from predictive samples.

    For empirical draws ``x_1, ..., x_m``, the per-observation score is
    ``mean(|x_j-y|) - 0.5 * mean(|x_j-x_k|)``. The pairwise term is calculated
    exactly from sorted draws in ``O(m log m)`` rather than materializing an
    ``m x m`` matrix. Randomness is explicit and reproducible by default.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
        raise InputValidationError("n_samples must be an integer of at least 2.")
    observed, weights = _distribution_inputs(
        y_true,
        distribution,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    draws = np.asarray(
        distribution.sample(n_samples, random_state=random_state),
        dtype=np.float64,
    )
    if draws.shape != (n_samples, len(observed)) or not np.all(np.isfinite(draws)):
        raise InputValidationError(
            "Distribution samples must be finite with shape "
            f"({n_samples}, {len(observed)})."
        )
    first_term = np.mean(np.abs(draws - observed), axis=0)
    ordered = np.sort(draws, axis=0)
    coefficients = 2 * np.arange(1, n_samples + 1) - n_samples - 1
    half_pairwise_term = np.sum(coefficients[:, None] * ordered, axis=0) / (
        n_samples**2
    )
    scores = first_term - half_pairwise_term
    return float(np.average(scores, weights=weights))


@register_metric(
    name="log_score",
    tasks=_ALL_TASKS,
    category="probabilistic",
    higher_is_better=False,
    requires_distribution=True,
    description="Negative predictive log probability or log density.",
)
def log_score(
    y_true: ArrayLike,
    distribution: PredictiveDistribution,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute weighted mean negative log predictive probability/density.

    Discrete adapters use probability mass and continuous adapters use density.
    Scores across those two measure classes are not directly comparable.
    Impossible observations legitimately produce an infinite score.
    """
    observed, weights = _distribution_inputs(
        y_true,
        distribution,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    log_probabilities = _distribution_values(
        distribution.log_prob(observed),
        length=len(observed),
        name="distribution.log_prob",
        allow_infinite=True,
    )
    if np.any(log_probabilities == np.inf):
        raise InputValidationError("Log probabilities cannot be positive infinity.")
    return float(np.average(-log_probabilities, weights=weights))


@register_metric(
    name="brier_score",
    tasks=_ALL_TASKS,
    category="probabilistic",
    higher_is_better=False,
    requires_distribution=True,
    description="Brier score for the event that the outcome is at most a threshold.",
)
def brier_score(
    y_true: ArrayLike,
    distribution: PredictiveDistribution,
    *,
    threshold: float,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute Brier score for ``P(Y <= threshold)``.

    The event direction and threshold are explicit. This avoids treating Brier
    score as a generic regression metric.
    """
    if not np.isfinite(threshold):
        raise InputValidationError("threshold must be finite.")
    observed, weights = _distribution_inputs(
        y_true,
        distribution,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    probabilities = _distribution_values(
        distribution.cdf(float(threshold)),
        length=len(observed),
        name="distribution.cdf",
    )
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise InputValidationError("Distribution CDF values must lie in [0, 1].")
    events = observed <= threshold
    return float(np.average(np.square(probabilities - events), weights=weights))


def quantile_score(
    y_true: ArrayLike,
    y_quantile: ArrayLike,
    *,
    quantile: float,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute the pinball score for explicit predictive quantiles.

    The score is ``q * (y-p)`` for underprediction and
    ``(1-q) * (p-y)`` for overprediction. Lower is better.
    """
    probability = validate_probability(quantile, name="quantile")
    inputs = validate_inputs(
        y_true,
        y_quantile,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    residual = inputs.y_true - inputs.y_pred
    losses = np.maximum(probability * residual, (probability - 1) * residual)
    return float(np.average(losses, weights=combine_weights(inputs)))


@register_metric(
    name="quantile_score",
    tasks=_ALL_TASKS,
    category="probabilistic",
    higher_is_better=False,
    requires_distribution=True,
    description="Pinball score at an explicit predictive-distribution quantile.",
)
def distribution_quantile_score(
    y_true: ArrayLike,
    distribution: PredictiveDistribution,
    *,
    quantile: float,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute quantile score after extracting a distribution quantile."""
    prediction = distribution.quantile(quantile)
    return quantile_score(
        y_true,
        prediction,
        quantile=quantile,
        sample_weight=sample_weight,
        exposure=exposure,
    )


def interval_score(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    coverage: float = 0.9,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute the proper central prediction-interval score.

    Width is penalized, with additional penalties of ``2 / alpha`` for misses,
    where ``alpha = 1 - coverage``. Lower is better.
    """
    central_coverage = validate_probability(coverage, name="coverage")
    lower_inputs = validate_inputs(
        y_true,
        lower,
        sample_weight=sample_weight,
        exposure=exposure,
    )
    upper_values = _distribution_values(
        upper,
        length=len(lower_inputs.y_true),
        name="upper",
    )
    if np.any(lower_inputs.y_pred > upper_values):
        raise InputValidationError("lower must not exceed upper.")
    alpha = 1 - central_coverage
    scores = upper_values - lower_inputs.y_pred
    scores += (2 / alpha) * np.maximum(lower_inputs.y_pred - lower_inputs.y_true, 0)
    scores += (2 / alpha) * np.maximum(lower_inputs.y_true - upper_values, 0)
    return float(np.average(scores, weights=combine_weights(lower_inputs)))


@register_metric(
    name="interval_score",
    tasks=_ALL_TASKS,
    category="probabilistic",
    higher_is_better=False,
    requires_distribution=True,
    description="Proper score for a central predictive interval.",
)
def distribution_interval_score(
    y_true: ArrayLike,
    distribution: PredictiveDistribution,
    *,
    coverage: float = 0.9,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute interval score from a distribution's central interval."""
    central_coverage = validate_probability(coverage, name="coverage")
    alpha = 1 - central_coverage
    lower = distribution.quantile(alpha / 2)
    upper = distribution.quantile(1 - alpha / 2)
    return interval_score(
        y_true,
        lower,
        upper,
        coverage=central_coverage,
        sample_weight=sample_weight,
        exposure=exposure,
    )
