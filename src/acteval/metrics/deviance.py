"""Actuarial deviance metrics backed by scikit-learn."""

from numpy.typing import ArrayLike
from sklearn.metrics import (
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_tweedie_deviance,
)

from acteval.exceptions import InputValidationError
from acteval.registry import register_metric
from acteval.validation import combine_weights, validate_inputs
from acteval.validation.inputs import Domain


@register_metric(
    name="poisson_deviance",
    tasks=("claim_frequency",),
    category="accuracy",
    higher_is_better=False,
    description="Mean Poisson deviance for nonnegative outcomes.",
    reference="https://scikit-learn.org/stable/modules/model_evaluation.html#mean-poisson-gamma-and-tweedie-deviances",
)
def poisson_deviance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute mean Poisson deviance.

    Observations must be nonnegative and predictions strictly positive. Lower
    values indicate better fit under a Poisson deviance objective; zero means
    perfect point predictions. This diagnostic does not prove that the data
    follow a Poisson distribution.
    """
    inputs = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
        y_pred_domain="positive",
    )
    return float(
        mean_poisson_deviance(
            inputs.y_true,
            inputs.y_pred,
            sample_weight=combine_weights(inputs),
        )
    )


@register_metric(
    name="gamma_deviance",
    tasks=("claim_severity",),
    category="accuracy",
    higher_is_better=False,
    description="Mean Gamma deviance for strictly positive outcomes.",
    reference="https://scikit-learn.org/stable/modules/model_evaluation.html#mean-poisson-gamma-and-tweedie-deviances",
)
def gamma_deviance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute mean Gamma deviance.

    Observations and predictions must both be strictly positive. Lower values
    indicate better fit under a Gamma deviance objective; zero means perfect
    point predictions. It is not appropriate for zero-valued severities.
    """
    inputs = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="positive",
        y_pred_domain="positive",
    )
    return float(
        mean_gamma_deviance(
            inputs.y_true,
            inputs.y_pred,
            sample_weight=combine_weights(inputs),
        )
    )


@register_metric(
    name="tweedie_deviance",
    tasks=("pure_premium",),
    category="accuracy",
    higher_is_better=False,
    description="Mean Tweedie deviance with an explicit variance power.",
    reference="https://scikit-learn.org/stable/modules/model_evaluation.html#mean-poisson-gamma-and-tweedie-deviances",
)
def tweedie_deviance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    power: float,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> float:
    """Compute mean Tweedie deviance for an explicit variance power.

    Powers in ``(0, 1)`` are undefined. For compound Poisson-Gamma pure
    premium models, ``1 < power < 2`` is typical and permits zero outcomes.
    Predictions must be strictly positive for every nonzero power. ActEval
    never estimates or silently infers the power from observations.
    """
    if not float("-inf") < power < float("inf"):
        raise InputValidationError("power must be finite.")
    if 0 < power < 1:
        raise InputValidationError(
            "Tweedie power values between 0 and 1 are undefined."
        )
    if power == 0:
        true_domain: Domain = "real"
        predicted_domain: Domain = "real"
    elif power < 2:
        true_domain = "nonnegative"
        predicted_domain = "positive"
    else:
        true_domain = "positive"
        predicted_domain = "positive"
    inputs = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain=true_domain,
        y_pred_domain=predicted_domain,
    )
    return float(
        mean_tweedie_deviance(
            inputs.y_true,
            inputs.y_pred,
            sample_weight=combine_weights(inputs),
            power=power,
        )
    )
