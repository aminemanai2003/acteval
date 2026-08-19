"""High-level model-agnostic evaluation and comparison APIs."""

import re
from collections.abc import Mapping, Sequence
from functools import cache
from inspect import Signature, signature
from typing import Any

from numpy.typing import ArrayLike

from acteval.exceptions import InputValidationError
from acteval.registry import get_metric
from acteval.reports import ComparisonResult, EvaluationResult
from acteval.tasks import DEFAULT_METRICS
from acteval.types import MetricSpec, PredictiveDistribution, Task
from acteval.validation import validate_inputs, validate_task

MetricSelection = str | MetricSpec
_TAIL_ALIAS = re.compile(r"^(tail_mae|tail_rmse|tail_ae)_(\d+(?:\.\d+)?)$")
_DEFAULT_DISTRIBUTION_METRICS = (
    MetricSpec("crps", {"n_samples": 2_000, "random_state": 0}),
    MetricSpec("predictive_variance"),
    MetricSpec("interval_coverage", {"coverage": 0.9}, label="coverage_90"),
    MetricSpec("interval_width", {"coverage": 0.9}, label="interval_width_90"),
)


@cache
def _metric_signature(function: Any) -> Signature:
    """Cache call signatures used repeatedly by bootstrap and segment runs."""

    return signature(function)


def _parse_metric(selection: MetricSelection) -> MetricSpec:
    if isinstance(selection, MetricSpec):
        return selection
    if not isinstance(selection, str):
        raise InputValidationError(
            "Each metric selection must be a registry name or MetricSpec."
        )
    normalized = selection.strip().lower()
    match = _TAIL_ALIAS.fullmatch(normalized)
    if match:
        metric_name, percentile_text = match.groups()
        percentile = float(percentile_text)
        if not 0 < percentile < 100:
            raise InputValidationError(
                f"Tail metric percentile must be between 0 and 100: {selection!r}."
            )
        canonical = "tail_ae_ratio" if metric_name == "tail_ae" else metric_name
        return MetricSpec(
            canonical,
            {"quantile": percentile / 100.0},
            label=normalized,
        )
    return MetricSpec(normalized)


def _metric_label(specification: MetricSpec) -> str:
    if specification.label is not None:
        return specification.label
    if not specification.parameters:
        return specification.name
    parameter_text = ",".join(
        f"{name}={value:g}" for name, value in sorted(specification.parameters.items())
    )
    return f"{specification.name}[{parameter_text}]"


def _evaluate_metric(
    specification: MetricSpec,
    *,
    task: Task,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None,
    exposure: ArrayLike | None,
) -> tuple[str, float, dict[str, Any]]:
    definition = get_metric(specification.name)
    if task not in definition.supported_tasks:
        raise InputValidationError(
            f"Metric {specification.name!r} does not support task {task!r}."
        )
    keyword_arguments = {
        "sample_weight": sample_weight,
        "exposure": exposure,
        **specification.parameters,
    }
    try:
        _metric_signature(definition.function).bind(
            y_true,
            y_pred,
            **keyword_arguments,
        )
    except TypeError as error:
        raise InputValidationError(
            f"Invalid parameters for metric {specification.name!r}: "
            f"{dict(specification.parameters)!r}."
        ) from error
    value = definition(y_true, y_pred, **keyword_arguments)
    label = _metric_label(specification)
    metadata = {
        "name": definition.name,
        "category": definition.category,
        "higher_is_better": definition.higher_is_better,
        "target": definition.target,
        "prediction_functional": definition.prediction_functional,
        "parameters": dict(specification.parameters),
        "reference": definition.reference,
    }
    return label, value, metadata


def evaluate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    task: str,
    exposure: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    metrics: Sequence[MetricSelection] | None = None,
) -> EvaluationResult:
    """Evaluate one prediction array for an actuarial task.

    ``y_true`` and ``y_pred`` must be nonnegative values on the same scale.
    For frequency and pure premium rates, exposure is portfolio volume. When
    sample weights are also supplied, effective weight is their elementwise
    product. Defaults are task-specific and all parameter values are retained
    in result metadata.
    """
    resolved_task = validate_task(task)
    validated = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
        y_pred_domain="nonnegative",
    )
    selections = (
        tuple(DEFAULT_METRICS[resolved_task])
        if metrics is None
        else tuple(_parse_metric(metric) for metric in metrics)
    )
    if not selections:
        raise InputValidationError("metrics must contain at least one metric.")
    prediction_functionals = {
        definition.prediction_functional
        for specification in selections
        if (definition := get_metric(specification.name)).prediction_functional
        is not None
    }
    if len(prediction_functionals) > 1:
        names = ", ".join(sorted(prediction_functionals))
        raise InputValidationError(
            "Selected metrics evaluate incompatible prediction functionals: "
            f"{names}. Evaluate mean and median predictions separately."
        )
    values: dict[str, float] = {}
    specifications: dict[str, dict[str, Any]] = {}
    for specification in selections:
        label, value, metadata = _evaluate_metric(
            specification,
            task=resolved_task,
            y_true=validated.y_true,
            y_pred=validated.y_pred,
            sample_weight=validated.sample_weight,
            exposure=validated.exposure,
        )
        if label in values:
            raise InputValidationError(f"Duplicate metric output label: {label!r}.")
        values[label] = value
        specifications[label] = metadata
    if exposure is not None and sample_weight is not None:
        weighting = "sample_weight * exposure"
    elif exposure is not None:
        weighting = "exposure"
    elif sample_weight is not None:
        weighting = "sample_weight"
    else:
        weighting = "uniform"
    result_metadata = {
        "n_observations": len(validated.y_true),
        "has_exposure": exposure is not None,
        "has_sample_weight": sample_weight is not None,
        "weighting": weighting,
        "target_scale": "same_scale_nonnegative",
        "prediction_functional": next(
            iter(prediction_functionals), "ranking_or_diagnostic"
        ),
        "metric_specs": specifications,
    }
    return EvaluationResult(resolved_task, values, result_metadata)


def compare(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    *,
    task: str,
    exposure: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    metrics: Sequence[MetricSelection] | None = None,
) -> ComparisonResult:
    """Evaluate several prediction arrays against the same observations."""
    resolved_task = validate_task(task)
    if not predictions:
        raise InputValidationError("predictions must contain at least one model.")
    results: dict[str, EvaluationResult] = {}
    for raw_name, prediction in predictions.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise InputValidationError("Every prediction model name must be non-empty.")
        model_name = raw_name.strip()
        if model_name in results:
            raise InputValidationError(
                f"Duplicate model name after trimming: {model_name!r}."
            )
        results[model_name] = evaluate(
            y_true,
            prediction,
            task=resolved_task,
            exposure=exposure,
            sample_weight=sample_weight,
            metrics=metrics,
        )
    return ComparisonResult(
        resolved_task,
        results,
        {
            "models": tuple(results),
            "n_models": len(results),
            "no_universal_best_model": True,
        },
    )


def _evaluate_distribution_metric(
    specification: MetricSpec,
    *,
    task: Task,
    y_true: ArrayLike,
    distribution: PredictiveDistribution,
    sample_weight: ArrayLike | None,
    exposure: ArrayLike | None,
) -> tuple[str, float, dict[str, Any]]:
    definition = get_metric(specification.name)
    if not definition.requires_distribution:
        raise InputValidationError(
            f"Metric {specification.name!r} is not a predictive-distribution metric."
        )
    if task not in definition.supported_tasks:
        raise InputValidationError(
            f"Metric {specification.name!r} does not support task {task!r}."
        )
    keyword_arguments = {
        "sample_weight": sample_weight,
        "exposure": exposure,
        **specification.parameters,
    }
    try:
        _metric_signature(definition.function).bind(
            y_true,
            distribution,
            **keyword_arguments,
        )
    except TypeError as error:
        raise InputValidationError(
            f"Invalid parameters for metric {specification.name!r}: "
            f"{dict(specification.parameters)!r}."
        ) from error
    value = definition(y_true, distribution, **keyword_arguments)
    label = _metric_label(specification)
    return (
        label,
        value,
        {
            "name": definition.name,
            "category": definition.category,
            "higher_is_better": definition.higher_is_better,
            "target": definition.target,
            "prediction_functional": definition.prediction_functional,
            "parameters": dict(specification.parameters),
            "reference": definition.reference,
            "requires_distribution": True,
        },
    )


def evaluate_distribution(
    y_true: ArrayLike,
    distribution: PredictiveDistribution,
    *,
    task: str,
    exposure: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    metrics: Sequence[MetricSelection] | None = None,
) -> EvaluationResult:
    """Evaluate one predictive distribution per observation.

    The default report contains sample-approximated CRPS, predictive variance,
    and 90% central-interval coverage and width. Log score and entropy remain
    opt-in because their interpretation depends on the distribution's base
    measure and they do not share a universal optimization objective.
    """
    resolved_task = validate_task(task)
    validated = validate_inputs(
        y_true,
        y_true,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
    )
    if distribution.n_observations != len(validated.y_true):
        raise InputValidationError(
            "Predictive distribution length does not match y_true."
        )
    selections = (
        _DEFAULT_DISTRIBUTION_METRICS
        if metrics is None
        else tuple(_parse_metric(metric) for metric in metrics)
    )
    if not selections:
        raise InputValidationError("metrics must contain at least one metric.")
    values: dict[str, float] = {}
    specifications: dict[str, dict[str, Any]] = {}
    for specification in selections:
        label, value, metadata = _evaluate_distribution_metric(
            specification,
            task=resolved_task,
            y_true=validated.y_true,
            distribution=distribution,
            sample_weight=validated.sample_weight,
            exposure=validated.exposure,
        )
        if label in values:
            raise InputValidationError(f"Duplicate metric output label: {label!r}.")
        values[label] = value
        specifications[label] = metadata
    return EvaluationResult(
        resolved_task,
        values,
        {
            "n_observations": len(validated.y_true),
            "distribution_type": type(distribution).__name__,
            "has_exposure": exposure is not None,
            "has_sample_weight": sample_weight is not None,
            "metric_specs": specifications,
            "entropy_is_not_universal_quality": True,
        },
    )


def compare_distributions(
    y_true: ArrayLike,
    distributions: Mapping[str, PredictiveDistribution],
    *,
    task: str,
    exposure: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    metrics: Sequence[MetricSelection] | None = None,
) -> ComparisonResult:
    """Compare predictive distributions under identical scoring rules."""
    resolved_task = validate_task(task)
    if not distributions:
        raise InputValidationError("distributions must contain at least one model.")
    results: dict[str, EvaluationResult] = {}
    for raw_name, distribution in distributions.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise InputValidationError(
                "Every distribution model name must be non-empty."
            )
        model_name = raw_name.strip()
        if model_name in results:
            raise InputValidationError(f"Duplicate model name: {model_name!r}.")
        results[model_name] = evaluate_distribution(
            y_true,
            distribution,
            task=resolved_task,
            exposure=exposure,
            sample_weight=sample_weight,
            metrics=metrics,
        )
    return ComparisonResult(
        resolved_task,
        results,
        {
            "models": tuple(results),
            "n_models": len(results),
            "distribution_comparison": True,
            "no_universal_best_model": True,
        },
    )
