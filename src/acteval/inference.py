"""Bootstrap uncertainty for actuarial evaluation and model comparison."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acteval.api import MetricSelection, compare, evaluate
from acteval.exceptions import ActEvalError, InputValidationError
from acteval.metrics import calibration_by_quantile
from acteval.reports import EvaluationResult
from acteval.types import NumericArray, Task
from acteval.utils import risk_bin_indices
from acteval.validation import (
    combine_weights,
    validate_input_scale,
    validate_inputs,
    validate_probability,
)

BootstrapMethod = Literal["percentile"]


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A two-sided bootstrap confidence interval for one scalar estimand."""

    estimate: float
    lower: float
    upper: float
    standard_error: float
    confidence_level: float
    n_resamples: int
    method: BootstrapMethod = "percentile"

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-friendly representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BootstrapEvaluationResult:
    """Point estimates and bootstrap intervals for one model."""

    task: Task
    point_estimate: EvaluationResult
    intervals: dict[str, ConfidenceInterval]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "task": self.task,
            "point_estimate": self.point_estimate.to_dict(),
            "intervals": {
                metric: interval.to_dict()
                for metric, interval in self.intervals.items()
            },
            "metadata": dict(self.metadata),
        }

    def to_dataframe(self) -> Any:
        """Return one row per metric as a pandas DataFrame."""

        import pandas as pd

        rows = [
            {"metric": metric, **interval.to_dict()}
            for metric, interval in self.intervals.items()
        ]
        return pd.DataFrame.from_records(rows).set_index("metric")

    def summary(self) -> str:
        """Return a concise printable interval table."""

        return str(self.to_dataframe().to_string())


@dataclass(frozen=True, slots=True)
class PairedMetricComparison:
    """Paired bootstrap comparison of one model with a reference model.

    ``objective_delta`` is transformed so that negative values favor ``model``:
    lower-is-better metrics use ``model - reference``; higher-is-better metrics
    reverse the sign; target metrics compare absolute distance from the target.
    """

    model: str
    reference: str
    metric: str
    model_estimate: float
    reference_estimate: float
    raw_delta: float
    objective_delta: float
    lower: float
    upper: float
    standard_error: float
    confidence_level: float
    n_resamples: int

    @property
    def confidence_excludes_zero(self) -> bool:
        """Whether the two-sided interval excludes no objective difference."""

        return self.lower > 0 or self.upper < 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            **asdict(self),
            "confidence_excludes_zero": self.confidence_excludes_zero,
        }


@dataclass(frozen=True, slots=True)
class PairedComparisonResult:
    """Metric-specific paired bootstrap comparisons against one reference."""

    task: Task
    reference: str
    comparisons: tuple[PairedMetricComparison, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "task": self.task,
            "reference": self.reference,
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "metadata": dict(self.metadata),
        }

    def to_dataframe(self) -> Any:
        """Return one row per model and metric as a pandas DataFrame."""

        import pandas as pd

        frame = pd.DataFrame.from_records(
            [comparison.to_dict() for comparison in self.comparisons]
        )
        return frame.set_index(["model", "metric"])

    def summary(self) -> str:
        """Return a concise printable comparison table."""

        return str(self.to_dataframe().to_string())


@dataclass(frozen=True, slots=True)
class CalibrationIntervalBin:
    """Fixed risk-bin calibration estimates with bootstrap intervals."""

    bin: int
    count: int
    mean_prediction: ConfidenceInterval
    mean_observed: ConfidenceInterval
    ae_ratio: ConfidenceInterval

    def to_dict(self) -> dict[str, Any]:
        """Flatten the bin into a tabular JSON-friendly record."""

        record: dict[str, Any] = {"bin": self.bin, "count": self.count}
        for name in ("mean_prediction", "mean_observed", "ae_ratio"):
            interval = getattr(self, name)
            record[name] = interval.estimate
            record[f"{name}_lower"] = interval.lower
            record[f"{name}_upper"] = interval.upper
        return record


@dataclass(frozen=True, slots=True)
class CalibrationIntervalTable:
    """Risk calibration table with fixed-bin stratified bootstrap intervals."""

    bins: tuple[CalibrationIntervalBin, ...]
    requested_bins: int
    confidence_level: float
    n_resamples: int
    metadata: dict[str, Any]

    @property
    def effective_bins(self) -> int:
        """Number of populated bins after prediction ties are collapsed."""

        return len(self.bins)

    def to_dict(self) -> list[dict[str, Any]]:
        """Return one JSON-friendly record per bin."""

        return [bin_result.to_dict() for bin_result in self.bins]

    def to_dataframe(self) -> Any:
        """Return one row per risk bin as a pandas DataFrame."""

        import pandas as pd

        return pd.DataFrame.from_records(self.to_dict()).set_index("bin")


def _validate_bootstrap(n_resamples: int, confidence_level: float) -> tuple[int, float]:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise InputValidationError("n_resamples must be an integer.")
    if n_resamples < 100:
        raise InputValidationError("n_resamples must be at least 100.")
    return n_resamples, validate_probability(confidence_level, name="confidence_level")


def _interval(
    estimate: float,
    samples: NumericArray,
    *,
    confidence_level: float,
) -> ConfidenceInterval:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return ConfidenceInterval(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        standard_error=float(np.std(samples, ddof=1)),
        confidence_level=confidence_level,
        n_resamples=len(samples),
    )


def _slice_optional(
    values: NumericArray | None, indices: NDArray[np.integer[Any]]
) -> NumericArray | None:
    return None if values is None else values[indices.astype(np.intp)]


def bootstrap_evaluate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    task: str,
    exposure: ArrayLike | None = None,
    input_scale: str | None = None,
    sample_weight: ArrayLike | None = None,
    metrics: Sequence[MetricSelection] | None = None,
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    random_state: int | None = 0,
) -> BootstrapEvaluationResult:
    """Estimate percentile bootstrap intervals for point-prediction metrics.

    Rows are resampled jointly, preserving alignment among observations,
    predictions, exposures, and weights. An undefined resample fails the run
    instead of being discarded, which avoids conditioning the bootstrap
    distribution on estimand validity.
    """

    count, confidence = _validate_bootstrap(n_resamples, confidence_level)
    validated = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
        y_pred_domain="nonnegative",
    )
    point = evaluate(
        validated.y_true,
        validated.y_pred,
        task=task,
        exposure=validated.exposure,
        input_scale=input_scale,
        sample_weight=validated.sample_weight,
        metrics=metrics,
    )
    metric_samples: dict[str, list[float]] = {metric: [] for metric in point.metrics}
    generator = np.random.default_rng(random_state)
    for resample_index in range(count):
        indices = generator.integers(0, len(validated.y_true), len(validated.y_true))
        try:
            result = evaluate(
                validated.y_true[indices],
                validated.y_pred[indices],
                task=task,
                exposure=_slice_optional(validated.exposure, indices),
                input_scale=input_scale,
                sample_weight=_slice_optional(validated.sample_weight, indices),
                metrics=metrics,
            )
        except ActEvalError as error:
            raise InputValidationError(
                f"Bootstrap resample {resample_index + 1} of {count} made a "
                "requested metric undefined. No resamples were discarded; use "
                "an estimand defined on degenerate samples or a different "
                "inference design."
            ) from error
        if not all(np.isfinite(value) for value in result.metrics.values()):
            raise InputValidationError(
                f"Bootstrap resample {resample_index + 1} of {count} produced a "
                "non-finite metric. No resamples were discarded."
            )
        for metric, value in result.metrics.items():
            metric_samples[metric].append(value)
    intervals = {
        metric: _interval(
            point.metrics[metric],
            np.asarray(samples, dtype=np.float64),
            confidence_level=confidence,
        )
        for metric, samples in metric_samples.items()
    }
    return BootstrapEvaluationResult(
        point.task,
        point,
        intervals,
        {
            "method": "iid_percentile_bootstrap",
            "n_resamples": count,
            "confidence_level": confidence,
            "random_state": random_state,
            "undefined_resample_policy": "fail",
        },
    )


def _objective_value(value: float, metadata: Mapping[str, Any]) -> float:
    target = metadata.get("target")
    if target is not None:
        return abs(value - float(target))
    direction = metadata.get("higher_is_better")
    if direction is True:
        return -value
    if direction is False:
        return value
    raise InputValidationError("Metric has no comparison direction or target.")


def paired_bootstrap_compare(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    *,
    task: str,
    reference: str | None = None,
    exposure: ArrayLike | None = None,
    input_scale: str | None = None,
    sample_weight: ArrayLike | None = None,
    metrics: Sequence[MetricSelection] | None = None,
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    random_state: int | None = 0,
) -> PairedComparisonResult:
    """Compare models using shared row resamples and objective-aware deltas.

    Pairing removes bootstrap noise caused by evaluating models on different
    resamples. A negative objective delta favors the candidate model. Confidence
    intervals quantify sampling uncertainty and are not multiple-test adjusted.
    """

    count, confidence = _validate_bootstrap(n_resamples, confidence_level)
    point = compare(
        y_true,
        predictions,
        task=task,
        exposure=exposure,
        input_scale=input_scale,
        sample_weight=sample_weight,
        metrics=metrics,
    )
    model_names = tuple(point.results)
    if reference is not None and (
        not isinstance(reference, str) or not reference.strip()
    ):
        raise InputValidationError("reference must be a non-empty model name.")
    reference_name = model_names[0] if reference is None else reference.strip()
    if reference_name not in point.results:
        raise InputValidationError(f"Unknown reference model: {reference!r}.")
    candidates = tuple(name for name in model_names if name != reference_name)
    if not candidates:
        raise InputValidationError("At least two models are required for comparison.")
    validated_predictions: dict[str, NumericArray] = {}
    baseline_inputs = None
    for name, prediction in predictions.items():
        current = validate_inputs(
            y_true,
            prediction,
            sample_weight=sample_weight,
            exposure=exposure,
            y_true_domain="nonnegative",
            y_pred_domain="nonnegative",
        )
        baseline_inputs = current
        validated_predictions[name.strip()] = current.y_pred
    assert baseline_inputs is not None
    metric_names = tuple(point.results[reference_name].metrics)
    samples: dict[tuple[str, str], list[float]] = {
        (model, metric): [] for model in candidates for metric in metric_names
    }
    generator = np.random.default_rng(random_state)
    for resample_index in range(count):
        indices = generator.integers(
            0, len(baseline_inputs.y_true), len(baseline_inputs.y_true)
        )
        try:
            result = compare(
                baseline_inputs.y_true[indices],
                {
                    name: prediction[indices]
                    for name, prediction in validated_predictions.items()
                },
                task=task,
                exposure=_slice_optional(baseline_inputs.exposure, indices),
                input_scale=input_scale,
                sample_weight=_slice_optional(baseline_inputs.sample_weight, indices),
                metrics=metrics,
            )
        except ActEvalError as error:
            raise InputValidationError(
                f"Paired bootstrap resample {resample_index + 1} of {count} made "
                "a requested metric undefined. No resamples were discarded."
            ) from error
        reference_result = result.results[reference_name]
        if not all(
            np.isfinite(metric_value)
            for evaluation in result.results.values()
            for metric_value in evaluation.metrics.values()
        ):
            raise InputValidationError(
                f"Paired bootstrap resample {resample_index + 1} of {count} "
                "produced a non-finite metric. No resamples were discarded."
            )
        for model in candidates:
            for metric in metric_names:
                metadata = reference_result.metadata["metric_specs"][metric]
                model_value = result.results[model].metrics[metric]
                reference_value = reference_result.metrics[metric]
                delta = _objective_value(model_value, metadata) - _objective_value(
                    reference_value, metadata
                )
                samples[(model, metric)].append(delta)
    comparisons: list[PairedMetricComparison] = []
    reference_result = point.results[reference_name]
    for model in candidates:
        model_result = point.results[model]
        for metric in metric_names:
            metadata = reference_result.metadata["metric_specs"][metric]
            model_value = model_result.metrics[metric]
            reference_value = reference_result.metrics[metric]
            objective_delta = _objective_value(
                model_value, metadata
            ) - _objective_value(reference_value, metadata)
            interval = _interval(
                objective_delta,
                np.asarray(samples[(model, metric)], dtype=np.float64),
                confidence_level=confidence,
            )
            comparisons.append(
                PairedMetricComparison(
                    model=model,
                    reference=reference_name,
                    metric=metric,
                    model_estimate=model_value,
                    reference_estimate=reference_value,
                    raw_delta=model_value - reference_value,
                    objective_delta=objective_delta,
                    lower=interval.lower,
                    upper=interval.upper,
                    standard_error=interval.standard_error,
                    confidence_level=confidence,
                    n_resamples=count,
                )
            )
    return PairedComparisonResult(
        point.task,
        reference_name,
        tuple(comparisons),
        {
            "method": "paired_iid_percentile_bootstrap",
            "n_resamples": count,
            "confidence_level": confidence,
            "random_state": random_state,
            "undefined_resample_policy": "fail",
            "negative_objective_delta_favors_candidate": True,
            "multiple_testing_adjustment": None,
        },
    )


def bootstrap_calibration_by_quantile(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    n_bins: int = 10,
    exposure: ArrayLike | None = None,
    input_scale: str | None = None,
    sample_weight: ArrayLike | None = None,
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    random_state: int | None = 0,
) -> CalibrationIntervalTable:
    """Add stratified percentile intervals to fixed prediction-risk bins.

    Bins are defined once from the original predictions. Rows are then
    resampled within each bin, so intervals describe conditional calibration
    without allowing bootstrap cut points to change their interpretation.
    """

    count, confidence = _validate_bootstrap(n_resamples, confidence_level)
    resolved_scale = validate_input_scale(input_scale, exposure=exposure)
    validated = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
        y_pred_domain="nonnegative",
    )
    weights = combine_weights(validated)
    effective = (
        np.ones(len(validated.y_true), dtype=np.float64) if weights is None else weights
    )
    point_table = calibration_by_quantile(
        validated.y_true,
        validated.y_pred,
        n_bins=n_bins,
        exposure=validated.exposure,
        sample_weight=validated.sample_weight,
    )
    assignments = risk_bin_indices(
        validated.y_pred, n_bins=n_bins, weights=effective
    ).astype(np.intp)
    generator = np.random.default_rng(random_state)
    result_bins: list[CalibrationIntervalBin] = []
    for point_bin in point_table.bins:
        members = np.flatnonzero((assignments == point_bin.bin) & (effective > 0))
        if not len(members):
            continue
        predicted_values: list[float] = []
        observed_values: list[float] = []
        ratio_values: list[float] = []
        for resample_index in range(count):
            selected = generator.choice(members, size=len(members), replace=True)
            selected_weights = effective[selected]
            weight_total = np.sum(selected_weights)
            predicted_total = np.sum(selected_weights * validated.y_pred[selected])
            observed_total = np.sum(selected_weights * validated.y_true[selected])
            if weight_total <= 0 or predicted_total <= 0:
                raise InputValidationError(
                    f"Calibration bootstrap resample {resample_index + 1} of "
                    f"{count} is undefined in risk bin {point_bin.bin}. No "
                    "resamples were discarded."
                )
            predicted_values.append(float(predicted_total / weight_total))
            observed_values.append(float(observed_total / weight_total))
            ratio_values.append(float(observed_total / predicted_total))
        predicted_samples = np.asarray(predicted_values, dtype=np.float64)
        observed_samples = np.asarray(observed_values, dtype=np.float64)
        ratio_samples = np.asarray(ratio_values, dtype=np.float64)
        result_bins.append(
            CalibrationIntervalBin(
                bin=point_bin.bin,
                count=point_bin.count,
                mean_prediction=_interval(
                    point_bin.mean_prediction,
                    predicted_samples,
                    confidence_level=confidence,
                ),
                mean_observed=_interval(
                    point_bin.mean_observed,
                    observed_samples,
                    confidence_level=confidence,
                ),
                ae_ratio=_interval(
                    point_bin.ae_ratio,
                    ratio_samples,
                    confidence_level=confidence,
                ),
            )
        )
    return CalibrationIntervalTable(
        tuple(result_bins),
        n_bins,
        confidence,
        count,
        {
            "method": "fixed_bin_stratified_percentile_bootstrap",
            "random_state": random_state,
            "input_scale": resolved_scale,
            "undefined_resample_policy": "fail",
            "zero_effective_weight_rows_excluded": True,
        },
    )


def save_interval_csv(
    result: BootstrapEvaluationResult
    | PairedComparisonResult
    | CalibrationIntervalTable,
    path: str | Path,
) -> Path:
    """Write an inference result table to CSV and return the resolved path."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_dataframe().to_csv(destination)
    return destination
