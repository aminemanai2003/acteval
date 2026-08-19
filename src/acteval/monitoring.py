"""Portfolio-segment, temporal, and prediction-drift diagnostics."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acteval.api import MetricSelection, compare, evaluate
from acteval.exceptions import InputValidationError
from acteval.reports import ComparisonResult, EvaluationResult
from acteval.types import NumericArray, Task
from acteval.utils import weighted_quantile
from acteval.validation import combine_weights, validate_inputs

ObjectArray = NDArray[np.object_]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SegmentEvaluationResult:
    """Metric results split by a caller-supplied portfolio segment."""

    task: Task
    segments: dict[str, EvaluationResult]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "task": self.task,
            "segments": {
                segment: result.to_dict() for segment, result in self.segments.items()
            },
            "metadata": dict(self.metadata),
        }

    def to_dataframe(self) -> Any:
        """Return metrics indexed by segment and metric."""

        import pandas as pd

        rows = [
            {"segment": segment, "metric": metric, "value": value}
            for segment, result in self.segments.items()
            for metric, value in result.metrics.items()
        ]
        return pd.DataFrame.from_records(rows).set_index(["segment", "metric"])

    def summary(self) -> str:
        """Return a concise printable segment table."""

        return str(self.to_dataframe().to_string())


@dataclass(frozen=True, slots=True)
class SegmentComparisonResult:
    """Multi-model comparisons repeated consistently within each segment."""

    task: Task
    segments: dict[str, ComparisonResult]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "task": self.task,
            "segments": {
                segment: result.to_dict() for segment, result in self.segments.items()
            },
            "metadata": dict(self.metadata),
        }

    def to_dataframe(self) -> Any:
        """Return segment and metric rows with one column per model."""

        import pandas as pd

        frames = {
            segment: comparison.to_dataframe()
            for segment, comparison in self.segments.items()
        }
        result = pd.concat(frames, names=["segment", "metric"])
        return result

    def summary(self) -> str:
        """Return a concise printable segmented comparison table."""

        return str(self.to_dataframe().to_string())


@dataclass(frozen=True, slots=True)
class TemporalEvaluationResult:
    """Chronologically ordered metric results and changes from baseline."""

    task: Task
    periods: dict[str, EvaluationResult]
    baseline_period: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "task": self.task,
            "baseline_period": self.baseline_period,
            "periods": {
                period: result.to_dict() for period, result in self.periods.items()
            },
            "metadata": dict(self.metadata),
        }

    def to_dataframe(self) -> Any:
        """Return period estimates and signed changes from the first period."""

        import pandas as pd

        baseline = self.periods[self.baseline_period]
        rows = []
        for period, result in self.periods.items():
            for metric, value in result.metrics.items():
                rows.append(
                    {
                        "period": period,
                        "metric": metric,
                        "value": value,
                        "baseline_value": baseline.metrics[metric],
                        "change_from_baseline": value - baseline.metrics[metric],
                    }
                )
        return pd.DataFrame.from_records(rows).set_index(["period", "metric"])

    def summary(self) -> str:
        """Return a concise printable temporal table."""

        return str(self.to_dataframe().to_string())


@dataclass(frozen=True, slots=True)
class DriftBin:
    """Reference and current portfolio shares in one fixed score bin."""

    bin: int
    lower: float
    upper: float
    reference_proportion: float
    current_proportion: float
    psi_contribution: float


@dataclass(frozen=True, slots=True)
class PredictionDriftResult:
    """Distribution shift diagnostics for predictions across two populations."""

    population_stability_index: float
    reference_mean: float
    current_mean: float
    mean_shift: float
    relative_mean_shift: float | None
    bins: tuple[DriftBin, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "population_stability_index": self.population_stability_index,
            "reference_mean": self.reference_mean,
            "current_mean": self.current_mean,
            "mean_shift": self.mean_shift,
            "relative_mean_shift": self.relative_mean_shift,
            "bins": [asdict(bin_result) for bin_result in self.bins],
            "metadata": dict(self.metadata),
        }

    def to_dataframe(self) -> Any:
        """Return one row per fixed prediction bin."""

        import pandas as pd

        return pd.DataFrame.from_records([asdict(row) for row in self.bins]).set_index(
            "bin"
        )

    def summary(self) -> str:
        """Return a concise printable drift report."""

        header = (
            f"PSI={self.population_stability_index:.6g}, "
            f"mean_shift={self.mean_shift:.6g}"
        )
        return f"{header}\n{self.to_dataframe().to_string()}"


def _labels(values: ArrayLike, *, name: str, expected_length: int) -> ObjectArray:
    import pandas as pd

    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise InputValidationError(f"{name} must be one-dimensional.")
    if len(array) != expected_length:
        raise InputValidationError(
            f"{name} has length {len(array)}; expected {expected_length}."
        )
    if any(
        pd.isna(value) or (isinstance(value, str) and not value.strip())
        for value in array
    ):
        raise InputValidationError(f"{name} must not contain missing or empty labels.")
    return cast(ObjectArray, array)


def _normalized_groups(values: ObjectArray) -> tuple[tuple[str, BoolArray], ...]:
    labels: list[str] = []
    masks: list[BoolArray] = []
    seen: dict[str, object] = {}
    for value in values:
        label = str(value).strip()
        if label in seen and seen[label] != value:
            raise InputValidationError(
                f"Distinct group values collapse to the same label {label!r}."
            )
        if label not in seen:
            seen[label] = value
            labels.append(label)
            masks.append(np.asarray(values == value, dtype=np.bool_))
    return tuple((label, mask) for label, mask in zip(labels, masks, strict=True))


def _slice(values: ArrayLike | None, mask: BoolArray) -> ArrayLike | None:
    if values is None:
        return None
    return np.asarray(values)[mask]


def _validate_min_observations(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InputValidationError("min_observations must be a positive integer.")
    return value


def evaluate_by_segment(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    segments: ArrayLike,
    *,
    task: str,
    exposure: ArrayLike | None = None,
    input_scale: str | None = None,
    sample_weight: ArrayLike | None = None,
    metrics: Sequence[MetricSelection] | None = None,
    min_observations: int = 2,
) -> SegmentEvaluationResult:
    """Evaluate one model separately within each portfolio segment."""

    validated = validate_inputs(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
        y_pred_domain="nonnegative",
    )
    minimum = _validate_min_observations(min_observations)
    labels = _labels(segments, name="segments", expected_length=len(validated.y_true))
    results: dict[str, EvaluationResult] = {}
    skipped: list[str] = []
    for label, mask in _normalized_groups(labels):
        if np.count_nonzero(mask) < minimum:
            skipped.append(label)
            continue
        results[label] = evaluate(
            validated.y_true[mask],
            validated.y_pred[mask],
            task=task,
            exposure=_slice(validated.exposure, mask),
            input_scale=input_scale,
            sample_weight=_slice(validated.sample_weight, mask),
            metrics=metrics,
        )
    if not results:
        raise InputValidationError("No segment meets min_observations.")
    first = next(iter(results.values()))
    return SegmentEvaluationResult(
        first.task,
        results,
        {
            "n_segments": len(results),
            "min_observations": minimum,
            "skipped_segments": tuple(skipped),
        },
    )


def compare_by_segment(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    segments: ArrayLike,
    *,
    task: str,
    exposure: ArrayLike | None = None,
    input_scale: str | None = None,
    sample_weight: ArrayLike | None = None,
    metrics: Sequence[MetricSelection] | None = None,
    min_observations: int = 2,
) -> SegmentComparisonResult:
    """Compare models within each portfolio segment using identical rows."""

    if not predictions:
        raise InputValidationError("predictions must contain at least one model.")
    minimum = _validate_min_observations(min_observations)
    first_prediction = next(iter(predictions.values()))
    validated = validate_inputs(
        y_true,
        first_prediction,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
        y_pred_domain="nonnegative",
    )
    labels = _labels(segments, name="segments", expected_length=len(validated.y_true))
    results: dict[str, ComparisonResult] = {}
    skipped: list[str] = []
    for label, mask in _normalized_groups(labels):
        if np.count_nonzero(mask) < minimum:
            skipped.append(label)
            continue
        results[label] = compare(
            validated.y_true[mask],
            {name: np.asarray(values)[mask] for name, values in predictions.items()},
            task=task,
            exposure=_slice(validated.exposure, mask),
            input_scale=input_scale,
            sample_weight=_slice(validated.sample_weight, mask),
            metrics=metrics,
        )
    if not results:
        raise InputValidationError("No segment meets min_observations.")
    first = next(iter(results.values()))
    return SegmentComparisonResult(
        first.task,
        results,
        {
            "n_segments": len(results),
            "min_observations": minimum,
            "skipped_segments": tuple(skipped),
            "no_universal_best_model": True,
        },
    )


def evaluate_over_time(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    periods: ArrayLike,
    *,
    task: str,
    exposure: ArrayLike | None = None,
    input_scale: str | None = None,
    sample_weight: ArrayLike | None = None,
    metrics: Sequence[MetricSelection] | None = None,
    min_observations: int = 2,
) -> TemporalEvaluationResult:
    """Evaluate chronologically sorted periods and changes from the first period.

    Period labels should sort chronologically as strings; ISO dates and ordered
    year/month or quarter labels satisfy this contract.
    """

    segmented = evaluate_by_segment(
        y_true,
        y_pred,
        periods,
        task=task,
        exposure=exposure,
        input_scale=input_scale,
        sample_weight=sample_weight,
        metrics=metrics,
        min_observations=min_observations,
    )
    ordered = dict(sorted(segmented.segments.items()))
    baseline = next(iter(ordered))
    return TemporalEvaluationResult(
        segmented.task,
        ordered,
        baseline,
        {
            **segmented.metadata,
            "period_order": tuple(ordered),
            "change_is_signed_current_minus_baseline": True,
        },
    )


def _weighted_mean(values: NumericArray, weights: NumericArray | None) -> float:
    return float(
        np.mean(values) if weights is None else np.average(values, weights=weights)
    )


def prediction_drift(
    reference_predictions: ArrayLike,
    current_predictions: ArrayLike,
    *,
    n_bins: int = 10,
    reference_weight: ArrayLike | None = None,
    current_weight: ArrayLike | None = None,
    epsilon: float = 1e-6,
) -> PredictionDriftResult:
    """Measure score-distribution drift with fixed reference quantile bins.

    PSI is descriptive and has no universal alert threshold. ActEval therefore
    returns the value and bin contributions without labelling drift as good,
    bad, significant, or operationally material.
    """

    reference = validate_inputs(
        reference_predictions,
        reference_predictions,
        sample_weight=reference_weight,
    )
    current = validate_inputs(
        current_predictions,
        current_predictions,
        sample_weight=current_weight,
    )
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise InputValidationError("n_bins must be an integer of at least 2.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise InputValidationError("epsilon must be strictly positive and finite.")
    reference_weights = combine_weights(reference)
    current_weights = combine_weights(current)
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    cut_points = np.unique(
        [
            weighted_quantile(reference.y_true, float(q), reference_weights)
            for q in quantiles
        ]
    )
    edges = np.concatenate(([-np.inf], cut_points, [np.inf]))
    reference_counts, _ = np.histogram(
        reference.y_true, bins=edges, weights=reference_weights
    )
    current_counts, _ = np.histogram(
        current.y_true, bins=edges, weights=current_weights
    )
    reference_proportions = reference_counts / np.sum(reference_counts)
    current_proportions = current_counts / np.sum(current_counts)
    reference_safe = np.clip(reference_proportions, epsilon, None)
    current_safe = np.clip(current_proportions, epsilon, None)
    contributions = (current_safe - reference_safe) * np.log(
        current_safe / reference_safe
    )
    bins = tuple(
        DriftBin(
            bin=index + 1,
            lower=float(edges[index]),
            upper=float(edges[index + 1]),
            reference_proportion=float(reference_proportions[index]),
            current_proportion=float(current_proportions[index]),
            psi_contribution=float(contributions[index]),
        )
        for index in range(len(edges) - 1)
    )
    reference_mean = _weighted_mean(reference.y_true, reference_weights)
    current_mean = _weighted_mean(current.y_true, current_weights)
    shift = current_mean - reference_mean
    relative = shift / reference_mean if reference_mean != 0 else None
    return PredictionDriftResult(
        population_stability_index=float(np.sum(contributions)),
        reference_mean=reference_mean,
        current_mean=current_mean,
        mean_shift=shift,
        relative_mean_shift=relative,
        bins=bins,
        metadata={
            "requested_bins": n_bins,
            "effective_bins": len(bins),
            "epsilon": epsilon,
            "psi_has_no_universal_threshold": True,
            "binning": "reference_prediction_quantiles",
        },
    )
