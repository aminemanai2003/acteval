"""Explicit financial decision and consequence diagnostics.

These functions do not define a universal actuarial objective. Every regret
names a model decision, a benchmark decision, and a financial loss function.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from acteval.exceptions import InputValidationError
from acteval.types import NumericArray, PredictiveDistribution
from acteval.utils import effective_weights, freeze_mapping, thaw
from acteval.validation import (
    as_1d_float_array,
    combine_weights,
    validate_inputs,
    validate_probability,
)

DecisionLoss = Callable[[NumericArray, NumericArray], NumericArray]


@dataclass(frozen=True, slots=True)
class DecisionEvaluation:
    """Financial loss of a model decision relative to a named benchmark."""

    decision: str
    model_loss: float
    benchmark_loss: float
    regret: float
    relative_regret: float | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "decision": self.decision,
            "model_loss": self.model_loss,
            "benchmark_loss": self.benchmark_loss,
            "regret": self.regret,
            "relative_regret": self.relative_regret,
            "metadata": thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class LossRatioResult:
    """Realized portfolio loss ratio and displacement from a target."""

    loss_ratio: float
    target_loss_ratio: float
    signed_impact: float
    absolute_impact: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ShortfallResult:
    """Aggregate and frequency diagnostics for an insufficient decision."""

    decision: str
    aggregate_shortfall: float
    mean_shortfall: float
    shortfall_frequency: float
    conditional_mean_shortfall: float
    total_weight: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReinsuranceOption:
    """Quoted excess-of-loss option for one loss aggregate.

    The cedent retains ``min(loss, retention)`` and pays ``premium``.
    """

    name: str
    retention: float
    premium: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InputValidationError("Reinsurance option name must not be empty.")
        if not np.isfinite(self.retention) or self.retention < 0:
            raise InputValidationError("Reinsurance retention must be nonnegative.")
        if not np.isfinite(self.premium) or self.premium < 0:
            raise InputValidationError("Reinsurance premium must be nonnegative.")


@dataclass(frozen=True, slots=True)
class ReinsuranceSelection:
    """Selected quoted option and modeled cost by option."""

    selected: ReinsuranceOption
    projected_costs: Mapping[str, float]
    risk_measure: str
    risk_quantile: float
    capital_cost_rate: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "projected_costs", freeze_mapping(self.projected_costs)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "selected": asdict(self.selected),
            "projected_costs": dict(self.projected_costs),
            "risk_measure": self.risk_measure,
            "risk_quantile": self.risk_quantile,
            "capital_cost_rate": self.capital_cost_rate,
        }


def _decisions(values: ArrayLike, *, length: int, name: str) -> NumericArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(length, float(array), dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise InputValidationError(f"{name} must be finite and have length {length}.")
    return array


def asymmetric_absolute_loss(
    y_true: NumericArray,
    decision: NumericArray,
    *,
    underprediction_cost: float = 1.0,
    overprediction_cost: float = 1.0,
) -> NumericArray:
    """Return asymmetric under/over-decision financial loss per observation."""
    if (
        not np.isfinite(underprediction_cost)
        or underprediction_cost < 0
        or not np.isfinite(overprediction_cost)
        or overprediction_cost < 0
    ):
        raise InputValidationError("Decision cost multipliers must be nonnegative.")
    if underprediction_cost == 0 and overprediction_cost == 0:
        raise InputValidationError("At least one decision cost must be positive.")
    return underprediction_cost * np.maximum(y_true - decision, 0) + (
        overprediction_cost * np.maximum(decision - y_true, 0)
    )


def decision_regret(
    y_true: ArrayLike,
    model_decision: ArrayLike,
    benchmark_decision: ArrayLike,
    *,
    loss_function: DecisionLoss,
    decision_name: str,
    benchmark_name: str = "benchmark",
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
    metadata: dict[str, Any] | None = None,
) -> DecisionEvaluation:
    """Evaluate financial loss difference from an explicit benchmark decision.

    Regret is ``model_loss - benchmark_loss`` in the loss function's financial
    unit. It may be negative when the model decision outperforms the benchmark.
    Relative regret is omitted when benchmark loss is zero.
    """
    inputs = validate_inputs(
        y_true,
        y_true,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
    )
    model_values = _decisions(
        model_decision, length=len(inputs.y_true), name="model_decision"
    )
    benchmark_values = _decisions(
        benchmark_decision,
        length=len(inputs.y_true),
        name="benchmark_decision",
    )
    model_losses = np.asarray(
        loss_function(inputs.y_true, model_values), dtype=np.float64
    )
    benchmark_losses = np.asarray(
        loss_function(inputs.y_true, benchmark_values), dtype=np.float64
    )
    expected_shape = inputs.y_true.shape
    if (
        model_losses.shape != expected_shape
        or benchmark_losses.shape != expected_shape
        or np.any(model_losses < 0)
        or np.any(benchmark_losses < 0)
        or not np.all(np.isfinite(model_losses))
        or not np.all(np.isfinite(benchmark_losses))
    ):
        raise InputValidationError(
            "loss_function must return finite nonnegative loss per observation."
        )
    weights = effective_weights(len(inputs.y_true), combine_weights(inputs))
    model_loss = float(np.average(model_losses, weights=weights))
    benchmark_loss = float(np.average(benchmark_losses, weights=weights))
    regret = model_loss - benchmark_loss
    relative_regret = regret / benchmark_loss if benchmark_loss > 0 else None
    result_metadata = {
        "benchmark": benchmark_name,
        "loss_unit": "same_as_loss_function",
        "lower_loss_is_better": True,
        "regret_can_be_negative": True,
    }
    if metadata:
        result_metadata.update(metadata)
    return DecisionEvaluation(
        decision=decision_name,
        model_loss=model_loss,
        benchmark_loss=benchmark_loss,
        regret=regret,
        relative_regret=relative_regret,
        metadata=result_metadata,
    )


def premium_from_distribution(
    distribution: PredictiveDistribution,
    *,
    profit_loading: float = 0.0,
    expense_ratio: float = 0.0,
) -> NumericArray:
    """Convert predictive means into premiums with explicit loadings.

    ``premium = mean * (1 + profit_loading) / (1 - expense_ratio)``.
    """
    if not np.isfinite(profit_loading) or profit_loading < 0:
        raise InputValidationError("profit_loading must be nonnegative.")
    if not np.isfinite(expense_ratio) or not 0 <= expense_ratio < 1:
        raise InputValidationError("expense_ratio must lie in [0, 1).")
    means = np.asarray(distribution.mean(), dtype=np.float64)
    if (
        means.shape != (distribution.n_observations,)
        or np.any(means < 0)
        or not np.all(np.isfinite(means))
    ):
        raise InputValidationError("Distribution means must be finite and nonnegative.")
    return means * (1 + profit_loading) / (1 - expense_ratio)


def pricing_regret(
    y_true: ArrayLike,
    premium: ArrayLike,
    benchmark_premium: ArrayLike,
    *,
    underpricing_cost: float = 1.0,
    overpricing_cost: float = 1.0,
    benchmark_name: str = "benchmark premium",
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> DecisionEvaluation:
    """Evaluate premium decisions under explicit asymmetric financial costs."""

    def loss(observed: NumericArray, decision: NumericArray) -> NumericArray:
        return asymmetric_absolute_loss(
            observed,
            decision,
            underprediction_cost=underpricing_cost,
            overprediction_cost=overpricing_cost,
        )

    return decision_regret(
        y_true,
        premium,
        benchmark_premium,
        loss_function=loss,
        decision_name="pricing",
        benchmark_name=benchmark_name,
        sample_weight=sample_weight,
        exposure=exposure,
        metadata={
            "underpricing_cost": underpricing_cost,
            "overpricing_cost": overpricing_cost,
        },
    )


def loss_ratio_impact(
    y_true: ArrayLike,
    premium: ArrayLike,
    *,
    target_loss_ratio: float = 1.0,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> LossRatioResult:
    """Compute realized aggregate loss ratio and displacement from a target."""
    if not np.isfinite(target_loss_ratio) or target_loss_ratio <= 0:
        raise InputValidationError("target_loss_ratio must be strictly positive.")
    inputs = validate_inputs(
        y_true,
        premium,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
        y_pred_domain="nonnegative",
    )
    weights = effective_weights(len(inputs.y_true), combine_weights(inputs))
    premium_total = float(np.sum(weights * inputs.y_pred))
    if premium_total <= 0:
        raise InputValidationError("Loss ratio is undefined when premium is zero.")
    ratio = float(np.sum(weights * inputs.y_true)) / premium_total
    impact = ratio - target_loss_ratio
    return LossRatioResult(ratio, target_loss_ratio, impact, abs(impact))


def _shortfall(
    y_true: ArrayLike,
    decision: ArrayLike,
    *,
    decision_name: str,
    sample_weight: ArrayLike | None,
    exposure: ArrayLike | None,
) -> ShortfallResult:
    inputs = validate_inputs(
        y_true,
        decision,
        sample_weight=sample_weight,
        exposure=exposure,
        y_true_domain="nonnegative",
        y_pred_domain="nonnegative",
    )
    weights = effective_weights(len(inputs.y_true), combine_weights(inputs))
    values = np.maximum(inputs.y_true - inputs.y_pred, 0)
    total_weight = float(np.sum(weights))
    aggregate = float(np.sum(weights * values))
    occurrence_weight = float(np.sum(weights[values > 0]))
    conditional = (
        float(np.sum(weights[values > 0] * values[values > 0]) / occurrence_weight)
        if occurrence_weight > 0
        else 0.0
    )
    return ShortfallResult(
        decision=decision_name,
        aggregate_shortfall=aggregate,
        mean_shortfall=aggregate / total_weight,
        shortfall_frequency=occurrence_weight / total_weight,
        conditional_mean_shortfall=conditional,
        total_weight=total_weight,
    )


def reserve_shortfall(
    y_true: ArrayLike,
    reserve: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> ShortfallResult:
    """Evaluate realized amounts by which losses exceed held reserves."""
    return _shortfall(
        y_true,
        reserve,
        decision_name="reserve",
        sample_weight=sample_weight,
        exposure=exposure,
    )


def capital_shortfall(
    y_true: ArrayLike,
    capital: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    exposure: ArrayLike | None = None,
) -> ShortfallResult:
    """Evaluate realized amounts by which losses exceed available capital."""
    return _shortfall(
        y_true,
        capital,
        decision_name="capital",
        sample_weight=sample_weight,
        exposure=exposure,
    )


def quantile_decision(
    distribution: PredictiveDistribution,
    *,
    quantile: float,
) -> NumericArray:
    """Return a per-observation distribution quantile as a decision array."""
    probability = validate_probability(quantile, name="quantile")
    values = np.asarray(distribution.quantile(probability), dtype=np.float64)
    if values.shape != (distribution.n_observations,) or not np.all(
        np.isfinite(values)
    ):
        raise InputValidationError(
            "Distribution quantile has an invalid shape or value."
        )
    return values


def _tail_mean(samples: NumericArray, quantile: float) -> NumericArray:
    """Return empirical expected shortfall with fractional boundary mass.

    Averaging every value at or above VaR is only equivalent to expected
    shortfall for continuous distributions.  Insurance losses commonly have
    atoms, so integrate the empirical quantile function over the requested
    upper tail instead.  Fractional boundary weight makes the result coherent
    even when ``(1 - quantile) * n_samples`` is not an integer.
    """
    ordered = np.sort(samples, axis=0)
    tail_mass = (1.0 - quantile) * len(ordered)
    full_count = int(np.floor(tail_mass))
    boundary_weight = tail_mass - full_count
    totals = np.zeros(ordered.shape[1], dtype=np.float64)
    if full_count:
        totals += np.sum(ordered[-full_count:], axis=0)
    if boundary_weight:
        boundary_index = len(ordered) - full_count - 1
        totals += boundary_weight * ordered[boundary_index]
    return totals / tail_mass


def select_reinsurance_option(
    distribution: PredictiveDistribution,
    options: Sequence[ReinsuranceOption],
    *,
    risk_measure: Literal["var", "expected_shortfall"] = "expected_shortfall",
    risk_quantile: float = 0.995,
    capital_cost_rate: float = 0.0,
    n_samples: int = 20_000,
    random_state: int | np.random.Generator | None = 0,
) -> ReinsuranceSelection:
    """Select a quoted stop-loss option under an explicit projected-cost rule.

    For each option, projected cost is reinsurance premium plus mean retained
    loss plus ``capital_cost_rate`` times retained-loss VaR or expected
    shortfall. This is one documented decision rule, not a universal optimum.
    """
    if not options:
        raise InputValidationError("options must contain at least one quote.")
    names = [option.name for option in options]
    if len(set(names)) != len(names):
        raise InputValidationError("Reinsurance option names must be unique.")
    probability = validate_probability(risk_quantile, name="risk_quantile")
    if not np.isfinite(capital_cost_rate) or capital_cost_rate < 0:
        raise InputValidationError("capital_cost_rate must be nonnegative.")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
        raise InputValidationError("n_samples must be an integer of at least 2.")
    if risk_measure not in {"var", "expected_shortfall"}:
        raise InputValidationError(
            "risk_measure must be 'var' or 'expected_shortfall'."
        )
    samples = np.asarray(
        distribution.sample(n_samples, random_state=random_state), dtype=np.float64
    )
    expected_shape = (n_samples, distribution.n_observations)
    if samples.shape != expected_shape or not np.all(np.isfinite(samples)):
        raise InputValidationError(
            f"Distribution samples must be finite with shape {expected_shape}."
        )
    aggregate_losses = np.sum(samples, axis=1, keepdims=True)
    projected_costs: dict[str, float] = {}
    for option in options:
        retained = np.minimum(aggregate_losses, option.retention)
        mean_retained = float(np.mean(retained))
        if risk_measure == "var":
            capital_measure = float(np.quantile(retained, probability))
        else:
            capital_measure = float(_tail_mean(retained, probability)[0])
        projected_costs[option.name] = (
            option.premium + mean_retained + capital_cost_rate * capital_measure
        )
    selected_name = min(projected_costs, key=projected_costs.__getitem__)
    selected = next(option for option in options if option.name == selected_name)
    return ReinsuranceSelection(
        selected=selected,
        projected_costs=projected_costs,
        risk_measure=risk_measure,
        risk_quantile=probability,
        capital_cost_rate=capital_cost_rate,
    )


def reinsurance_decision_regret(
    aggregate_loss: ArrayLike,
    selected: ReinsuranceOption,
    benchmark: ReinsuranceOption,
    *,
    benchmark_name: str = "benchmark reinsurance option",
    sample_weight: ArrayLike | None = None,
) -> DecisionEvaluation:
    """Compare realized premium-plus-retained-loss cost for two options."""
    observed = as_1d_float_array(aggregate_loss, name="aggregate_loss")
    if np.any(observed < 0):
        raise InputValidationError("aggregate_loss must be nonnegative.")
    selected_cost = selected.premium + np.minimum(observed, selected.retention)
    benchmark_cost = benchmark.premium + np.minimum(observed, benchmark.retention)
    return decision_regret(
        observed,
        selected_cost,
        benchmark_cost,
        loss_function=lambda _observed, decision: decision,
        decision_name="reinsurance",
        benchmark_name=benchmark_name,
        sample_weight=sample_weight,
        metadata={
            "selected_option": asdict(selected),
            "benchmark_option": asdict(benchmark),
            "realized_cost": "premium + min(loss, retention)",
        },
    )
