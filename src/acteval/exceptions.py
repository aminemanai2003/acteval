"""Custom exceptions raised by ActEval."""


class ActEvalError(Exception):
    """Base class for ActEval-specific errors."""


class InputValidationError(ActEvalError, ValueError):
    """Raised when metric inputs violate their documented contract."""


class MetricRegistrationError(ActEvalError, RuntimeError):
    """Raised when a metric cannot be added to the registry."""


class UnknownMetricError(ActEvalError, KeyError):
    """Raised when a metric name is not registered."""
