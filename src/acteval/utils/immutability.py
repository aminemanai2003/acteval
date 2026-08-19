"""Defensive freezing and detachment for public result data."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast


def freeze(value: Any) -> Any:
    """Recursively freeze common mutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    return value


def freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a recursively immutable copy of a string-keyed mapping."""
    return cast(Mapping[str, Any], freeze(value))


def thaw(value: Any) -> Any:
    """Return detached built-in containers suitable for serialization."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(item) for item in value]
    if isinstance(value, (frozenset, set)):
        return [thaw(item) for item in value]
    return value
