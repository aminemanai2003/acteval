"""Version and input identity metadata for evaluation results."""

import hashlib
import platform
from collections.abc import Mapping
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from acteval._version import __version__
from acteval.exceptions import InputValidationError

_DEPENDENCIES = ("numpy", "pandas", "scikit-learn", "scipy")


@cache
def runtime_metadata() -> dict[str, Any]:
    """Return package, interpreter, and direct dependency versions."""
    dependency_versions: dict[str, str] = {}
    for dependency in _DEPENDENCIES:
        try:
            dependency_versions[dependency] = version(dependency)
        except PackageNotFoundError:
            dependency_versions[dependency] = "not-installed"
    return {
        "acteval_version": __version__,
        "python_version": platform.python_version(),
        "dependency_versions": dependency_versions,
    }


def input_fingerprint(**arrays: ArrayLike | None) -> str:
    """Hash named numerical inputs using a stable float64 representation."""
    digest = hashlib.sha256(b"acteval-input-v1")
    for name, values in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        if values is None:
            digest.update(b"<none>")
            continue
        array = np.asarray(values, dtype="<f8")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def validate_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and detach caller-supplied model/data provenance context."""
    if context is None:
        return {}
    result: dict[str, Any] = {}
    for key, value in context.items():
        if not isinstance(key, str) or not key.strip():
            raise InputValidationError(
                "Evaluation context keys must be non-empty strings."
            )
        result[key.strip()] = value
    return result
