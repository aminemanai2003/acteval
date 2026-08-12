# API stability policy

ActEval follows Semantic Versioning beginning with version 1.0.0.

## Public API

Names documented in the README, files under `docs/`, and `acteval.__all__` are
public. Their call signatures, return types, field meanings, and documented
mathematical conventions are compatibility commitments within a major release.

Modules, functions, and attributes beginning with an underscore are internal.
Registry implementation details and exact text formatting of `summary()` are not
stable serialization formats; use `to_dict()` or `to_dataframe()` instead.

## Deprecations

An incompatible public change will normally be introduced with a documented
deprecation warning and retained for at least one minor release before removal.
Immediate removal may occur for a security issue, materially incorrect
calculation, or behavior that cannot be preserved safely. Such changes will be
called out prominently in the changelog.

## Numerical compatibility

Metric definitions, weighting direction, tail-selection rules, distribution
parameterizations, and decision-loss conventions are public behavior. Floating
point results can vary slightly across supported NumPy/SciPy versions and
platforms. Seeded Monte Carlo and bootstrap methods promise reproducible random
streams for a fixed supported dependency stack, not bitwise equality forever.

## Supported Python

The package metadata and CI matrix define supported Python versions. Dropping a
Python version requires at least a minor release and changelog entry.
