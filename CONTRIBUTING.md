# Contributing to ActEval

ActEval welcomes focused fixes, tests, documentation, and actuarially justified
metrics.

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pre-commit install
```

Before opening a pull request, run:

```bash
ruff check .
ruff format --check .
mypy src/acteval
pytest
python -m build
```

CI must pass on every Python version listed in `pyproject.toml`. Public API
changes must update the API guide, README, changelog, tests, and stability
policy where relevant. Backward-incompatible changes require a major release
unless they correct a security issue or materially invalid calculation.

New metrics must document their definition, direction or target, supported
tasks, domain assumptions, limitations, and references where applicable. Add
basic, weighted, edge, invalid-input, and known-value tests. Do not introduce a
composite model score without theoretical justification.
