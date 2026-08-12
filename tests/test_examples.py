import json
import runpy
from pathlib import Path


def test_synthetic_example_executes(capsys: object) -> None:
    runpy.run_path("examples/synthetic_frequency.py", run_name="__main__")
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Model A" in output
    assert "tail_ae_95" in output


def test_probabilistic_example_executes(capsys: object) -> None:
    runpy.run_path("examples/probabilistic_frequency.py", run_name="__main__")
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Negative Binomial" in output
    assert "log_score" in output


def test_decision_example_executes(capsys: object) -> None:
    runpy.run_path("examples/decision_aware.py", run_name="__main__")
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Pricing:" in output
    assert "Reinsurance selection:" in output


def test_inference_and_monitoring_example_executes(
    capsys: object, tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    example = Path(__file__).parents[1] / "examples" / "inference_and_monitoring.py"
    runpy.run_path(str(example), run_name="__main__")
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "objective_delta" not in output
    assert "PSI=" in output
    assert (tmp_path / "acteval-example-report.html").exists()


def test_example_notebook_is_valid_notebook_json() -> None:
    notebook = json.loads(
        Path("examples/synthetic_frequency.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 3
