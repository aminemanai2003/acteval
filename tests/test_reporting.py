import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import acteval as ae


def test_standalone_html_report_escapes_title_and_contains_metadata(
    tmp_path: Path,
) -> None:
    result = ae.evaluate(
        [1, 2, 3], [1, 2, 2.5], task="claim_frequency", metrics=["rmse"]
    )
    html = result.to_html(title="<ActEval & report>")
    assert "<!doctype html>" in html
    assert "&lt;ActEval &amp; report&gt;" in html
    assert "Evaluation metadata" in html
    assert "universal best model" in html
    destination = result.save_html(tmp_path / "reports" / "evaluation.html")
    assert destination.exists()
    assert destination.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_export_table_csv_json_and_html(tmp_path: Path) -> None:
    comparison = ae.compare(
        [1, 2, 3],
        {"A": [1, 2, 3], "B": [1.1, 1.9, 2.8]},
        task="claim_frequency",
        metrics=["rmse"],
    )
    csv_path = ae.export_table(comparison, tmp_path / "comparison.csv")
    json_path = ae.export_table(comparison, tmp_path / "comparison.json")
    html_path = ae.export_table(comparison, tmp_path / "comparison.html")
    assert "metric" in csv_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["task"] == (
        "claim_frequency"
    )
    assert "<!doctype html>" in html_path.read_text(encoding="utf-8")
    assert "Model A" not in comparison.to_html()
    with pytest.raises(ValueError, match="csv, json, html"):
        ae.export_table(comparison, tmp_path / "bad.txt")


def test_save_plot(tmp_path: Path) -> None:
    axis = ae.plot_residuals([1, 2, 3], [1.1, 1.9, 3.2])
    destination = ae.save_plot(axis, tmp_path / "plot.png", dpi=100)
    assert destination.exists()
    assert destination.stat().st_size > 0
    plt.close(axis.figure)
    with pytest.raises(ValueError, match="positive integer"):
        ae.save_plot(axis, tmp_path / "bad.png", dpi=0)
    with pytest.raises(TypeError, match="Matplotlib axis"):
        ae.save_plot(object(), tmp_path / "bad.png")


def test_json_export_is_strict_for_nonfinite_metric_values(tmp_path: Path) -> None:
    result = ae.EvaluationResult(
        "claim_frequency",
        {"positive": float("inf"), "negative": float("-inf"), "nan": float("nan")},
        {},
    )
    path = ae.export_table(result, tmp_path / "nonfinite.json")
    raw = path.read_text(encoding="utf-8")
    assert ": Infinity" not in raw
    assert ": NaN" not in raw
    assert json.loads(raw)["metrics"] == {
        "positive": "Infinity",
        "negative": "-Infinity",
        "nan": "NaN",
    }
