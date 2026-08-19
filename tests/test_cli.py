import json
from pathlib import Path

import pandas as pd
import pytest

from acteval.cli import main


@pytest.fixture
def prediction_csv(tmp_path: Path) -> Path:
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "actual": [0.0, 1.0, 2.0, 4.0],
            "estimate": [0.1, 0.9, 1.8, 3.7],
            "exposure": [1.0, 0.5, 2.0, 1.5],
            "weight": [1.0, 2.0, 1.0, 1.0],
        }
    ).to_csv(path, index=False)
    return path


def test_evaluate_prints_selected_metrics(
    prediction_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "evaluate",
            str(prediction_csv),
            "--task",
            "claim_frequency",
            "--observed",
            "actual",
            "--predicted",
            "estimate",
            "--metric",
            "rmse",
            "--metric",
            "ae_ratio",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "rmse" in captured.out
    assert "ae_ratio" in captured.out
    assert captured.err == ""


def test_evaluate_uses_default_column_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "defaults.csv"
    pd.DataFrame({"y_true": [0.0, 1.0, 2.0], "y_pred": [0.1, 0.9, 2.1]}).to_csv(
        path, index=False
    )

    status = main(
        [
            "evaluate",
            str(path),
            "--task",
            "claim_frequency",
            "--metric",
            "rmse",
        ]
    )

    assert status == 0
    assert "rmse" in capsys.readouterr().out


def test_evaluate_exports_json_with_weighting_metadata(
    prediction_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "reports" / "evaluation.json"
    status = main(
        [
            "evaluate",
            str(prediction_csv),
            "--task",
            "claim_frequency",
            "--observed",
            "actual",
            "--predicted",
            "estimate",
            "--exposure",
            "exposure",
            "--sample-weight",
            "weight",
            "--input-scale",
            "rate",
            "--metric",
            "rmse",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert Path(captured.out.strip()) == output.resolve()
    assert payload["task"] == "claim_frequency"
    assert payload["metadata"]["weighting"] == "sample_weight * exposure"
    assert payload["metadata"]["evaluation_context"]["source"] == str(prediction_csv)


def test_evaluate_reports_missing_column_without_traceback(
    prediction_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "evaluate",
            str(prediction_csv),
            "--task",
            "claim_frequency",
            "--observed",
            "missing",
            "--predicted",
            "estimate",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert "column 'missing' was not found" in captured.err
    assert "Traceback" not in captured.err


def test_evaluate_requires_explicit_rate_scale_for_exposure(
    prediction_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "evaluate",
            str(prediction_csv),
            "--task",
            "claim_frequency",
            "--observed",
            "actual",
            "--predicted",
            "estimate",
            "--exposure",
            "exposure",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "input_scale='rate' is required" in captured.err


def test_evaluate_does_not_replace_existing_output_without_force(
    prediction_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "result.csv"
    output.write_text("keep me", encoding="utf-8")
    arguments = [
        "evaluate",
        str(prediction_csv),
        "--task",
        "claim_frequency",
        "--observed",
        "actual",
        "--predicted",
        "estimate",
        "--metric",
        "rmse",
        "--output",
        str(output),
    ]

    assert main(arguments) == 1
    assert output.read_text(encoding="utf-8") == "keep me"
    assert "pass --force" in capsys.readouterr().err

    assert main([*arguments, "--force"]) == 0
    assert "rmse" in output.read_text(encoding="utf-8")


def test_evaluate_never_replaces_its_input(
    prediction_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    original = prediction_csv.read_bytes()
    status = main(
        [
            "evaluate",
            str(prediction_csv),
            "--task",
            "claim_frequency",
            "--observed",
            "actual",
            "--predicted",
            "estimate",
            "--output",
            str(prediction_csv),
            "--force",
        ]
    )

    assert status == 1
    assert prediction_csv.read_bytes() == original
    assert "must not replace the input CSV" in capsys.readouterr().err


def test_evaluate_rejects_force_without_output(
    prediction_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "evaluate",
            str(prediction_csv),
            "--task",
            "claim_frequency",
            "--force",
        ]
    )

    assert status == 1
    assert "--force requires --output" in capsys.readouterr().err


def test_evaluate_rejects_unknown_output_extension(
    prediction_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "result.txt"
    status = main(
        [
            "evaluate",
            str(prediction_csv),
            "--task",
            "claim_frequency",
            "--observed",
            "actual",
            "--predicted",
            "estimate",
            "--output",
            str(output),
        ]
    )

    assert status == 1
    assert not output.exists()
    assert "format must be one of" in capsys.readouterr().err


def test_evaluate_reports_unreadable_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.csv"
    status = main(["evaluate", str(missing), "--task", "claim_frequency"])

    assert status == 1
    assert "could not read" in capsys.readouterr().err
