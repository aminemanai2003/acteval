from pathlib import Path

import pandas as pd
import pytest

import space.acteval_space as demo


@pytest.fixture
def upload(tmp_path: Path) -> Path:
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "policy_id": ["private-a", "private-b", "private-c", "private-d"],
            "y_true": [0.0, 0.2, 0.4, 0.8],
            "y_pred": [0.05, 0.18, 0.45, 0.72],
            "exposure": [1.0, 0.5, 1.5, 0.8],
            "weight": [1.0, 2.0, 1.0, 1.0],
        }
    ).to_csv(path, index=False)
    return path


def test_profile_upload_returns_bounded_preview_and_suggestions(upload: Path) -> None:
    profile = demo.profile_upload(upload)

    assert profile.rows == 4
    assert profile.columns == (
        "policy_id",
        "y_true",
        "y_pred",
        "exposure",
        "weight",
    )
    assert len(profile.preview) == 4
    assert demo.suggested_column(profile.columns, "observed") == "y_true"
    assert demo.suggested_column(profile.columns, "predicted") == "y_pred"
    assert demo.suggested_column(profile.columns, "exposure") == "exposure"
    assert demo.suggested_column(profile.columns, "sample_weight") == "weight"


def test_evaluate_upload_produces_metrics_bands_and_aggregate_report(
    upload: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(demo, "REPORT_DIRECTORY", tmp_path / "reports")

    artifacts = demo.evaluate_upload(
        upload,
        task="claim_frequency",
        observed_column="y_true",
        predicted_column="y_pred",
        exposure_column="exposure",
        sample_weight_column="weight",
        input_scale="rate",
    )

    assert set(artifacts.metrics["Metric"]) == {
        "rmse",
        "poisson_deviance",
        "ae_ratio",
        "normalized_gini",
    }
    assert set(artifacts.calibration["Series"]) == {"Observed", "Predicted"}
    assert "sample weight * exposure" in artifacts.summary_html
    report = artifacts.report_path.read_text(encoding="utf-8")
    assert "ActEval interactive evaluation" in report
    assert "private-a" not in report


@pytest.mark.parametrize("name", ["data.json", "data.csv.gz"])
def test_read_upload_rejects_non_csv_extensions(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("y_true,y_pred\n1,1\n", encoding="utf-8")

    with pytest.raises(demo.SpaceInputError, match="Only uncompressed"):
        demo.read_upload(path)


def test_read_upload_rejects_duplicate_or_blank_headers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("value,value\n1,2\n", encoding="utf-8")
    blank = tmp_path / "blank.csv"
    blank.write_text("value,\n1,2\n", encoding="utf-8")

    with pytest.raises(demo.SpaceInputError, match="unique"):
        demo.read_upload(duplicate)
    with pytest.raises(demo.SpaceInputError, match="non-empty"):
        demo.read_upload(blank)


def test_read_upload_enforces_file_and_row_limits(
    upload: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(demo, "MAX_FILE_SIZE_BYTES", 1)
    with pytest.raises(demo.SpaceInputError, match="5 MiB"):
        demo.read_upload(upload)

    monkeypatch.setattr(demo, "MAX_FILE_SIZE_BYTES", 5 * 1024 * 1024)
    monkeypatch.setattr(demo, "MAX_ROWS", 2)
    with pytest.raises(demo.SpaceInputError, match="2-row"):
        demo.read_upload(upload)


def test_evaluate_upload_rejects_invalid_column_contract(upload: Path) -> None:
    with pytest.raises(demo.SpaceInputError, match="must be different"):
        demo.evaluate_upload(
            upload,
            task="claim_frequency",
            observed_column="y_true",
            predicted_column="y_true",
        )
    with pytest.raises(demo.SpaceInputError, match="rate input scale"):
        demo.evaluate_upload(
            upload,
            task="claim_frequency",
            observed_column="y_true",
            predicted_column="y_pred",
            exposure_column="exposure",
            input_scale="aggregate",
        )


def test_evaluate_upload_rejects_non_numeric_selection(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("actual,prediction\none,0.2\ntwo,0.3\n", encoding="utf-8")

    with pytest.raises(demo.SpaceInputError, match="only numbers"):
        demo.evaluate_upload(
            path,
            task="claim_frequency",
            observed_column="actual",
            predicted_column="prediction",
        )
