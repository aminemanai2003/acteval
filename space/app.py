"""Gradio interface for the public ActEval demonstration Space."""

from __future__ import annotations

import logging
from html import escape
from pathlib import Path

import gradio as gr
import pandas as pd
from acteval_space import (
    MAX_FILE_SIZE_BYTES,
    SpaceInputError,
    evaluate_upload,
    profile_upload,
    suggested_column,
)

from acteval.exceptions import ActEvalError

LOGGER = logging.getLogger("acteval.space")
ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = ROOT / "sample_predictions.csv"
STYLE_PATH = ROOT / "styles.css"

EMPTY_METRICS = pd.DataFrame(columns=["Metric", "Value", "Role", "Interpretation"])
EMPTY_CALIBRATION = None


def _status(kind: str, title: str, detail: str) -> str:
    return (
        f'<div class="status {escape(kind)}" role="status">'
        f"<strong>{escape(title)}</strong><span>{escape(detail)}</span></div>"
    )


def inspect_upload(upload: str | None) -> tuple[object, ...]:
    """Populate safe column choices after an upload changes."""

    try:
        profile = profile_upload(upload)
    except SpaceInputError as error:
        empty = gr.Dropdown(choices=[], value=None, interactive=False)
        return (
            _status("error", "CSV not ready", str(error)),
            pd.DataFrame(),
            empty,
            empty,
            empty,
            empty,
            gr.Dropdown(
                choices=[("Aggregate values", "aggregate"), ("Rates", "rate")],
                value="aggregate",
            ),
        )

    columns = list(profile.columns)
    observed = suggested_column(profile.columns, "observed")
    predicted = suggested_column(profile.columns, "predicted")
    exposure = suggested_column(profile.columns, "exposure")
    sample_weight = suggested_column(profile.columns, "sample_weight")
    scale = "rate" if exposure is not None else "aggregate"
    return (
        _status(
            "ready",
            "CSV ready",
            (
                f"{profile.rows:,} rows and {len(columns)} columns passed "
                "the upload limits."
            ),
        ),
        profile.preview,
        gr.Dropdown(choices=columns, value=observed, interactive=True),
        gr.Dropdown(choices=columns, value=predicted, interactive=True),
        gr.Dropdown(choices=[("None", None), *columns], value=exposure),
        gr.Dropdown(choices=[("None", None), *columns], value=sample_weight),
        gr.Dropdown(
            choices=[("Aggregate values", "aggregate"), ("Rates", "rate")],
            value=scale,
        ),
    )


def align_scale(exposure_column: str | None) -> gr.Dropdown:
    """Keep the required rate scale visible when exposure is selected."""

    return gr.Dropdown(
        choices=[("Aggregate values", "aggregate"), ("Rates", "rate")],
        value="rate" if exposure_column else "aggregate",
    )


def run_evaluation(
    upload: str | None,
    task: str,
    observed_column: str | None,
    predicted_column: str | None,
    exposure_column: str | None,
    sample_weight_column: str | None,
    input_scale: str,
) -> tuple[object, ...]:
    """Run ActEval while returning safe, actionable failure states."""

    try:
        artifacts = evaluate_upload(
            upload,
            task=task,
            observed_column=observed_column,
            predicted_column=predicted_column,
            exposure_column=exposure_column,
            sample_weight_column=sample_weight_column,
            input_scale=input_scale,
        )
    except (SpaceInputError, ActEvalError) as error:
        return (
            _status("error", "Evaluation blocked", str(error)),
            EMPTY_METRICS,
            EMPTY_CALIBRATION,
            "",
            None,
        )
    except Exception:
        LOGGER.exception("Unexpected Space evaluation failure")
        return (
            _status(
                "error",
                "Evaluation failed",
                "The data could not be evaluated. Review the selections and try again.",
            ),
            EMPTY_METRICS,
            EMPTY_CALIBRATION,
            "",
            None,
        )

    return (
        _status(
            "success",
            "Evaluation complete",
            "Task defaults were calculated; inspect each objective separately.",
        ),
        artifacts.metrics,
        artifacts.calibration,
        artifacts.summary_html,
        str(artifacts.report_path),
    )


with gr.Blocks(
    title="ActEval — actuarial model evaluation",
    analytics_enabled=False,
    delete_cache=(3600, 3600),
) as demo:
    gr.HTML(
        """
        <header class="app-header">
          <div>
            <p class="eyebrow">ACTUARIAL MODEL EVIDENCE</p>
            <h1>ActEval workbench</h1>
          </div>
          <p class="header-note">Compare objectives, not a synthetic overall score.</p>
        </header>
        """
    )
    gr.Markdown(
        "Upload holdout predictions, map the actuarial columns, and calculate "
        "task-specific accuracy, calibration, and discrimination diagnostics. "
        "The bundled sample is loaded initially; uploaded data is not added to "
        "the project or any model-training pipeline."
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320, elem_classes="control-rail"):
            gr.Markdown("### 1 · Portfolio data")
            upload = gr.File(
                value=str(SAMPLE_CSV),
                label="Prediction CSV",
                file_types=[".csv"],
                file_count="single",
                type="filepath",
            )
            dataset_status = gr.HTML()
            preview = gr.Dataframe(
                label="First rows",
                interactive=False,
                max_height=250,
            )

            gr.Markdown("### 2 · Evaluation contract")
            task = gr.Dropdown(
                choices=[
                    ("Claim frequency", "claim_frequency"),
                    ("Claim severity", "claim_severity"),
                    ("Pure premium", "pure_premium"),
                ],
                value="claim_frequency",
                label="Actuarial task",
            )
            with gr.Row():
                observed = gr.Dropdown(label="Observed column", interactive=True)
                predicted = gr.Dropdown(label="Predicted column", interactive=True)
            with gr.Row():
                exposure = gr.Dropdown(label="Exposure column", interactive=True)
                sample_weight = gr.Dropdown(
                    label="Sample-weight column", interactive=True
                )
            input_scale = gr.Dropdown(
                choices=[("Aggregate values", "aggregate"), ("Rates", "rate")],
                value="aggregate",
                label="Input scale",
            )
            gr.Markdown(
                "Exposure is portfolio volume for rate inputs. Sample weights "
                "are multiplied by exposure when both are selected."
            )
            with gr.Row():
                evaluate_button = gr.Button("Evaluate portfolio", variant="primary")
                clear_button = gr.ClearButton(
                    [upload, preview, observed, predicted, exposure, sample_weight],
                    value="Clear",
                )

        with gr.Column(scale=6, min_width=420, elem_classes="result-panel"):
            gr.Markdown("### 3 · Evidence")
            evaluation_status = gr.HTML(
                _status(
                    "idle",
                    "Ready to evaluate",
                    "Confirm the task and column mapping, then run the portfolio.",
                )
            )
            summary = gr.HTML()
            with gr.Tabs():
                with gr.Tab("Metric table"):
                    metrics = gr.Dataframe(
                        value=EMPTY_METRICS,
                        label="Task-default metrics",
                        interactive=False,
                    )
                    gr.Markdown(
                        "Metrics represent different actuarial objectives. "
                        "Direction and target are reported per row."
                    )
                with gr.Tab("Calibration bands"):
                    calibration = gr.LinePlot(
                        value=EMPTY_CALIBRATION,
                        x="Prediction band",
                        y="Mean value",
                        color="Series",
                        color_map={"Observed": "#0891b2", "Predicted": "#d97706"},
                        title="Observed vs predicted by score band",
                        x_title="Prediction band (low to high)",
                        y_title="Weighted mean",
                        tooltip=["Series", "Mean value", "Observations"],
                        height=390,
                    )
                    gr.Markdown(
                        "Bands are ordered by prediction and use exposure/sample "
                        "weights when selected. This is a diagnostic, not a "
                        "statistical calibration test."
                    )
                with gr.Tab("Download"):
                    report = gr.File(
                        label="Standalone HTML evidence report",
                        interactive=False,
                    )
                    gr.Markdown(
                        "The report contains aggregate metrics, evaluation metadata, "
                        "and an input fingerprint—not the uploaded row-level data."
                    )

    upload_outputs = [
        dataset_status,
        preview,
        observed,
        predicted,
        exposure,
        sample_weight,
        input_scale,
    ]
    demo.load(
        inspect_upload,
        inputs=upload,
        outputs=upload_outputs,
        api_name=False,
    )
    upload.change(
        inspect_upload,
        inputs=upload,
        outputs=upload_outputs,
        api_name=False,
    )
    exposure.change(
        align_scale,
        inputs=exposure,
        outputs=input_scale,
        api_name=False,
    )
    evaluate_button.click(
        run_evaluation,
        inputs=[
            upload,
            task,
            observed,
            predicted,
            exposure,
            sample_weight,
            input_scale,
        ],
        outputs=[evaluation_status, metrics, calibration, summary, report],
        concurrency_limit=2,
        api_name=False,
    )

    gr.Markdown(
        "[Documentation](https://aminemanai2003.github.io/acteval/) · "
        "[Source](https://github.com/aminemanai2003/acteval) · "
        "ActEval 2.0.0 · Apache-2.0"
    )

demo.queue(default_concurrency_limit=2, max_size=16)

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(),
        css_paths=STYLE_PATH,
        max_file_size=MAX_FILE_SIZE_BYTES,
        enable_monitoring=False,
        show_error=False,
    )
