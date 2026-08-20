# Interactive demo

The versioned Hugging Face Space bundle lives in `space/`. It provides a
browser workflow for evaluating one holdout-prediction CSV without installing
Python or writing code.

## Workflow

1. Upload one UTF-8 `.csv` file, or keep the bundled frequency example.
2. Select claim frequency, claim severity, or pure premium.
3. Map observed, predicted, optional exposure, and optional sample-weight
   columns.
4. Choose whether values are aggregates or rates.
5. Evaluate the portfolio and inspect the metric table and calibration bands.
6. Download the standalone HTML evidence report.

Task defaults deliberately retain separate accuracy, calibration, and
discrimination objectives. The demo does not combine them into a universal
score.

## Data and resource boundaries

- one uncompressed UTF-8 CSV per evaluation;
- maximum file size of 5 MiB;
- maximum shape of 50,000 rows and 100 columns;
- maximum parsed DataFrame size of 64 MiB;
- numeric validation for every selected evaluation column;
- a two-request evaluation concurrency limit and a 16-request queue;
- no database, model training, or outbound data request;
- downloaded reports contain aggregate results, runtime metadata, and an
  input fingerprint, not the uploaded rows.

Hugging Face and Gradio manage temporary upload files as part of the hosted
runtime. Do not upload production policyholder data to a public demonstration
service; use synthetic or appropriately anonymized holdout data.

## Run locally

From the repository root:

```bash
python -m pip install "gradio==6.25.0"
python -m pip install -e .
cd space
python app.py
```

Open `http://127.0.0.1:7860/`. The same pinned Gradio release is declared in
the Space metadata.

## Deployment

The manual `Hugging Face Space` GitHub Actions workflow validates the bundle,
creates or reuses a public Gradio Space, and uploads `space/` as one Hub
commit. Configure a write-scoped `HF_TOKEN` secret in the protected
`hugging-face-space` GitHub environment before running it. Enter the target as
`owner/name`; the workflow defaults to `aminemanai2003/acteval`.

Keep the token in GitHub's encrypted environment secret store. Never commit it,
place it in workflow input, or paste it into an issue or chat.
