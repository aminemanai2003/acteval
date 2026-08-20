---
title: ActEval Workbench
emoji: 🧮
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.25.0
python_version: 3.13
app_file: app.py
fullWidth: true
header: mini
short_description: Evaluate actuarial prediction CSVs across distinct objectives.
license: apache-2.0
---

# ActEval Workbench

Interactive demonstration of
[ActEval](https://github.com/aminemanai2003/acteval), a model-agnostic
evaluation library for non-life insurance predictions.

The Space accepts one UTF-8 CSV up to 5 MiB, 50,000 rows, and 100 columns. It
calculates the default metrics for claim frequency, claim severity, or pure
premium and produces a standalone HTML evidence report. Uploaded rows are not
included in the report and are not sent to a training service.

Metrics remain separate actuarial objectives. The demo does not calculate a
universal model score or make a production deployment decision.
