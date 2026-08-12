"""Synthetic example showing accuracy and tail-calibration trade-offs."""

import numpy as np

import acteval as ae

y_true = np.r_[np.tile([0.5, 1.0, 1.5, 1.0, 0.5], 19), np.repeat(10.0, 5)]
model_a = np.r_[y_true[:95] + 0.5, np.repeat(10.0, 5)]
model_b = np.r_[y_true[:95], np.repeat(9.0, 5)]

comparison = ae.compare(
    y_true,
    {"Model A": model_a, "Model B": model_b},
    task="claim_frequency",
    metrics=["rmse", "poisson_deviance", "ae_ratio", "tail_ae_95"],
)

print(comparison.to_dataframe())
print("\nRMSE ranking")
print(comparison.rank("rmse"))
print("\nTail A/E ranking")
print(comparison.rank("tail_ae_95"))
