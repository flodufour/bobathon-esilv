"""Smoke test for ``experiments/02_ridge.py``.

Fits the Ridge pipeline on a training split and predicts on a
held-out slice from the real ``data/`` source.  The hard assertion
checks that the prediction count exactly matches the held-out row
count — catching any pipeline that silently drops rows.  The soft
assertion guards against NaN-poisoned predictions.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset
from parkinson.pipeline import build_ridge_learner

DATA_DIR = PROJECT_ROOT / "data"

# Hardcoded from journal/02_ridge.md § Status.headline.
# Update this constant when the headline result changes.
# No skore import — this test is intentionally portable to any sklearn env.
CV_RMSE_MEAN: float = 10.49  # RMSE 10.49 (80/20 row holdout)


@pytest.fixture
def train_predict_split():
    """Return (X_train, y_train, X_predict, y_predict).

    Holds out the last 20 % of rows as the predict slice and trains on
    the rest.  No random shuffle — row order is preserved so the split
    is deterministic and reproducible.
    """
    X, y = load_dataset()
    n = len(X)
    split_idx = int(n * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_predict = X.iloc[split_idx:]
    y_predict = y.iloc[split_idx:]
    return X_train, y_train, X_predict, y_predict


def test_02_ridge(train_predict_split):
    """Predict-time shape must match the held-out slice row count exactly."""
    X_train, y_train, X_predict, y_predict = train_predict_split
    n_predict_rows = len(X_predict)

    learner = build_ridge_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_predict)

    # HARD: every row in the predict slice must receive exactly one prediction.
    assert len(predictions) == n_predict_rows, (
        f"got {len(predictions)} predictions for {n_predict_rows} rows — "
        "pipeline is dropping rows; check imputer and Ridge for unexpected filtering."
    )

    # HARD: predictions must not be NaN.
    assert not np.isnan(predictions).any(), (
        "predictions contain NaN — imputation may have failed for one or more columns."
    )

    # SOFT: if the CV headline is known, check that smoke MAE is not wildly off.
    if CV_RMSE_MEAN is not None:
        smoke_mae = mean_absolute_error(y_predict, predictions)
        assert smoke_mae < 3 * CV_RMSE_MEAN, (
            f"smoke MAE {smoke_mae:.2f} > 3 × CV RMSE mean ({CV_RMSE_MEAN:.2f}) — "
            "predictions may be degenerate; inspect the pipeline output."
        )
