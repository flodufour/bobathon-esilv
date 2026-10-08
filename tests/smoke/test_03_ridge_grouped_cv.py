"""Smoke test for ``experiments/03_ridge_grouped_cv.py``.

Fits the Ridge pipeline on a subset of patients and predicts on a
held-out set of patients from the real ``data/`` source.  The hard
assertion checks that every row in the predict set receives exactly one
prediction — catching any pipeline that silently drops rows.  The soft
assertion guards against NaN-poisoned predictions.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset_with_groups
from parkinson.pipeline import build_ridge_learner

DATA_DIR = PROJECT_ROOT / "data"

# Hardcoded from journal/03_ridge_grouped_cv.md § Status.headline.
# Update this constant when the headline result changes.
# No skore import — this test is intentionally portable to any sklearn env.
CV_RMSE_MEAN: float = 10.60  # RMSE 10.60 ± 0.27 (GroupKFold-5, patient-grouped)


@pytest.fixture
def train_predict_split():
    """Return (X_train, y_train, X_predict, y_predict) split by patient.

    Uses ``GroupShuffleSplit`` with ``test_size=0.2`` to hold out ~20 % of
    *patients* (not rows) as the predict slice, mirroring the grouped-CV
    evaluation condition.  No patient appears on both sides of the split.
    """
    X, y, groups = load_dataset_with_groups()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return (
        X.iloc[train_idx],
        y.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[test_idx],
    )


def test_03_ridge_grouped_cv(train_predict_split):
    """Predict-time shape must match the patient-held-out slice row count exactly."""
    X_train, y_train, X_predict, y_predict = train_predict_split
    n_predict_rows = len(X_predict)

    learner = build_ridge_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_predict)

    # HARD: every row in the predict slice must receive exactly one prediction.
    assert len(predictions) == n_predict_rows, (
        f"got {len(predictions)} predictions for {n_predict_rows} rows — "
        "pipeline is dropping rows; check imputer for unexpected filtering."
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
