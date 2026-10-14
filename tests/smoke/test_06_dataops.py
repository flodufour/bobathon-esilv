"""Smoke test for ``experiments/06_dataops.py``.

Fits the DataOps SkrubLearner on a patient-grouped train split and predicts on
a held-out patient slice.  The hard assertion checks row-count preservation;
the soft assertion guards against degenerate NaN predictions.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from parkinson import PROJECT_ROOT
from parkinson.data import load_visits
from parkinson.pipeline import build_dataops_learner

DATA_DIR = PROJECT_ROOT / "data"

CV_RMSE_MEAN: float | None = None


@pytest.fixture
def train_predict_split():
    """Return train/predict env-dicts split by patient."""
    visits = load_visits()
    groups = visits["patient_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(gss.split(visits, groups=groups))
    return (
        visits.iloc[train_idx].reset_index(drop=True),
        visits.iloc[test_idx].reset_index(drop=True),
    )


def test_06_dataops(train_predict_split):
    """Predict-time shape must match the patient-held-out slice row count exactly."""
    visits_train, visits_predict = train_predict_split
    n_predict_rows = len(visits_predict)
    y_predict = visits_predict["target"].to_numpy()

    learner = build_dataops_learner()
    learner.fit({"visits": visits_train})
    predictions = learner.predict({"visits": visits_predict})

    assert len(predictions) == n_predict_rows, (
        f"got {len(predictions)} predictions for {n_predict_rows} rows — "
        "DataOps pipeline is dropping rows at predict time."
    )

    assert not np.isnan(predictions).any(), (
        "predictions contain NaN — DataOps learner produced degenerate output."
    )

    if CV_RMSE_MEAN is not None:
        smoke_mae = mean_absolute_error(y_predict, predictions)
        assert smoke_mae < 3 * CV_RMSE_MEAN, (
            f"smoke MAE {smoke_mae:.2f} > 3 × CV RMSE mean ({CV_RMSE_MEAN:.2f})."
        )
