"""Dummy mean-prediction pipeline — the baseline for this challenge.

The ``DummyRegressor(strategy="mean")`` always predicts the training
mean, ignoring all features.  Its RMSE sets the floor: any model
that learns something from the features must beat this number.
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor


def build_learner() -> DummyRegressor:
    """Return the baseline dummy regressor.

    Returns
    -------
    DummyRegressor
        Unfitted estimator that predicts the training mean for every row.
    """
    return DummyRegressor(strategy="mean")
