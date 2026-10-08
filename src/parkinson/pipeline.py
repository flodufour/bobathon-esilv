"""Pipeline builders for the Parkinson's motor-score prediction challenge.

``build_learner`` returns the dummy mean baseline (01_dummy).
``build_ridge_learner`` returns a median-imputer + Ridge pipeline (02_ridge).
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def build_learner() -> DummyRegressor:
    """Return the baseline dummy regressor.

    Returns
    -------
    DummyRegressor
        Unfitted estimator that predicts the training mean for every row.
    """
    return DummyRegressor(strategy="mean")


def build_ridge_learner() -> Pipeline:
    """Return a median-imputing Ridge regression pipeline.

    Median imputation handles the missing values present in several
    feature columns (e.g. ``ledd``, ``time_since_intake_off``).
    Ridge regularisation (default ``alpha=1.0``) prevents coefficient
    blow-up on the unscaled feature columns.

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline with steps ``imputer`` and ``ridge``.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("ridge", Ridge()),
        ]
    )
