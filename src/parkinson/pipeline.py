"""Pipeline builders for the Parkinson's motor-score prediction challenge.

``build_learner`` returns the dummy mean baseline (01_dummy).
``build_ridge_learner`` returns a median-imputer + Ridge pipeline (02_ridge / 03).
``build_hgbt_learner`` returns a HistGradientBoostingRegressor (04_hgbt+):
    handles NaN natively and accepts pandas Categorical columns for cohort/gene.
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
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


def build_hgbt_learner() -> HistGradientBoostingRegressor:
    """Return a HistGradientBoostingRegressor for the HGBT experiments.

    Handles missing values natively (dedicated NaN bin — no imputer needed)
    and accepts ``pandas.Categorical`` columns for ``cohort`` and ``gene``
    via ``categorical_features="from_dtype"``.  ``random_state=0`` makes
    runs reproducible.

    Returns
    -------
    HistGradientBoostingRegressor
        Unfitted estimator with default ``max_iter=100`` and
        ``categorical_features="from_dtype"``.
    """
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        random_state=0,
    )
