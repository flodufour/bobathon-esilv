"""Pipeline builders for the Parkinson's motor-score prediction challenge.

``build_learner`` returns the dummy mean baseline (01_dummy).
``build_ridge_learner`` returns a median-imputer + Ridge pipeline (02_ridge).
``build_hgbr_learner`` returns a HistGradientBoostingRegressor that handles
missing values natively, without imputation (04_hgbr).
``build_tabular_learner`` returns a skrub tabular_pipeline that encodes mixed
numeric and categorical columns automatically (05_tabular_pipeline).
``build_dataops_learner`` returns a skrub SkrubLearner built from a DataOps
graph with the patient-grouped CV split baked into the ``mark_as_X`` node
(06_dataops).
"""

from __future__ import annotations

import skrub
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
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


def build_hgbr_learner() -> HistGradientBoostingRegressor:
    """Return a HistGradientBoostingRegressor that handles missing values natively.

    Unlike ``build_ridge_learner``, no imputation step is needed: the
    histogram-based gradient boosting algorithm routes missing values to
    the best child at each split, treating their absence as a signal
    rather than noise to fill.

    ``random_state=0`` ensures reproducible fold scores across runs.

    Returns
    -------
    HistGradientBoostingRegressor
        Unfitted estimator ready to be passed to ``skore.evaluate``.
    """
    return HistGradientBoostingRegressor(random_state=0)


def build_tabular_learner() -> Pipeline:
    """Return a skrub tabular_pipeline for mixed numeric and categorical data.

    ``skrub.tabular_pipeline("regressor")`` wraps a ``TableVectorizer``
    (auto-selects an encoder per column: one-hot for low-cardinality strings,
    ``StringEncoder`` for high-cardinality, pass-through for numerics) followed
    by a ``HistGradientBoostingRegressor``.

    This learner accepts a feature matrix that includes categorical columns
    such as ``cohort``, ``gene``, and ``rater_id`` alongside the numeric
    columns in ``FEATURE_COLS``.  Missing values in numeric columns are
    handled natively by the HGBR; missing strings are handled by the
    ``TableVectorizer``.

    Returns
    -------
    skrub.Pipeline
        Unfitted pipeline ready to be passed to ``skore.evaluate``.
    """
    return skrub.tabular_pipeline("regressor")


def build_dataops_learner(visits_preview=None) -> skrub.SkrubLearner:
    """Return a SkrubLearner with patient-grouped CV baked into the graph.

    Builds a skrub DataOps graph rooted at ``skrub.var("visits")``.  The
    graph drops ``Index``, ``patient_id``, and ``target`` from the feature
    matrix, attaches ``patient_id`` as ``split_kwargs`` at ``mark_as_X`` so
    ``skore.evaluate`` can route groups to ``GroupKFold`` automatically, and
    chains ``TableVectorizer`` + ``HistGradientBoostingRegressor``.

    Unlike ``build_tabular_learner``, the caller does not need to maintain a
    separate ``groups`` variable: it is derived inside the graph from the
    ``visits`` DataFrame and stays aligned with ``X`` by construction.

    Parameters
    ----------
    visits_preview :
        Optional DataFrame passed as ``value=`` to ``skrub.var`` for
        interactive ``learner.skb.preview()`` calls.  Has no effect on
        fit or CV.  Pass ``None`` (default) in production.

    Returns
    -------
    skrub.SkrubLearner
        Frozen DataOps learner.  Fit and predict accept an environment
        dict: ``{"visits": <DataFrame>}``.
    """
    visits = (
        skrub.var("visits", value=visits_preview)
        if visits_preview is not None
        else skrub.var("visits")
    )

    groups = visits["patient_id"]
    X = (
        visits
        .drop(columns=["target", "patient_id", "Index"], errors="ignore")
        .skb.mark_as_X(
            cv=GroupKFold(n_splits=5),
            split_kwargs={"groups": groups},
        )
    )
    y = visits["target"].skb.mark_as_y()

    pred = (
        X.skb.apply(skrub.TableVectorizer())
         .skb.apply(HistGradientBoostingRegressor(random_state=0), y=y)
    )
    return pred.skb.make_learner()
