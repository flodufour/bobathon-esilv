"""Data loading for the Parkinson's motor-score prediction challenge.

Loads ``data/X_train.csv`` and ``data/y_train.csv``, merges them on the
``Index`` column, and returns the feature matrix ``X`` and target ``y``.
``load_dataset_with_groups`` additionally returns the ``patient_id`` column
for use as group labels in patient-grouped cross-validation.
The test set loader returns only the features from ``data/X_test.csv``.

``FEATURE_COLS_HGBT`` extends the base feature set with ``cohort`` and
``gene`` as ``pandas.Categorical`` columns, which
``HistGradientBoostingRegressor`` can use natively (no encoding needed).
Use the ``*_hgbt`` loader variants for experiments that use these features.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from parkinson import PROJECT_ROOT

FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

# Extended feature set for HistGradientBoosting experiments.
# Adds cohort and gene as categorical features; all other columns are
# inherited from FEATURE_COLS.  Do not modify FEATURE_COLS — prior
# experiments depend on its exact shape.
FEATURE_COLS_HGBT = [
    *FEATURE_COLS,
    "cohort",
    "gene",
]

_CATEGORICAL_COLS = ["cohort", "gene"]

_DATA_DIR = PROJECT_ROOT / "data"


def load_dataset(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load and merge the training features and target.

    Parameters
    ----------
    data_dir :
        Directory containing ``X_train.csv`` and ``y_train.csv``.
        Defaults to ``<project_root>/data``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns defined by ``FEATURE_COLS``.
    y : pd.Series
        Target series (``target`` column from ``y_train.csv``).
    """
    data_dir = Path(data_dir) if data_dir is not None else _DATA_DIR
    X_raw = pd.read_csv(data_dir / "X_train.csv")
    y_raw = pd.read_csv(data_dir / "y_train.csv")
    df = X_raw.merge(y_raw, on="Index")
    X = df[FEATURE_COLS]
    y = df["target"]
    return X, y


def load_dataset_with_groups(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load training data and return features, target, and patient groups.

    Identical to ``load_dataset`` but also returns the ``patient_id`` column
    so it can be passed as ``groups`` to a grouped cross-validator such as
    ``GroupKFold``.  The ``patient_id`` column is **not** included in ``X``.

    Parameters
    ----------
    data_dir :
        Directory containing ``X_train.csv`` and ``y_train.csv``.
        Defaults to ``<project_root>/data``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns defined by ``FEATURE_COLS``.
    y : pd.Series
        Target series (``target`` column from ``y_train.csv``).
    groups : pd.Series
        ``patient_id`` column, one entry per row of ``X``, for use as
        group labels in ``GroupKFold`` / ``GroupShuffleSplit``.
    """
    data_dir = Path(data_dir) if data_dir is not None else _DATA_DIR
    X_raw = pd.read_csv(data_dir / "X_train.csv")
    y_raw = pd.read_csv(data_dir / "y_train.csv")
    df = X_raw.merge(y_raw, on="Index")
    X = df[FEATURE_COLS]
    y = df["target"]
    groups = df["patient_id"]
    return X, y, groups


def load_test_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the test features.

    Parameters
    ----------
    data_dir :
        Directory containing ``X_test.csv``.
        Defaults to ``<project_root>/data``.

    Returns
    -------
    pd.DataFrame
        Test feature matrix with an ``Index`` column for Kaggle submission.
    """
    data_dir = Path(data_dir) if data_dir is not None else _DATA_DIR
    return pd.read_csv(data_dir / "X_test.csv")


def _cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast ``cohort`` and ``gene`` columns to ``pandas.Categorical`` dtype.

    Parameters
    ----------
    df :
        DataFrame containing ``cohort`` and ``gene`` string columns.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``cohort`` and ``gene`` cast to ``category``.
    """
    return df.assign(**{col: df[col].astype("category") for col in _CATEGORICAL_COLS})


def load_dataset_hgbt(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load training data with the extended HGBT feature set.

    Returns the 10-column feature matrix (``FEATURE_COLS_HGBT``) with
    ``cohort`` and ``gene`` cast to ``pandas.Categorical`` so that
    ``HistGradientBoostingRegressor(categorical_features="from_dtype")``
    can use them natively without encoding.  No imputation is applied —
    missingness is preserved as signal.

    Parameters
    ----------
    data_dir :
        Directory containing ``X_train.csv`` and ``y_train.csv``.
        Defaults to ``<project_root>/data``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns defined by ``FEATURE_COLS_HGBT``.
    y : pd.Series
        Target series (``target`` column from ``y_train.csv``).
    """
    data_dir = Path(data_dir) if data_dir is not None else _DATA_DIR
    X_raw = pd.read_csv(data_dir / "X_train.csv")
    y_raw = pd.read_csv(data_dir / "y_train.csv")
    df = X_raw.merge(y_raw, on="Index")
    X = _cast_categoricals(df[FEATURE_COLS_HGBT])
    y = df["target"]
    return X, y


def load_dataset_hgbt_with_groups(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load HGBT training data and return features, target, and patient groups.

    Identical to ``load_dataset_hgbt`` but also returns the ``patient_id``
    column for use as group labels in ``GroupKFold``.

    Parameters
    ----------
    data_dir :
        Directory containing ``X_train.csv`` and ``y_train.csv``.
        Defaults to ``<project_root>/data``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns defined by ``FEATURE_COLS_HGBT``.
    y : pd.Series
        Target series (``target`` column from ``y_train.csv``).
    groups : pd.Series
        ``patient_id`` column for use as group labels in ``GroupKFold``.
    """
    data_dir = Path(data_dir) if data_dir is not None else _DATA_DIR
    X_raw = pd.read_csv(data_dir / "X_train.csv")
    y_raw = pd.read_csv(data_dir / "y_train.csv")
    df = X_raw.merge(y_raw, on="Index")
    X = _cast_categoricals(df[FEATURE_COLS_HGBT])
    y = df["target"]
    groups = df["patient_id"]
    return X, y, groups


def load_test_dataset_hgbt(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the test features for HGBT experiments.

    Returns the test set with ``cohort`` and ``gene`` cast to
    ``pandas.Categorical``, matching the dtype expected by a model fitted
    with ``load_dataset_hgbt``.

    Parameters
    ----------
    data_dir :
        Directory containing ``X_test.csv``.
        Defaults to ``<project_root>/data``.

    Returns
    -------
    pd.DataFrame
        Test feature matrix with ``FEATURE_COLS_HGBT`` columns plus
        ``Index`` for Kaggle submission.
    """
    data_dir = Path(data_dir) if data_dir is not None else _DATA_DIR
    df = pd.read_csv(data_dir / "X_test.csv")
    for col in _CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df
