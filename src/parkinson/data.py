"""Data loading for the Parkinson's motor-score prediction challenge.

Loads ``data/X_train.csv`` and ``data/y_train.csv``, merges them on the
``Index`` column, and returns the feature matrix ``X`` and target ``y``.
``load_dataset_with_groups`` additionally returns the ``patient_id`` column
for use as group labels in patient-grouped cross-validation.
``load_full_dataset`` returns all clinical columns (numeric + categorical)
excluding identifiers, for use with mixed-type learners such as
``skrub.tabular_pipeline``.
``load_visits`` returns the raw merged DataFrame (all columns including
``patient_id`` and ``target``) for use as the DataOps graph source.
The test set loader returns only the features from ``data/X_test.csv``.
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

# Columns to drop before fitting a mixed-type learner.
# Index is the Kaggle row id; patient_id is the group key (not a feature).
_ID_COLS = ["Index", "patient_id"]

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


def load_full_dataset(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load training data with all clinical columns and patient groups.

    Returns all columns except the Kaggle identifier (``Index``) and the
    group key (``patient_id``).  This includes the numeric columns in
    ``FEATURE_COLS`` plus categorical columns such as ``cohort``, ``gene``,
    and ``rater_id``, making the feature matrix suitable for mixed-type
    learners like ``skrub.tabular_pipeline``.

    Parameters
    ----------
    data_dir :
        Directory containing ``X_train.csv`` and ``y_train.csv``.
        Defaults to ``<project_root>/data``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with all clinical columns (numeric + categorical),
        excluding ``Index`` and ``patient_id``.
    y : pd.Series
        Target series (``target`` column from ``y_train.csv``).
    groups : pd.Series
        ``patient_id`` column for use as group labels in ``GroupKFold``.
    """
    data_dir = Path(data_dir) if data_dir is not None else _DATA_DIR
    X_raw = pd.read_csv(data_dir / "X_train.csv")
    y_raw = pd.read_csv(data_dir / "y_train.csv")
    df = X_raw.merge(y_raw, on="Index")
    groups = df["patient_id"]
    X = df.drop(columns=_ID_COLS + ["target"])
    y = df["target"]
    return X, y, groups


def load_visits(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the raw merged training DataFrame for use as a DataOps graph source.

    Returns the full merged frame (all columns, including ``patient_id``,
    ``Index``, and ``target``).  The DataOps graph is responsible for
    dropping identifiers and splitting features from the target.

    Parameters
    ----------
    data_dir :
        Directory containing ``X_train.csv`` and ``y_train.csv``.
        Defaults to ``<project_root>/data``.

    Returns
    -------
    pd.DataFrame
        Merged frame with all columns from ``X_train.csv`` plus ``target``.
    """
    data_dir = Path(data_dir) if data_dir is not None else _DATA_DIR
    X_raw = pd.read_csv(data_dir / "X_train.csv")
    y_raw = pd.read_csv(data_dir / "y_train.csv")
    return X_raw.merge(y_raw, on="Index")

