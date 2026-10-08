"""Data loading for the Parkinson's motor-score prediction challenge.

Loads ``data/X_train.csv`` and ``data/y_train.csv``, merges them on the
``Index`` column, and returns the feature matrix ``X`` and target ``y``.
``load_dataset_with_groups`` additionally returns the ``patient_id`` column
for use as group labels in patient-grouped cross-validation.
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
