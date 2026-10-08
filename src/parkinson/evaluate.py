"""Cross-validation strategies for the Parkinson's motor-score experiments.

``splitter`` (float 0.2) is the random 80/20 row holdout used by
``01_dummy`` and ``02_ridge``.  It deliberately allows the same patient on
both sides of the split — an optimistic but comparable floor.

``grouped_splitter`` is a ``GroupKFold(n_splits=5)`` for ``03_ridge_grouped_cv``
and later experiments.  Pass ``patient_id`` as ``groups`` to
``skore.evaluate`` so no patient appears in both train and test within any
fold, matching the Kaggle evaluation condition.
"""

from __future__ import annotations

from sklearn.model_selection import GroupKFold

# 80/20 random row holdout — used by 01_dummy and 02_ridge.
splitter: float = 0.2

# Patient-grouped 5-fold CV — used from 03_ridge_grouped_cv onwards.
# Pass groups=patient_id to skore.evaluate so visits from the same patient
# stay on the same side of every fold boundary.
grouped_splitter: GroupKFold = GroupKFold(n_splits=5)
