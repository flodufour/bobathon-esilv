"""Cross-validation strategy for the dummy baseline.

The first two experiments (01_dummy, 02_ridge) use a random 80/20
row holdout (``splitter=0.2``).  This deliberately does *not* group
by patient — the resulting RMSE is optimistic because the same
patient can appear on both sides of the split.  That leak is the
very point: it establishes the row-holdout floor before the
patient-grouped CV is introduced.
"""

from __future__ import annotations

# 80/20 random train-test split — default for row-holdout experiments.
splitter: float = 0.2
