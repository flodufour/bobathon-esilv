# 01_dummy

<!--
Design note for experiments/01_dummy.py.
-->

## Question / hypothesis

Does predicting the training-mean true-OFF score for every visit
(ignoring all features) give a sensible RMSE floor for this dataset?

## Motivation

- **Sourcing strategy:** bootstrap — forced baseline
- **Source(s):**
  - User request: "Run a dummy regressor from sklearn (DummyRegressor,
    strategy='mean') to act as my baseline model."
- **Why this matters:** The mean-prediction RMSE is the simplest
  possible benchmark. Any model that learns from the features must
  beat this number. It also anchors the scale of the target and
  confirms the data-loading pipeline is working end-to-end.

## Method

- **Files touched:** `src/parkinson/data.py`, `src/parkinson/pipeline.py`,
  `src/parkinson/evaluate.py`, `experiments/01_dummy.py`
- **Change versus baseline:** This is the baseline. Features:
  `sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`,
  `time_since_intake_off`, `on`, `off`. Estimator: `DummyRegressor(strategy="mean")`.
- **Cross-validation:** 80/20 random row holdout (`splitter=0.2`).
  The same patient can appear on both sides of the split —
  the grouped-CV experiment will fix that later.
- **Out of scope for this experiment:** feature engineering, imputation,
  any learned transformation.

## Risks / things that could invalidate the result

- The 80/20 holdout leaks patient information (same patient on both
  sides), so the RMSE is likely lower than what a proper patient-grouped
  CV would give. The number is a row-level floor, not a generalisation
  estimate.
- Missing values in the feature columns are silently passed to
  `DummyRegressor`, which ignores them — this is harmless for the dummy
  but will need handling for real models.

## Status

- **State:** done
- **Approved by user on:** 2025-07-14
- **Headline result:** RMSE 16.4779 (80/20 row holdout)
- **Implication for next iteration:** The mean baseline predicts ~16.5 points off on the MDS-UPDRS scale. Any model using the features should substantially beat this. A Ridge regression on the same features (02_ridge) is the natural next step to see how much linear structure is present.
