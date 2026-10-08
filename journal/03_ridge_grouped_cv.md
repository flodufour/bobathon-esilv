# 03_ridge_grouped_cv

<!--
Design note for experiments/03_ridge_grouped_cv.py (same stem, one to
one with the script).
-->

## Question / hypothesis

Does switching from a random 80/20 row holdout to a patient-grouped
5-fold cross-validation reveal that the row-holdout RMSE (10.49) was
optimistic due to patient-level leakage?

## Motivation

- **Sourcing strategy:** lab guide — prescribed next section after 02_ridge
- **Source(s):**
  - Lab guide (`docs/GUIDED.md`): patient-grouped CV is the guide's
    next section after the row-holdout Ridge.
  - 02_ridge result: RMSE 10.49 on a row holdout that allows the same
    patient to appear on both sides — the split leaks patient-specific
    information (baseline motor score, disease trajectory) into the
    test set, producing an optimistic estimate.
- **Why this matters:** The Kaggle holdout is by `patient_id`, not by
  visit. A model that memorises per-patient offsets will fail on unseen
  patients. `GroupKFold(n_splits=5)` on `patient_id` simulates the
  Kaggle evaluation condition: each fold's test set contains only
  patients the model has never seen. The RMSE from this split is the
  honest generalisation estimate.

## Method

- **Files touched:** `src/parkinson/data.py`, `src/parkinson/evaluate.py`,
  `experiments/03_ridge_grouped_cv.py`
- **Change versus 02_ridge:** same `build_ridge_learner()` pipeline
  (median imputer + Ridge). The only change is the splitter:
  `GroupKFold(n_splits=5)` with `groups=patient_id`, replacing
  `splitter=0.2`. `data.py` gains a `load_dataset_with_groups()`
  function that returns `(X, y, groups)` so the `patient_id` column
  is available for the splitter without polluting the feature matrix.
- **Cross-validation:** `GroupKFold(n_splits=5)` on `patient_id` —
  no patient appears in both train and test within any fold.
- **Kaggle submission:** refit Ridge on all training rows (no CV
  split), write `submissions/03_ridge_grouped_cv.csv`. Hub report
  key: `03_ridge_grouped_cv`.
- **Out of scope:** feature engineering, scaling, tuning `alpha` —
  those come after we have an honest baseline RMSE to compare against.

## Risks / things that could invalidate the result

- `GroupKFold` does not stratify by target distribution. If patient
  subgroups have very different score ranges, some folds may be
  harder than others and fold variance will be high. This is expected
  and honest — it reflects real deployment variance.
- The number of unique patients determines the effective number of
  folds. With 5 folds and ~hundreds of patients, each test fold has
  tens of patients — enough for a stable RMSE estimate.
- The row-holdout RMSE (10.49) and the grouped-CV RMSE are not
  directly comparable in magnitude: different split mechanics produce
  different numbers. The comparison is directional: if grouped CV
  gives a higher RMSE, patient leakage was real and the row-holdout
  number was optimistic.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE 10.60 ± 0.27 (GroupKFold-5, patient-grouped)
- **Implication for next iteration:** Patient-grouped CV gives RMSE 10.60 ± 0.27 — almost identical to the row-holdout 10.49, meaning there was very little patient-level leakage in 02_ridge. The features don't strongly memorise per-patient offsets. The next lever to pull is feature engineering or a more powerful estimator (e.g. HistGradientBoosting) which can exploit non-linear interactions and handle missingness natively.
