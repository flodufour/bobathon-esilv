# 04_hgbt

<!--
Design note for experiments/04_hgbt.py.
-->

## Question / hypothesis

Does a `HistGradientBoostingRegressor` beat the Ridge RMSE of 10.60 by
exploiting non-linear interactions and treating missingness as signal
(including two new categorical features: `cohort` and `gene`)?

## Motivation

- **Sourcing strategy:** lab guide — `docs/DAY2.md` § 4
- **Source(s):**
  - `docs/DAY2.md` § 4 "HistGradientBoosting: missingness as signal":
    "ON/OFF/LEDD/genetics are often missing; missingness is part of the
    generative process. Ridge had to *fill* those holes with a median.
    Here we keep the holes."
  - Guide note: "Convert `cohort` / `gene` with `.astype('category')` if
    you add them to X."
  - `docs/CONTEXT.md`: "a missing OFF often means the visit was ON-only
    (uncomfortable OFF exams are skipped) — that is information, not
    noise to impute away."
- **Why this matters:** Ridge is the best linear model on 8 numeric
  features. HGBT can (a) use NaN as a dedicated split direction —
  "was OFF measured?" is a signal; (b) capture non-linear interactions
  (e.g. age × LEDD); (c) use `cohort` and `gene` as categorical
  features without any encoding. If RMSE drops materially below 10.60,
  the residual Ridge error was due to linearity and imputation.

## Method

- **Files touched:** `src/parkinson/data.py`, `src/parkinson/pipeline.py`,
  `experiments/04_hgbt.py`
- **Change versus 03_ridge_grouped_cv:**
  - Feature set expanded from 8 to 10 columns: add `cohort` and `gene`
    as `pandas.Categorical` dtype (HGBT treats them natively via
    `categorical_features="from_dtype"`).
  - New constant `FEATURE_COLS_HGBT` in `data.py`; `FEATURE_COLS`
    untouched so prior experiments are unaffected.
  - New loader `load_dataset_hgbt()` and `load_dataset_hgbt_with_groups()`
    that apply `.astype("category")` to `cohort` and `gene`.
  - New `build_hgbt_learner()` in `pipeline.py`:
    `HistGradientBoostingRegressor(random_state=0)`. No imputer.
  - Same `GroupKFold(n_splits=5)` on `patient_id`.
- **Kaggle submission:** refit on all training rows, write
  `submissions/04_hgbt.csv`. Hub report key: `04_hgbt`.
- **Out of scope:** hyperparameter tuning, feature scaling — default
  HGBT first to establish the non-linear baseline with categorical
  features.

## Risks / things that could invalidate the result

- Default `max_iter=100` may underfit on 44k rows. If RMSE is
  disappointing, increasing `max_iter` to 300–500 is the first knob.
- `categorical_features="from_dtype"` requires the columns to be
  `pandas.Categorical` at fit AND predict time; the loader must
  cast them consistently or HGBT will raise at predict.
- Comparison to `03_ridge_grouped_cv` is valid (same GroupKFold-5
  splitter, same data). Do not compare to `02_ridge` row-holdout.
- Do NOT add a `SimpleImputer` before HGBT — that would hide the
  missingness signal the guide specifically asks to preserve.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE 7.75 ± 0.15 (GroupKFold-5, patient-grouped)
- **Implication for next iteration:** HGBT cuts Ridge RMSE from 10.60 to 7.75 (−27%). Non-linearity and missingness-as-signal are the main drivers. Next levers: (1) tune max_iter (default 100 likely underfits on 44k rows — try 300–500); (2) add more features (rater_id, time-based features, visit number); (3) compare permutation importances to understand what the model learned.
