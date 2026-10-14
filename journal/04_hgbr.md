# 04_hgbr

## Question / hypothesis

Does a `HistGradientBoostingRegressor` — which natively handles missing values instead of median-imputing them — lower the grouped-CV RMSE compared with Ridge?

## Motivation

- **Sourcing strategy:** user (lab guide DAY2 § 4)
- **Source(s):**
  - `docs/DAY2.md` § 4 — "ON/OFF/LEDD/genetics are often missing; missingness is part of the generative process. Ridge had to *fill* those holes with a median. Here we keep the holes."
- **Why this matters:** In the Parkinson dataset a missing `off` value often means the visit was ON-only (uncomfortable OFF exams are skipped). That absence is clinical signal, not random noise. Imputing it away discards that signal; HGBR can learn to route missing rows differently at each tree split, effectively keeping the information.

## Method

- **Files touched:** `src/parkinson/pipeline.py` (add `build_hgbr_learner()`), `src/parkinson/evaluate.py` (no change — `grouped_splitter` already defined)
- **Change versus previous experiment (03_ridge_grouped_cv):**
  - Replace `SimpleImputer(strategy="median") + Ridge` with `HistGradientBoostingRegressor(random_state=0)` — no imputer, NaNs are passed through directly.
  - Same 8 numeric feature columns as before (`FEATURE_COLS`).
  - Same grouped CV splitter (`grouped_splitter`, GroupKFold n=5 on `patient_id`).
- **Cross-validation:** `grouped_splitter` (GroupKFold-5 on `patient_id`) — same as 03, for a directly comparable result.
- **Out of scope for this experiment:** adding categorical features (`gene`, `cohort`), hyperparameter tuning, feature engineering.

## Risks / things that could invalidate the result

- HGBR has more implicit capacity than Ridge; any apparent gain could partly come from overfitting to the training folds rather than from the missingness signal alone.
- We still use only the 8 numeric features from `FEATURE_COLS` — categorical columns (`gene`, `cohort`) are excluded, so the comparison with Ridge is clean but does not yet exploit all available information.
- `random_state=0` makes the run repeatable but does not guarantee stability across very different random seeds; a single result should be treated as indicative.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE 7.7556 ± 0.1528 (GroupKFold-5, patient-grouped)
- **Implication for next iteration:** HGBR drops RMSE from 10.60 to 7.76 simply by keeping missing values as signal instead of imputing them. The next logical step is to add the categorical features (`gene`, `cohort`) that were excluded here — `skrub.tabular_pipeline` or `TableVectorizer + HGBR` handles mixed-type columns without manual encoding.
