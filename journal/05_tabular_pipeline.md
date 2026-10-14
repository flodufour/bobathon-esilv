# 05_tabular_pipeline

## Question / hypothesis

Does including the categorical columns (`gene`, `cohort`) via `skrub.tabular_pipeline` lower the grouped-CV RMSE compared with the numeric-only HGBR from 04?

## Motivation

- **Sourcing strategy:** user (lab guide DAY2 § 5)
- **Source(s):**
  - `docs/DAY2.md` § 5 — "Ridge and the numeric HGBR above never saw `gene` or `cohort`: those are **strings**. A sklearn regressor cannot multiply `"GBA"` by a weight. Something has to turn text into numbers first."
- **Why this matters:** `gene` (mutation carrier status) and `cohort` (clinical study origin) are clinically meaningful. If these columns carry signal beyond the 8 numeric features, the model should be able to use them — but only if an encoder turns them into numbers first. `skrub.tabular_pipeline("regressor")` auto-selects an encoder per column (one-hot for low-cardinality, StringEncoder for high-cardinality) and pairs it with HGBR, so we get mixed-type coverage in one step.

## Method

- **Files touched:** `src/parkinson/pipeline.py` (add `build_tabular_learner()`), `src/parkinson/data.py` (add `load_full_dataset()` that includes all non-id, non-target columns)
- **Change versus previous experiment (04_hgbr):**
  - Replace `build_hgbr_learner()` (8 numeric features, no imputer) with `skrub.tabular_pipeline("regressor")`, which wraps a `TableVectorizer` + `HistGradientBoostingRegressor`.
  - Feature matrix expands from 8 numeric columns (`FEATURE_COLS`) to all clinical columns except identifiers (`Index`, `patient_id`) and the target. This adds `gene` and `cohort` (strings) and any other non-id columns present.
  - The `TableVectorizer` picks an encoder per column automatically: low-cardinality strings → one-hot, high-cardinality → `StringEncoder`, numerics pass through.
  - Same grouped CV splitter (`GroupKFold-5` on `patient_id`) for a directly comparable result.
- **Cross-validation:** `grouped_splitter` (GroupKFold-5 on `patient_id`) — same as 03 and 04.
- **Out of scope for this experiment:** hyperparameter tuning, custom encoders per column, feature engineering (lags, aggregates).

## Risks / things that could invalidate the result

- `Index` and `patient_id` must be excluded from the feature matrix before fitting — if they leak in, the model can memorise patient ids and inflate the metric.
- `TableVectorizer` may encode `gene` / `cohort` differently depending on cardinality thresholds; the encoding choice should be checked in the fitted pipeline after the run.
- If `gene` or `cohort` have near-zero variance (e.g. one dominant category), the gain from adding them may be negligible or negative (noise added to the feature space).
- The comparison with 04_hgbr is not perfectly controlled: the feature set changes *and* the vectorizer adds an extra stateful step. A gain could come from either.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE 7.5182 ± 0.1413 (GroupKFold-5, patient-grouped)
- **Implication for next iteration:** Adding categorical columns (`gene`, `cohort`, `rater_id`) via `TableVectorizer` reduces RMSE from 7.76 to 7.52 — a modest but consistent gain. The next guide section (DAY2 § 6) proposes the skrub DataOps graph which bakes the grouped CV split directly into the pipeline declaration, removing the manual `groups` juggling.
