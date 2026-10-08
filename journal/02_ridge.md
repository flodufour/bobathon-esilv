# 02_ridge

<!--
Design note for experiments/02_ridge.py (same stem, one to
one with the script). Write and agree on it before creating the
script.
-->

## Question / hypothesis

Does a Ridge regression on the 8 clinical features reduce RMSE below
the 16.48 mean-prediction floor on the same 80/20 row holdout?

## Motivation

- **Sourcing strategy:** lab guide — forced next section
- **Source(s):**
  - Lab guide (`docs/GUIDED.md`): "Ridge on the same row holdout" is
    the prescribed step after the dummy baseline to measure how much
    linear structure the features contain.
  - 01_dummy result: RMSE 16.4779 — the floor to beat.
- **Why this matters:** Ridge is the simplest linear learner with
  L2 regularisation. If it meaningfully beats the dummy, the features
  carry real signal. The gap (dummy RMSE − Ridge RMSE) tells us how
  much a linear model gains over pure mean-prediction. The row holdout
  is kept identical to 01_dummy so the two numbers are directly
  comparable.

## Method

- **Files touched:** `src/parkinson/pipeline.py`, `experiments/02_ridge.py`
- **Change versus 01_dummy:** swap `DummyRegressor(strategy="mean")`
  for `Ridge()` (default `alpha=1.0`). The same 8 features
  (`sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`,
  `time_since_intake_off`, `on`, `off`) flow through the pipeline
  unchanged. No feature scaling is added in this step; Ridge with
  unscaled features still fits and the unscaled result is informative.
- **Cross-validation:** 80/20 random row holdout (`splitter=0.2`),
  identical to 01_dummy — no grouped split. The patient-leakage point
  is the same: same patient can appear on both sides.
- **Kaggle submission:** refit Ridge on all training rows, predict
  the test visits, write `submissions/02_ridge.csv` (header
  `Index,target`). Hub report key: `02_ridge`.
- **Out of scope for this experiment:** feature scaling, imputation,
  feature engineering, grouped CV — those are later experiments.

## Risks / things that could invalidate the result

- The 80/20 holdout still leaks patient information (same patient on
  both sides), so RMSE here is not a true generalisation estimate —
  it is only comparable to 01_dummy, not to any grouped-CV number.
- Ridge without feature scaling is sensitive to the relative
  magnitudes of the input columns. If columns like `ledd` dominate
  others by orders of magnitude, the regularisation will under-penalise
  their coefficients. The RMSE may still improve, but the coefficients
  will not be interpretable until scaling is added.
- Missing values in feature columns will cause Ridge to fail (unlike
  DummyRegressor which ignores them). If any feature has NaNs, the run
  will error out — imputation will need to be added to the pipeline.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE 10.49 (80/20 row holdout)
- **Implication for next iteration:** Ridge cuts the dummy floor from 16.48 to 10.49 — a 36 % reduction — confirming the features carry strong linear signal. The next step is either patient-grouped CV to get an honest generalisation estimate, or adding feature scaling (StandardScaler) to make coefficients interpretable and potentially squeeze more out of the regularisation.
