# 07_dataops_tuned

## Question / hypothesis

Does tuning `max_iter` and `max_leaf_nodes` on the `HistGradientBoostingRegressor` inside the DataOps graph lower the grouped-CV RMSE below 7.52?

## Motivation

- **Sourcing strategy:** my-pick
- **Source(s):**
  - 06_dataops implication: "La prochaine étape est d'explorer d'autres axes (feature engineering, tuning, modèles plus puissants)."
  - The default HGBR uses `max_iter=100` and `max_leaf_nodes=31`. For a dataset of ~44k rows with 11 features, more iterations and/or shallower trees often improve generalisation before hitting diminishing returns.
- **Why this matters:** We have never tuned any hyperparameter. A simple grid over `max_iter ∈ {200, 500}` and `max_leaf_nodes ∈ {15, 31, 63}` costs at most 6 × CV time and may yield a meaningful RMSE drop. We use the DataOps pattern established in 06.

## Method

- **Files touched:** `src/parkinson/pipeline.py` (add `build_dataops_tuned_learner(max_iter, max_leaf_nodes)`)
- **Change versus previous experiment (06_dataops):**
  - Same DataOps graph as 06 (TableVectorizer + HGBR, GroupKFold-5 baked in).
  - `HistGradientBoostingRegressor` parameterised with `max_iter` and `max_leaf_nodes` passed explicitly.
  - The experiment script runs a manual grid (no `GridSearchCV`) — evaluate each combination separately, print RMSE, pick the best, push that single report to Skore Hub, write the Kaggle CSV.
- **Cross-validation:** GroupKFold-5 baked at `mark_as_X` — same as 06.
- **Out of scope for this experiment:** learning rate tuning, `l2_regularization`, `min_samples_leaf`, any automated search.

## Risks / things that could invalidate the result

- Trying 6 combinations on the same data and picking the best is a form of hyperparameter leakage over the CV estimate — the reported RMSE for the best combination is slightly optimistic. A held-out refit on Kaggle is the honest check.
- `max_iter=500` roughly quintuples training time; on ~44k rows this is still tractable but slow.

## Status

- **State:** approved
- **Approved by user on:** 2026-10-07
- **Headline result:** n/a
- **Implication for next iteration:** n/a
