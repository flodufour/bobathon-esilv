# 06_dataops

## Question / hypothesis

Does baking the patient-grouped CV split directly into the skrub DataOps graph (via `mark_as_X(split_kwargs={"groups": ...})`) produce the same RMSE as 05_tabular_pipeline while eliminating the manual `groups` juggling?

## Motivation

- **Sourcing strategy:** user (lab guide DAY2 § 6)
- **Source(s):**
  - `docs/DAY2.md` § 6 — "DataOps is skrub's way to write the *recipe* once, as a graph, so features, target, and the grouped split live on the same object … Putting that on the graph is how you stop the leakage from step 3 from creeping back in through a forgotten argument."
- **Why this matters:** In experiments 03–05 we precomputed `groups` as a side variable and passed it manually to `grouped_splitter.split(X, y, groups)`. The DataOps graph attaches `patient_id` at the `mark_as_X` node, so `skore.evaluate` can read it automatically. This removes the risk of misaligning `groups` with `X` after a copy-paste and is the canonical pattern for this stack going forward.

## Method

- **Files touched:** `src/parkinson/pipeline.py` (add `build_dataops_learner()` that returns a `SkrubLearner`), `src/parkinson/data.py` (add `load_visits()` that returns the raw merged DataFrame for the DataOps graph)
- **Change versus previous experiment (05_tabular_pipeline):**
  - Instead of a bare `sklearn.Pipeline`, build a skrub DataOps graph rooted at `skrub.var("visits", visits)`.
  - Drop `target`, mark the feature node with `.skb.mark_as_X(split_kwargs={"groups": data["patient_id"]})` and mark `target` with `.skb.mark_as_y()`.
  - Attach `TableVectorizer` + `HistGradientBoostingRegressor` via `.skb.apply(...)`.
  - Freeze the graph into a `SkrubLearner` via `.skb.make_learner()`.
  - `skore.evaluate` is called with `data={"visits": visits}` (env-dict form) and `splitter=grouped_splitter` — the `groups` are read from `split_kwargs` at the marker.
  - The Kaggle submission uses `learner.fit({"visits": visits})` and `learner.predict({"visits": X_test})`.
- **Cross-validation:** `grouped_splitter` (GroupKFold-5 on `patient_id`) — same family as 03–05, now wired into the graph.
- **Out of scope for this experiment:** hyperparameter tuning, additional feature engineering, new estimator families.

## Risks / things that could invalidate the result

- If `skore.evaluate` with a `SkrubLearner` + `data=` form does not pass `groups` from `split_kwargs` to the splitter automatically, the CV will silently fall back to row-level splitting — producing an optimistic estimate matching 02_ridge rather than 05_tabular_pipeline.
- The `SkrubLearner.predict` env-dict must contain exactly the same `skrub.var` name used at build time (`"visits"`); any mismatch raises a `KeyError` at predict time.
- `target` must be dropped from the feature matrix before `mark_as_X`; if it leaks in, RMSE will collapse to near zero.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE 7.5389 ± 0.1436 (GroupKFold-5, patient-grouped, DataOps)
- **Implication for next iteration:** RMSE quasi identique à 05_tabular_pipeline (7.52 vs 7.54 — dans l'intervalle de confiance), confirmant que le DataOps ne change pas le modèle mais sécurise le wiring. C'est le pattern canonique à utiliser pour toutes les expériences suivantes. La prochaine étape est de soumettre la meilleure submission sur Kaggle et d'explorer d'autres axes (feature engineering, tuning, modèles plus puissants).
