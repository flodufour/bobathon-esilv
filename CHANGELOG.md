# Changelog

Newest entry at the top. One entry per logical commit.
Format: `## YYYY-MM-DD — <title>` with `### Features` and `### Fixes` sections.

---

## 2026-10-07 — 03_ridge_grouped_cv: patient-grouped CV experiment

### Features
- Added `load_dataset_with_groups()` to `src/parkinson/data.py` — returns `(X, y, groups)` with `patient_id` for grouped splitters
- Added `grouped_splitter` (`GroupKFold(n_splits=5)`) to `src/parkinson/evaluate.py`
- Added `experiments/03_ridge_grouped_cv.py` — Ridge with patient-grouped CV, hub key `03_ridge_grouped_cv`, writes `submissions/03_ridge_grouped_cv.csv`
- Added `tests/smoke/test_03_ridge_grouped_cv.py` — patient-split row-count and NaN-safety assertions
- Added `journal/03_ridge_grouped_cv.md` — done, RMSE 10.60 ± 0.27 (GroupKFold-5)

---

## 2026-10-07 — 02_ridge: Ridge regression experiment

### Features
- Added `build_ridge_learner()` to `src/parkinson/pipeline.py` — median-imputing `SimpleImputer` + `Ridge` pipeline
- Added `experiments/02_ridge.py` — evaluates Ridge on 80/20 holdout, pushes `02_ridge` report to Skore Hub, writes `submissions/02_ridge.csv`
- Added `tests/smoke/test_02_ridge.py` — row-count and NaN-safety assertions on a held-out split
- Added `journal/02_ridge.md` — approved design note for the Ridge experiment
- Added `ruff.toml` at project root (numpydoc + per-file ignores for `experiments/` and `audit/`)

---

## 2026-10-07 — Workspace scaffold + dummy baseline

### Features
- Scaffolded `src/parkinson/` package (`__init__.py`, `hub.py`, `data.py`, `pipeline.py`, `evaluate.py`)
- Added `pyproject.toml` declaring the package as editable-installable
- Added `journal/JOURNAL.md` project index and `journal/01_dummy.md` design note
- Added `experiments/01_dummy.py` — DummyRegressor baseline, RMSE 16.4779 on 80/20 row holdout
- Pushed `01_dummy` report to Skore Hub (project `bobathon-esilv`, workspace `croustibm`)
- Wrote `submissions/01_dummy.csv` (11 013 rows)
- Extended `.bob/rules/AGENTS.md` with reply-style, skills, code conventions, and changelog rules
