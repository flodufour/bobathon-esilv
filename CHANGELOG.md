# Changelog

Newest entry at the top. One entry per logical commit.
Format: `## YYYY-MM-DD — <title>` with `### Features` and `### Fixes` sections.

---

## 2025-07-14 — Workspace scaffold + dummy baseline

### Features
- Scaffolded `src/parkinson/` package (`__init__.py`, `hub.py`, `data.py`, `pipeline.py`, `evaluate.py`)
- Added `pyproject.toml` declaring the package as editable-installable
- Added `journal/JOURNAL.md` project index and `journal/01_dummy.md` design note
- Added `experiments/01_dummy.py` — DummyRegressor baseline, RMSE 16.4779 on 80/20 row holdout
- Pushed `01_dummy` report to Skore Hub (project `bobathon-esilv`, workspace `croustibm`)
- Wrote `submissions/01_dummy.csv` (11 013 rows)
- Extended `.bob/rules/AGENTS.md` with reply-style, skills, code conventions, and changelog rules
