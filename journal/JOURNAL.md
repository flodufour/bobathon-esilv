# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog.
-->

## Status

- **Project / dataset:** Parkinson's disease MDS-UPDRS motor score — tabular regression
- **Goal:** minimize RMSE for the true OFF MDS-UPDRS motor score, on patients held out from training
- **Last experiment:** 03_ridge_grouped_cv — done
- **Last result:** RMSE 10.60 ± 0.27 (GroupKFold-5, patient-grouped)

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2026-10-07
  - env manager: pip+venv - recorded: 2026-10-07
  - agent feature: not installed - recorded: 2026-10-07
  - optional features: none - recorded: 2026-10-07
  - package name (`src/<pkg>/`): parkinson - recorded: 2026-10-07
  - skore mode: hub - recorded: 2026-10-07
  - skore hub workspace: croustibm - recorded: 2026-10-07
  - skore mlflow tracking uri: n/a - recorded: 2026-10-07
  - student prior: some-sklearn - recorded: 2026-10-07
  - CV splitter family: float (0.2 row holdout) for 01_dummy and 02_ridge - recorded: 2026-10-07

## Data understanding (EDA)

- **Status:** skipped - 2026-10-07
- **Summary:** n/a
- **Report:** [data/eda.md](../data/eda.md)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor(strategy="mean") as the RMSE floor on an 80/20 row holdout | done | RMSE 16.4779 (80/20 row holdout) | [design note](01_dummy.md) |
| `02_ridge` | Ridge regression on the same 8 features and 80/20 row holdout — first linear model | done | RMSE 10.49 (80/20 row holdout) | [design note](02_ridge.md) |
| `03_ridge_grouped_cv` | Same Ridge pipeline with GroupKFold(5) on patient_id — honest generalisation estimate | done | RMSE 10.60 ± 0.27 (GroupKFold-5) | [design note](03_ridge_grouped_cv.md) |

## Backlog

| # | Item | Source |
|---|---|---|
