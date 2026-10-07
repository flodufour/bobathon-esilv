# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog.
-->

## Status

- **Project / dataset:** Parkinson's disease MDS-UPDRS motor score — tabular regression
- **Goal:** minimize RMSE for the true OFF MDS-UPDRS motor score, on patients held out from training
- **Last experiment:** 01_dummy — done
- **Last result:** RMSE 16.4779 (80/20 row holdout)

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-07-14
  - env manager: pip+venv - recorded: 2025-07-14
  - agent feature: not installed - recorded: 2025-07-14
  - optional features: none - recorded: 2025-07-14
  - package name (`src/<pkg>/`): parkinson - recorded: 2025-07-14
  - skore mode: hub - recorded: 2025-07-14
  - skore hub workspace: croustibm - recorded: 2025-07-14
  - skore mlflow tracking uri: n/a - recorded: 2025-07-14
  - student prior: some-sklearn - recorded: 2025-07-14
  - CV splitter family: float (0.2 row holdout) for 01_dummy and 02_ridge - recorded: 2025-07-14

## Data understanding (EDA)

- **Status:** skipped - 2025-07-14
- **Summary:** n/a
- **Report:** [data/eda.md](../data/eda.md)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor(strategy="mean") as the RMSE floor on an 80/20 row holdout | done | RMSE 16.4779 (80/20 row holdout) | [design note](01_dummy.md) |

## Backlog

| # | Item | Source |
|---|---|---|
