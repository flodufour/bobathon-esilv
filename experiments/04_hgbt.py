# %% [markdown]
# # Experiment 04 — HistGradientBoosting with categorical features
#
# **Date:** 2026-10-07
# **Goal:** Beat Ridge RMSE 10.60 by exploiting non-linear interactions
# and treating missingness as signal.  Adds cohort and gene as categorical
# features; no imputation.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

from parkinson import PROJECT_ROOT
from parkinson.data import (
    load_dataset_hgbt,
    load_dataset_hgbt_with_groups,
    load_test_dataset_hgbt,
)
from parkinson.evaluate import grouped_splitter
from parkinson.hub import load_skore_credentials
from parkinson.pipeline import build_hgbt_learner

# %% [markdown]
# ## Load data and build the learner

# %%
X, y, groups = load_dataset_hgbt_with_groups()
learner = build_hgbt_learner()

# %% [markdown]
# ## Evaluate with patient-grouped 5-fold CV
#
# Same GroupKFold(n_splits=5) on patient_id as 03_ridge_grouped_cv —
# results are directly comparable.  HGBT handles NaN natively so no
# imputer is needed; missing OFF/LEDD/gene values are a signal, not noise.

# %%
report = skore.evaluate(
    learner,
    X=X,
    y=y,
    splitter=grouped_splitter.split(X, y, groups),
)
rmse_df = report.metrics.rmse()
rmse_mean = rmse_df.loc["RMSE", ("HistGradientBoostingRegressor", "mean")]
rmse_std = rmse_df.loc["RMSE", ("HistGradientBoostingRegressor", "std")]
print(f"RMSE (GroupKFold-5, patient-grouped): {rmse_mean:.4f} ± {rmse_std:.4f}")

# %% [markdown]
# ## Push to Skore Hub

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put("04_hgbt", report)

hub_base = cfg["hub_url"].replace("api.", "")
report_url = f"{hub_base}/{cfg['workspace']}/projects/bobathon-esilv/reports/04_hgbt"
print(f"Skore Hub report URL: {report_url}")

# %% [markdown]
# ## Kaggle submission
#
# Refit on all training rows (no CV split) and predict the test visits.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_train_full, y_train_full = load_dataset_hgbt()
X_test = load_test_dataset_hgbt()
final = build_hgbt_learner()
final.fit(X_train_full, y_train_full)
X_test_features = X_test[[c for c in X_train_full.columns if c in X_test.columns]]
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test_features)
submission.to_csv(submissions_dir / "04_hgbt.csv", index=False)
print(f"Submission written to submissions/04_hgbt.csv ({len(submission)} rows)")
