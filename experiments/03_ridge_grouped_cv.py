# %% [markdown]
# # Experiment 03 — Ridge with patient-grouped CV
#
# **Date:** 2026-10-07
# **Goal:** Replace the leaky 80/20 row holdout with a patient-grouped
# 5-fold CV to get an honest RMSE estimate on unseen patients.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset, load_dataset_with_groups, load_test_dataset
from parkinson.evaluate import grouped_splitter
from parkinson.hub import load_skore_credentials
from parkinson.pipeline import build_ridge_learner

# %% [markdown]
# ## Load data and build the learner
#
# We load the dataset twice: once with patient groups (for CV) and once
# without (for the final refit on all training rows).

# %%
X, y, groups = load_dataset_with_groups()
learner = build_ridge_learner()

# %% [markdown]
# ## Evaluate with patient-grouped 5-fold CV
#
# `GroupKFold(n_splits=5)` ensures no patient's visits appear on both sides
# of a fold boundary.  `skore.evaluate` does not accept a `groups` argument
# directly, so we pre-compute the fold indices and pass the generator as
# `splitter`.  This is equivalent to calling `GroupKFold.split(X, y, groups)`
# at evaluation time.

# %%
report = skore.evaluate(
    learner,
    X=X,
    y=y,
    splitter=grouped_splitter.split(X, y, groups),
)
rmse_df = report.metrics.rmse()
rmse_mean = rmse_df.loc["RMSE", ("Ridge", "mean")]
rmse_std = rmse_df.loc["RMSE", ("Ridge", "std")]
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
project.put("03_ridge_grouped_cv", report)

hub_base = cfg["hub_url"].replace("api.", "")
report_url = (
    f"{hub_base}/{cfg['workspace']}/projects/bobathon-esilv/reports/03_ridge_grouped_cv"
)
print(f"Skore Hub report URL: {report_url}")

# %% [markdown]
# ## Kaggle submission
#
# Refit on all training rows (no CV split) and predict the test visits.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_train_full, y_train_full = load_dataset()
X_test = load_test_dataset()
final = build_ridge_learner()
final.fit(X_train_full, y_train_full)
X_test_features = X_test[[c for c in X_train_full.columns if c in X_test.columns]]
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test_features)
submission.to_csv(submissions_dir / "03_ridge_grouped_cv.csv", index=False)
print(
    f"Submission written to submissions/03_ridge_grouped_cv.csv"
    f" ({len(submission)} rows)"
)
