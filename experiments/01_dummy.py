# %% [markdown]
# # Experiment 01 — Dummy baseline
#
# **Date:** 2025-07-14
# **Goal:** Establish the mean-prediction RMSE floor on an 80/20 row holdout.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset, load_test_dataset
from parkinson.evaluate import splitter
from parkinson.hub import load_skore_credentials
from parkinson.pipeline import build_learner

# %% [markdown]
# ## Load data and build the learner

# %%
X, y = load_dataset()
learner = build_learner()

# %% [markdown]
# ## Evaluate
#
# `skore.evaluate` runs the learner on the 80/20 holdout and returns an
# `EstimatorReport`.  The report stores predictions, metrics, and
# diagnostic plots.  We then print the RMSE from the held-out 20 %.

# %%
report = skore.evaluate(learner, X=X, y=y, splitter=splitter)
rmse = report.metrics.rmse()
print(f"RMSE (80/20 holdout): {rmse:.4f}")

# %% [markdown]
# ## Push to Skore Hub
#
# `load_skore_credentials()` reads `.skore` and exports the API key as an
# environment variable so `login(mode="hub")` does not open a browser.
# The project name is `bobathon-esilv` and the report key is `01_dummy`.

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put("01_dummy", report)

hub_base = cfg["hub_url"].replace("api.", "")
report_url = f"{hub_base}/{cfg['workspace']}/projects/bobathon-esilv/reports/01_dummy"
print(f"Skore Hub report URL: {report_url}")

# %% [markdown]
# ## Kaggle submission
#
# Refit on all training rows (no CV split) and write predictions for the
# test set to `submissions/01_dummy.csv`.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset()
final = build_learner()
final.fit(X, y)
X_test_features = X_test[[c for c in X.columns if c in X_test.columns]]
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test_features)
submission.to_csv(submissions_dir / "01_dummy.csv", index=False)
print(f"Submission written to submissions/01_dummy.csv ({len(submission)} rows)")
