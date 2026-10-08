# %% [markdown]
# # Experiment 02 — Ridge regression
#
# **Date:** 2026-10-07
# **Goal:** Measure how much linear structure the 8 clinical features
# contain by replacing the dummy mean baseline with a Ridge regression.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset, load_test_dataset
from parkinson.evaluate import splitter
from parkinson.hub import load_skore_credentials
from parkinson.pipeline import build_ridge_learner

# %% [markdown]
# ## Load data and build the learner

# %%
X, y = load_dataset()
learner = build_ridge_learner()

# %% [markdown]
# ## Evaluate
#
# `skore.evaluate` runs the learner on the 80/20 holdout and returns an
# `EstimatorReport`.  We then print the RMSE from the held-out 20 %.
# The same `splitter=0.2` is used as for 01_dummy so the two numbers
# are directly comparable (same random holdout fraction, same data).

# %%
report = skore.evaluate(learner, X=X, y=y, splitter=splitter)
rmse = report.metrics.rmse()
print(f"RMSE (80/20 holdout): {rmse:.4f}")

# %% [markdown]
# ## Push to Skore Hub
#
# `load_skore_credentials()` reads `.skore` and exports the API key as an
# environment variable so `login(mode="hub")` does not open a browser.
# The project name is `bobathon-esilv` and the report key is `02_ridge`.

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put("02_ridge", report)

hub_base = cfg["hub_url"].replace("api.", "")
report_url = f"{hub_base}/{cfg['workspace']}/projects/bobathon-esilv/reports/02_ridge"
print(f"Skore Hub report URL: {report_url}")

# %% [markdown]
# ## Kaggle submission
#
# Refit on all training rows (no CV split) and write predictions for the
# test set to `submissions/02_ridge.csv`.  The learner is refit from
# scratch on all available training data — the CV report above is kept
# separate and not reused here.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset()
final = build_ridge_learner()
final.fit(X, y)
X_test_features = X_test[[c for c in X.columns if c in X_test.columns]]
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test_features)
submission.to_csv(submissions_dir / "02_ridge.csv", index=False)
print(f"Submission written to submissions/02_ridge.csv ({len(submission)} rows)")
