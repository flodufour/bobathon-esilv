# %% [markdown]
# # Experiment 06 — skrub DataOps: grouped CV baked into the graph
#
# **Date:** 2026-10-07
# **Goal:** Verify that baking `GroupKFold(n_splits=5)` + `patient_id` groups
# into the DataOps graph via `mark_as_X(cv=..., split_kwargs=...)` produces
# the same RMSE as 05_tabular_pipeline while removing the manual `groups`
# variable from the experiment script.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

from parkinson import PROJECT_ROOT
from parkinson.data import load_visits, load_test_dataset
from parkinson.hub import load_skore_credentials
from parkinson.pipeline import build_dataops_learner

# %% [markdown]
# ## Load data and build the learner
#
# `load_visits()` returns the raw merged DataFrame (all columns including
# `patient_id` and `target`).  The DataOps graph handles column selection
# and group wiring internally.

# %%
visits = load_visits()
learner = build_dataops_learner(visits_preview=visits)

# %% [markdown]
# ## Evaluate with patient-grouped 5-fold CV
#
# No explicit `splitter=` needed — `skore.evaluate` reads `cv=GroupKFold(5)`
# and `split_kwargs={"groups": patient_id}` from the `mark_as_X` node.

# %%
report = skore.evaluate(learner, data={"visits": visits})
rmse_df = report.metrics.rmse()
estimator_label = rmse_df.columns.get_level_values(0)[0]
rmse_mean = rmse_df.loc["RMSE", (estimator_label, "mean")]
rmse_std = rmse_df.loc["RMSE", (estimator_label, "std")]
print(f"Estimator label: {estimator_label}")
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
project.put("06_dataops", report)

hub_base = cfg["hub_url"].replace("api.", "")
report_url = (
    f"{hub_base}/{cfg['workspace']}/projects/bobathon-esilv/reports/06_dataops"
)
print(f"Skore Hub report URL: {report_url}")

# %% [markdown]
# ## Kaggle submission
#
# Refit on all training visits and predict the test set.
# The test DataFrame is passed under the same "visits" key; the graph
# drops `patient_id` and `Index` from features automatically.
# `target` is absent in the test set — the graph uses only feature columns
# at predict time.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset()
learner.fit({"visits": visits})
submission = X_test[["Index"]].copy()
submission["target"] = learner.predict({"visits": X_test})
submission.to_csv(submissions_dir / "06_dataops.csv", index=False)
print(
    f"Submission written to submissions/06_dataops.csv ({len(submission)} rows)"
)
