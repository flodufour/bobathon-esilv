# %% [markdown]
# # Experiment 05 — skrub tabular_pipeline with mixed-type features
#
# **Date:** 2026-10-07
# **Goal:** Add categorical columns (`gene`, `cohort`, `rater_id`) to the
# feature matrix.  `skrub.tabular_pipeline("regressor")` selects an encoder
# per column automatically (one-hot for low-cardinality strings, StringEncoder
# for high-cardinality, pass-through for numerics) and pairs it with HGBR.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

from parkinson import PROJECT_ROOT
from parkinson.data import load_full_dataset, load_test_dataset
from parkinson.evaluate import grouped_splitter
from parkinson.hub import load_skore_credentials
from parkinson.pipeline import build_tabular_learner

# %% [markdown]
# ## Load data and build the learner
#
# `load_full_dataset()` returns all clinical columns except `Index` and
# `patient_id`, including the categorical columns `gene`, `cohort`, and
# `rater_id`.

# %%
X, y, groups = load_full_dataset()
learner = build_tabular_learner()

# %% [markdown]
# ## Evaluate with patient-grouped 5-fold CV
#
# Same `GroupKFold(n_splits=5)` splitter used in 03 and 04 so results
# are directly comparable.

# %%
report = skore.evaluate(
    learner,
    X=X,
    y=y,
    splitter=grouped_splitter.split(X, y, groups),
)
rmse_df = report.metrics.rmse()
# tabular_pipeline wraps an unnamed pipeline — estimator label may vary
estimator_label = rmse_df.columns.get_level_values(0)[0]
rmse_mean = rmse_df.loc["RMSE", (estimator_label, "mean")]
rmse_std = rmse_df.loc["RMSE", (estimator_label, "std")]
print(
    f"Estimator label: {estimator_label}"
)
print(
    f"RMSE (GroupKFold-5, patient-grouped): {rmse_mean:.4f} ± {rmse_std:.4f}"
)

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
project.put("05_tabular_pipeline", report)

hub_base = cfg["hub_url"].replace("api.", "")
report_url = (
    f"{hub_base}/{cfg['workspace']}/projects/bobathon-esilv/reports/05_tabular_pipeline"
)
print(f"Skore Hub report URL: {report_url}")

# %% [markdown]
# ## Kaggle submission
#
# Refit on all training rows (no CV split) and predict the test visits.
# The test feature matrix must use the same columns as the training matrix
# (all clinical columns except `Index` and `patient_id`).

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test_raw = load_test_dataset()
# Align test columns: same set as X (no patient_id, no Index beyond keeping for submission)
test_feature_cols = [c for c in X.columns if c in X_test_raw.columns]
X_test_features = X_test_raw[test_feature_cols]

final = build_tabular_learner()
final.fit(X, y)
submission = X_test_raw[["Index"]].copy()
submission["target"] = final.predict(X_test_features)
submission.to_csv(submissions_dir / "05_tabular_pipeline.csv", index=False)
print(
    f"Submission written to submissions/05_tabular_pipeline.csv"
    f" ({len(submission)} rows)"
)
