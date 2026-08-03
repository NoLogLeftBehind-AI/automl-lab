"""End-to-end smokes and leaderboard-discipline tests."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from automl import AutoML

from test_guardrails import make_classification_df, make_regression_df, small_config


@pytest.fixture
def artifact_dir(tmp_path):
    return str(tmp_path / "artifacts")


def test_classification_end_to_end(artifact_dir):
    df = make_classification_df(n=500)
    cfg = small_config(target="y", task="classification", artifact_dir=artifact_dir,
                       ensemble=True)
    aml = AutoML(cfg).run(df)

    # leaderboard discipline: champion comes from the fresh-fold column
    assert aml.champion_name in aml.final_scores
    lb = aml.leaderboard
    assert "final_LogLoss" in lb.columns and "screen_LogLoss" in lb.columns
    assert (lb["source"].str.contains("fresh-fold").fillna(False)
            | (lb["source"] == "stage-1 screen")).all()

    # artifacts round-trip
    bundle = joblib.load(f"{artifact_dir}/model.joblib")
    meta = json.loads(open(f"{artifact_dir}/metadata.json").read())
    X_new = df.drop(columns=["y"]).head(10)
    pred = bundle["pipeline"].predict(X_new)
    assert len(pred) == 10
    # the audit trail records the champion's own tuned params (keyed by name,
    # not family — variant champions once got the sibling's params)
    assert meta["tuned_params"] == aml.tuned.get(aml.champion_name, {}).get("params", {})
    # binary: threshold ships inside the pipeline and metadata agrees
    assert meta["optimal_threshold"] is not None
    proba = bundle["pipeline"].predict_proba(X_new)
    manual = (proba[:, 1] >= meta["optimal_threshold"]).astype(int)
    assert (pred == manual).all()
    # honest reporting fields
    assert meta["holdout_metrics_note"].startswith("holdout metrics come from")
    assert "fresh-fold" in meta["cv_score_basis"] or "fresh" in meta["cv_score_basis"]
    # the parsimony deployment pick ships in the metadata (leaderboard winner
    # stays the champion under the default policy)
    assert meta["deployment_recommendation"]["model"] in aml.final_scores
    assert meta["deployment_recommendation"]["policy"] == "best_score"
    # report exists
    assert (open(f"{artifact_dir}/report.html").read()).startswith("<html")


def test_regression_end_to_end(artifact_dir):
    df = make_regression_df(n=500)
    cfg = small_config(target="y", task="regression", artifact_dir=artifact_dir,
                       ensemble=True)
    aml = AutoML(cfg).run(df)
    assert "RMSE" in aml.holdout_metrics
    meta = json.loads(open(f"{artifact_dir}/metadata.json").read())
    assert meta["optimal_threshold"] is None  # regression never records a threshold
    bundle = joblib.load(f"{artifact_dir}/model.joblib")
    assert len(bundle["pipeline"].predict(df.drop(columns=["y"]).head(5))) == 5


def test_multiclass_threshold_is_none_and_rare_holdout_class_survives(artifact_dir):
    """Two original bugs: multiclass got threshold=0.5 serialized (deployment
    recipe then computed nonsense), and a class missing from the holdout
    crashed log_loss/roc_auc."""
    rs = np.random.RandomState(3)
    n = 400
    df = pd.DataFrame({"x1": rs.randn(n), "x2": rs.randn(n)})
    df["y"] = np.where(df.x1 > 0.5, "a", np.where(df.x2 > 0.5, "b", "c"))
    # a small (but survivable) class to stress per-class metrics
    df.loc[df.index[:12], "y"] = "d"
    cfg = small_config(target="y", task="classification", artifact_dir=artifact_dir)
    aml = AutoML(cfg).run(df)
    meta = json.loads(open(f"{artifact_dir}/metadata.json").read())
    assert meta["optimal_threshold"] is None
    assert np.isfinite(aml.holdout_metrics["LogLoss"])


def test_tuning_never_ships_worse_than_defaults():
    """Original bug: Optuna's best value replaced the default config even when
    strictly worse (and RF defaults weren't even representable in the space)."""
    df = make_regression_df(n=500)
    aml = AutoML(small_config(target="y", task="regression", optuna_trials=2))
    aml.load(df)
    aml.split()
    aml.scan_leakage()
    aml.prune_correlation()
    aml.screen()
    tuned = aml.tune()
    for name, info in tuned.items():
        if not info["improved"]:
            assert info["params"] == {}  # defaults kept, not a worse trial


def test_group_partition_keeps_groups_together():
    rs = np.random.RandomState(4)
    n = 600
    groups = rs.randint(0, 60, n)
    df = pd.DataFrame({"g": groups, "x": rs.randn(n) + groups * 0.05})
    df["y"] = (df.x + rs.randn(n) * 0.1 > 1.5).astype(int)
    cfg = small_config(target="y", task="classification", group_column="g",
                       force_keep_columns=["g"])
    aml = AutoML(cfg)
    aml.load(df)
    p = aml.split()
    # no group straddles the boundary, and 'g' is not a feature
    assert "g" not in p.X_train.columns
    train_groups = set(np.unique(p.groups_train))
    holdout_groups = set(df.loc[p.X_holdout.index, "g"].unique())
    assert not (train_groups & holdout_groups)


def test_time_partition_is_out_of_time():
    rs = np.random.RandomState(5)
    n = 500
    df = pd.DataFrame({
        "t": np.arange(n).astype(float),
        "x": rs.randn(n),
    })
    df["y"] = df.x * 2 + rs.randn(n) * 0.1
    cfg = small_config(target="y", task="regression", time_column="t",
                       force_keep_columns=["t"])
    aml = AutoML(cfg)
    aml.load(df)
    p = aml.split()
    assert p.X_train["t"].max() < p.X_holdout["t"].min()


def test_parquet_roundtrip(tmp_path):
    df = make_classification_df(n=300)
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    cfg = small_config(target="y", task="classification", data_path=str(path))
    aml = AutoML(cfg)
    aml.load()
    assert len(aml.X_all) > 0


def test_predict_script_runs(artifact_dir, tmp_path):
    """The original Dockerfile referenced a predict.py nothing generated."""
    import subprocess
    import sys
    df = make_classification_df(n=400)
    cfg = small_config(target="y", task="classification", artifact_dir=artifact_dir)
    AutoML(cfg).run(df)
    inp = tmp_path / "new.csv"
    outp = tmp_path / "preds.csv"
    df.drop(columns=["y"]).head(25).to_csv(inp, index=False)
    res = subprocess.run([sys.executable, f"{artifact_dir}/predict.py",
                          "--input", str(inp), "--output", str(outp)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    preds = pd.read_csv(outp)
    assert len(preds) == 25
    assert "prediction" in preds.columns
    assert any(c.startswith("proba_") for c in preds.columns)


def test_requirements_pin_only_the_champions_libraries(artifact_dir):
    """The artifact used to pin every model library present while *training*.

    On the Diamonds demo that shipped xgboost — and, through it, nvidia-nccl —
    inside the requirements of a champion that never touches xgboost: 485 MB of
    download closure for nothing. Pins now come from walking the exported
    pipeline, so an unused library must not appear.
    """
    from automl.artifacts import libraries_used

    df = make_regression_df(n=400)
    cfg = small_config(target="y", task="regression", artifact_dir=artifact_dir)
    aml = AutoML(cfg).run(df)

    text = (Path(artifact_dir) / "requirements.txt").read_text()
    required = [ln.strip() for ln in text.splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
    names = {ln.split("==")[0] for ln in required}

    # the core stack is always present and always pinned to an exact version
    assert {"scikit-learn", "pandas", "numpy", "joblib"} <= names
    assert all("==" in ln for ln in required)

    # every pinned model library must actually be reachable from the champion
    used = libraries_used(aml.final_model)
    for lib in ("xgboost", "lightgbm", "catboost"):
        if lib in names:
            assert lib in used, f"{lib} pinned but the champion does not use it"

    # metadata records the same list, so the claim is auditable from the bundle
    meta = json.loads((Path(artifact_dir) / "metadata.json").read_text())
    assert meta["inference_libraries"] == required

    # pyarrow is a convenience for parquet input, not a requirement for scoring
    assert "pyarrow" not in names
    if "pyarrow" in text:
        assert "# pyarrow" in text


def test_drift_checker_flags_shift(artifact_dir):
    df = make_regression_df(n=500)
    cfg = small_config(target="y", task="regression", artifact_dir=artifact_dir)
    aml = AutoML(cfg).run(df)
    shifted = aml.partition.X_holdout.copy()
    shifted["x1"] = shifted["x1"] + 10  # blatant drift
    report = aml.drift_check(shifted)
    row = report[report.feature == "x1"].iloc[0]
    assert row["status"] == "MAJOR DRIFT"


def test_mae_metric_switches_gbm_objectives():
    """Original bug: MAE was selected for skewed targets while every model kept
    optimizing squared error."""
    rs = np.random.RandomState(6)
    n = 500
    df = pd.DataFrame({"x": rs.randn(n)})
    df["y"] = np.exp(2 + df.x + rs.randn(n) * 1.2)  # heavy right skew, strictly positive
    aml = AutoML(small_config(target="y", task="regression"))
    aml.load(df)
    aml.split()
    assert aml.ta.primary_metric == "MAE"
    aml.scan_leakage()
    aml.prune_correlation()
    aml.screen()
    hgb = aml.roster["HistGradientBoosting"].steps[-1][1]
    assert hgb.loss == "absolute_error"
    # log-target siblings joined the roster
    assert any(name.endswith("_logy") for name in aml.roster)


def test_log_target_finalists_are_tunable():
    """Regression test: TransformedTargetRegressor roster variants need their
    search-space params remapped (model__X -> model__regressor__X); without the
    remap every Optuna trial died with 'Invalid parameter' and tuning was a
    silent no-op for _logy finalists."""
    from sklearn.base import clone

    from automl.search import remap_params_for_pipeline, search_space

    rs = np.random.RandomState(9)
    n = 400
    df = pd.DataFrame({"x": rs.randn(n)})
    df["y"] = np.exp(2 + df.x + rs.randn(n) * 1.2)
    aml = AutoML(small_config(target="y", task="regression"))
    aml.load(df)
    aml.split()
    aml.scan_leakage()
    aml.prune_correlation()
    from automl.models import build_roster
    roster = build_roster(aml.spec, aml.ta, aml.config, aml.log, n)
    logy = [name for name in roster if name.endswith("_logy")]
    assert logy, "skew should produce log-target variants"

    class FakeTrial:  # minimal stub: fixed values for every suggestion
        def suggest_float(self, name, lo, hi, log=False):
            return lo
        def suggest_int(self, name, lo, hi):
            return lo
        def suggest_categorical(self, name, choices):
            return choices[0]

    for name in logy:
        pipe = roster[name]
        params = search_space(FakeTrial(), name.split("_")[0])
        remapped = remap_params_for_pipeline(params, pipe)
        assert all(k.startswith("model__regressor__") for k in remapped)
        clone(pipe).set_params(**remapped)   # raised ValueError before the fix


def test_poisson_family_detection():
    rs = np.random.RandomState(7)
    n = 500
    df = pd.DataFrame({"x": rs.randn(n)})
    df["y"] = rs.poisson(np.exp(0.5 + 0.8 * df.x))
    aml = AutoML(small_config(target="y", task="regression"))
    aml.load(df)
    aml.split()
    assert aml.ta.objective_family == "poisson"


def test_imbalanced_adds_weighted_variants_not_forced_weighting():
    rs = np.random.RandomState(8)
    n = 1200
    df = pd.DataFrame({"x1": rs.randn(n), "x2": rs.randn(n)})
    df["y"] = (rs.rand(n) < 0.06).astype(int)  # ~16:1
    aml = AutoML(small_config(target="y", task="classification"))
    aml.load(df)
    aml.split()
    assert aml.ta.imbalanced
    aml.scan_leakage()
    aml.prune_correlation()
    from automl.models import build_roster
    roster = build_roster(aml.spec, aml.ta, aml.config, aml.log, n)
    assert any(name.endswith("_weighted") for name in roster)
    # the un-weighted models stay un-weighted (calibration protection)
    assert roster["LogisticRegression"].steps[-1][1].class_weight is None
