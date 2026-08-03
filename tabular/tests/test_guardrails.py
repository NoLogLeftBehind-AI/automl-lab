"""Guardrail regression tests.

Each test here pins a failure mode found during review of the original
notebooks: the engine must either handle the situation or stop with a clear
AutoMLError — never a cryptic crash or a silent wrong answer.
"""
import numpy as np
import pandas as pd
import pytest

from automl import AutoML, AutoMLConfig, AutoMLError

RS = np.random.RandomState(0)


def small_config(**kw):
    defaults = dict(cv_folds=3, optuna_trials=2, optuna_timeout=20, n_finalists=2,
                    bootstrap_samples=100, ensemble=False, n_jobs=2)
    defaults.update(kw)
    return AutoMLConfig(**defaults)


def make_classification_df(n=400, seed=0):
    rs = np.random.RandomState(seed)
    df = pd.DataFrame({
        "x1": rs.randn(n), "x2": rs.randn(n),
        "cat": rs.choice(["a", "b", "c"], n),
    })
    logit = 1.5 * df.x1 + (df.cat == "a") * 1.0
    df["y"] = (rs.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def make_regression_df(n=400, seed=0):
    rs = np.random.RandomState(seed)
    df = pd.DataFrame({
        "x1": rs.randn(n), "x2": rs.randn(n),
        "cat": rs.choice(["a", "b", "c"], n),
    })
    df["y"] = 3 * df.x1 - 2 * (df.cat == "b") + rs.randn(n) * 0.5
    return df


# ---------------------------------------------------------------- leakage ---

def test_regression_leak_target_copy_is_caught():
    """The original depth-3-tree scan could NOT flag a literal copy of a
    continuous target (R2 ceiling ~0.965 < 0.98 threshold)."""
    df = make_regression_df()
    df["leak"] = df["y"] * 1.0001 + RS.randn(len(df)) * 1e-4
    aml = AutoML(small_config(target="y", task="regression"))
    aml.load(df)
    aml.split()
    aml.scan_leakage()
    assert "leak" in aml.leaky


def test_regression_noisy_leak_is_caught():
    """A leak with noise (Spearman ~0.99, tree R2 ~0.9) must still be flagged."""
    df = make_regression_df(n=800)
    df["leak"] = df["y"] + RS.randn(len(df)) * df["y"].std() * 0.02
    aml = AutoML(small_config(target="y", task="regression"))
    aml.load(df)
    aml.split()
    aml.scan_leakage()
    assert "leak" in aml.leaky


def test_classification_leak_is_caught():
    df = make_classification_df()
    df["leak"] = df["y"]
    aml = AutoML(small_config(target="y", task="classification"))
    aml.load(df)
    aml.split()
    aml.scan_leakage()
    assert "leak" in aml.leaky


def test_force_keep_overrides_leak_flag():
    df = make_classification_df()
    df["leak"] = df["y"]
    aml = AutoML(small_config(target="y", task="classification",
                              force_keep_columns=["leak"]))
    aml.load(df)
    aml.split()
    aml.scan_leakage()
    assert "leak" not in aml.leaky


# ----------------------------------------------------------- rare classes ---

def test_single_member_class_does_not_crash():
    """The original notebooks crashed in train_test_split(stratify=...)."""
    df = make_classification_df()
    df.loc[df.index[-1], "y"] = 2  # one row of a third class
    aml = AutoML(small_config(target="y", task="classification"))
    aml.load(df)
    aml.split()  # must not raise
    assert aml.ta.n_classes == 2  # rare class dropped, loudly


def test_rare_class_policy_error():
    df = make_classification_df()
    df.loc[df.index[-1], "y"] = 2
    aml = AutoML(small_config(target="y", task="classification",
                              rare_class_policy="error"))
    with pytest.raises(AutoMLError, match="fewer than"):
        aml.load(df)


# ------------------------------------------------------------ bad targets ---

def test_zero_variance_target_is_guarded():
    """Originally: every feature scored R2=1.0, all dropped as 'leaky', then
    a cryptic KeyError. Now: a clear error at target analysis."""
    df = make_regression_df()
    df["y"] = 5.0
    aml = AutoML(small_config(target="y", task="regression"))
    with pytest.raises(AutoMLError, match="zero variance"):
        aml.load(df)


def test_string_target_for_regression_points_at_classification():
    df = make_regression_df()
    df["y"] = RS.choice(["low", "high"], len(df))
    aml = AutoML(small_config(target="y", task="regression"))
    with pytest.raises(AutoMLError, match="classification"):
        aml.load(df)


def test_task_auto_infers_classification_for_string_target():
    df = make_classification_df()
    df["y"] = df["y"].map({0: "no", 1: "yes"})
    aml = AutoML(small_config(target="y", task="auto"))
    aml.load(df)
    assert aml.ta.task == "classification"
    assert aml.ta.classes == ["no", "yes"]


# -------------------------------------------------------- feature hygiene ---

def test_all_features_dropped_is_a_clear_error():
    n = 300
    df = pd.DataFrame({
        "id": np.arange(n),                       # ID-like
        "const": 1.0,                             # constant
        "y": RS.randn(n),
    })
    aml = AutoML(small_config(target="y", task="regression"))
    with pytest.raises(AutoMLError, match="Every feature was dropped"):
        aml.load(df)


def test_duplicate_columns_is_a_clear_error():
    df = make_classification_df()
    df = pd.concat([df, df[["x1"]]], axis=1)  # duplicate 'x1'
    aml = AutoML(small_config(target="y", task="classification"))
    with pytest.raises(AutoMLError, match="Duplicate column names"):
        aml.load(df)


def test_special_characters_in_column_names_are_sanitized():
    df = make_classification_df()
    df = df.rename(columns={"x1": "x[1]<weird>", "x2": "x,2"})
    aml = AutoML(small_config(target="y", task="classification"))
    aml.load(df)
    assert aml.profile_result.recipe["renamed_columns"]
    assert not any("[" in c for c in aml.X_all.columns)


def test_missing_target_column_lists_available():
    df = make_classification_df()
    aml = AutoML(small_config(target="nope", task="classification"))
    with pytest.raises(AutoMLError, match="not found"):
        aml.load(df)


def test_datetime_columns_become_features_not_drops():
    n = 400
    rs = np.random.RandomState(1)
    dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(rs.randint(0, 700, n), unit="D")
    df = pd.DataFrame({"when": dates.astype(str), "x": rs.randn(n)})
    df["y"] = (dates.month.isin([6, 7, 8]).astype(int) + (df.x > 0)).clip(0, 1)
    aml = AutoML(small_config(target="y", task="classification"))
    aml.load(df)
    assert "when__month" in aml.X_all.columns
    assert "when" not in aml.X_all.columns
    assert "when" in aml.profile_result.recipe["date_decompositions"]


def test_text_columns_are_vectorized_not_dropped():
    n = 300
    rs = np.random.RandomState(2)
    words_pos = ["refund broken angry cancel terrible support waited weeks nobody answered"]
    words_neg = ["great love excellent fast wonderful perfect recommend happy amazing service"]
    text, y = [], []
    for _ in range(n):
        if rs.rand() < 0.5:
            text.append(words_pos[0] + " " + str(rs.randint(100)))
            y.append(1)
        else:
            text.append(words_neg[0] + " " + str(rs.randint(100)))
            y.append(0)
    df = pd.DataFrame({"comment": text, "x": rs.randn(n), "y": y})
    aml = AutoML(small_config(target="y", task="classification"))
    aml.load(df)
    assert aml.profile_result.text_columns == ["comment"]


# ------------------------------------------------- review regression tests ---

def test_screening_subsample_preserves_row_order():
    """The stage-1 sample must keep positional order: a shuffled subsample fed
    to TimeSeriesSplit would train expanding windows on the future."""
    from automl.core import _screen_sample_indices
    y = pd.Series(np.random.RandomState(0).randn(500).cumsum())
    idx = _screen_sample_indices(y, "regression", 100, 42)
    assert len(idx) == 100
    assert (np.diff(idx) > 0).all()


def test_time_partition_survives_unparseable_timestamps():
    """pandas-2 Series.argsort yields corrupt indices on missing values; the
    partitioner must order NaT-safely and keep every row exactly once."""
    df = make_regression_df(n=200)
    dates = pd.date_range("2023-01-01", periods=200, freq="D").astype(str).tolist()
    dates[7] = "not-a-date"
    df["when"] = dates
    aml = AutoML(small_config(target="y", task="regression", time_column="when"))
    aml.load(df)
    p = aml.split()
    assert len(p.X_train) + len(p.X_holdout) == len(aml.X_all)
    assert not p.X_train.index.duplicated().any()
    tr = p.X_train["when__epoch_days"].dropna()
    ho = p.X_holdout["when__epoch_days"].dropna()
    assert tr.max() <= ho.min()


def test_timezone_aware_datetimes_are_handled():
    df = make_classification_df(n=300)
    df["ts"] = pd.date_range("2023-01-01", periods=300, freq="h", tz="UTC")
    aml = AutoML(small_config(target="y", task="classification"))
    aml.load(df)
    assert "ts__epoch_days" in aml.X_all.columns


def test_invalid_config_strings_rejected():
    with pytest.raises(ValueError, match="rare_class_policy"):
        AutoMLConfig(rare_class_policy="ignore")
    with pytest.raises(ValueError, match="threshold_objective"):
        AutoMLConfig(threshold_objective="accuracy")
    with pytest.raises(ValueError, match="champion_policy"):
        AutoMLConfig(champion_policy="fastest")


def _scores(mean_by_name: dict, folds_by_name: dict | None = None) -> dict:
    """Build a parsimony_pick input; entries with folds get mean/std derived."""
    folds_by_name = folds_by_name or {}
    out = {}
    for name in [*mean_by_name, *(n for n in folds_by_name if n not in mean_by_name)]:
        if name in folds_by_name:
            f = folds_by_name[name]
            out[name] = {"mean": float(np.mean(f)), "std": float(np.std(f)), "folds": f}
        else:
            out[name] = {"mean": mean_by_name[name], "std": 0.004}
    return out


def test_parsimony_pick_prefers_simplicity_within_noise():
    """The deployment pick is the simplest candidate within one standard ERROR
    of the best final score — a blend must EARN its complexity with a margin
    the folds can actually resolve."""
    from automl.core import parsimony_pick
    # scores in sklearn's higher-is-better convention (neg-logloss here);
    # no 'folds' -> the documented fallback band (the best model's std)
    scores = _scores({"Voting(3)": -0.275, "XGBoost": -0.277, "CatBoost": -0.278})
    assert parsimony_pick(scores) == "XGBoost"
    # among singles within the band, the better score wins the tie
    scores["CatBoost"]["mean"] = -0.276
    assert parsimony_pick(scores) == "CatBoost"
    # a decisive blend win keeps the blend
    scores["Voting(3)"]["mean"] = -0.260
    assert parsimony_pick(scores) == "Voting(3)"
    # voting is preferred over stacking at equal evidence
    scores["Stacked(3)"] = {"mean": -0.260, "std": 0.004}
    assert parsimony_pick(scores) == "Voting(3)"


def test_parsimony_band_is_a_standard_error_on_paired_folds():
    """The band is the SE of the per-fold *differences*, not the marginal
    fold-to-fold spread.

    Candidates share one fixed splitter, so fold difficulty is common to both
    and cancels. A small-but-perfectly-consistent blend win is real evidence
    even when it is far inside either model's own fold spread — the marginal
    std this rule used to band on is ~sqrt(k) too wide and would have thrown
    that win away.
    """
    from automl.core import one_se_band, parsimony_pick

    # fold difficulty varies a lot (std ~0.0075) but the blend beats the single
    # model by exactly 0.002 on EVERY fold -> zero paired spread, real win.
    voting = [-0.270, -0.280, -0.265, -0.285, -0.275]
    xgb = [v - 0.002 for v in voting]
    scores = _scores({}, {"Voting(3)": voting, "XGBoost": xgb})
    assert np.std(voting) > 0.007                      # marginal spread is wide
    assert one_se_band(scores["Voting(3)"], scores["XGBoost"]) == pytest.approx(0, abs=1e-12)
    assert parsimony_pick(scores) == "Voting(3)"       # consistent win survives

    # same 0.002 mean margin, but now the per-fold differences are erratic ->
    # the paired SE is large and the simpler model ships.
    noisy = [v - d for v, d in zip(voting, [0.012, -0.008, 0.010, -0.006, 0.002])]
    scores = _scores({}, {"Voting(3)": voting, "XGBoost": noisy})
    assert np.mean(voting) - np.mean(noisy) == pytest.approx(0.002, abs=1e-9)
    assert one_se_band(scores["Voting(3)"], scores["XGBoost"]) > 0.002
    assert parsimony_pick(scores) == "XGBoost"


def test_fresh_cv_reshuffles_group_regression():
    """Group-regression re-scores must not silently reuse the selection folds."""
    from automl.partition import fresh_cv
    from automl.target import TargetAnalysis
    ta = TargetAnalysis()
    ta.task = "regression"
    cfg = small_config(target="y", group_column="g")
    cv = fresh_cv(ta, cfg, seed_offset=1000)
    assert cv.__class__.__name__ == "GroupKFold"
    assert getattr(cv, "shuffle", False) is True


def test_log_target_variant_early_stops():
    """TransformedTargetRegressor GBM variants must early-stop like their
    plain siblings: the probe unwraps the TTR and validates on the log scale."""
    lightgbm = pytest.importorskip("lightgbm")
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from automl.models import fit_with_early_stopping, supports_early_stopping

    rs = np.random.RandomState(0)
    X = pd.DataFrame({"x1": rs.randn(600), "x2": rs.randn(600)})
    y = pd.Series(np.exp(0.5 * X.x1 + rs.randn(600) * 0.1) * 50)
    pipe = Pipeline([
        ("prep", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=lightgbm.LGBMRegressor(n_estimators=400, random_state=0,
                                             n_jobs=1, verbose=-1),
            func=np.log1p, inverse_func=np.expm1)),
    ])
    assert supports_early_stopping(pipe)
    fitted, best_n = fit_with_early_stopping(pipe, X, y, "regression", 0)
    assert best_n is not None and best_n >= 1
    inner = fitted.steps[-1][1].regressor_
    assert inner.get_params()["n_estimators"] == max(10, int(best_n * 1.1))


def test_multiclass_xgboost_gets_mlogloss_and_early_stops():
    """XGBoost once hardcoded eval_metric='logloss', which xgboost rejects for
    multiclass eval sets — the early-stopping probe failed on every multiclass
    run and the bare except swallowed it, silently shipping the fixed 400
    trees. The roster must pick a per-task metric, and a failed probe must
    land in the decision log."""
    pytest.importorskip("xgboost")
    from sklearn.base import clone

    from automl import AutoML
    from automl.models import build_roster, fit_with_early_stopping
    from automl.utils import DecisionLog

    rs = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame({"x1": rs.randn(n), "x2": rs.randn(n)})
    df["y"] = (df.x1 > 0.4).astype(int) + (df.x1 > -0.4).astype(int)  # 3 classes
    aml = AutoML(small_config(target="y", task="classification"))
    aml.load(df)
    aml.split()
    aml.scan_leakage()
    aml.prune_correlation()
    roster = build_roster(aml.spec, aml.ta, aml.config, aml.log,
                          len(aml.partition.X_train))

    xgb_pipe = roster["XGBoost"]
    assert xgb_pipe.steps[-1][1].get_params()["eval_metric"] == "mlogloss"
    _, best_n = fit_with_early_stopping(xgb_pipe, aml.partition.X_train,
                                        aml.partition.y_train, "classification", 0)
    assert best_n is not None and best_n >= 1

    # the historical failure mode must now be logged, never swallowed
    broken = clone(xgb_pipe).set_params(model__eval_metric="logloss")
    log = DecisionLog(verbose=False)
    _, none_n = fit_with_early_stopping(broken, aml.partition.X_train,
                                        aml.partition.y_train, "classification", 0,
                                        log=log)
    assert none_n is None
    assert any("early-stopping probe failed" in r["decision"] for r in log.records)


def test_chronological_early_stopping_split_uses_the_tail():
    """Time-aware runs must not validate the early-stopping probe on a random
    slice — the probe would train on rows later than its validation data."""
    lightgbm = pytest.importorskip("lightgbm")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from automl.models import fit_with_early_stopping

    n = 500
    X = pd.DataFrame({"t": np.arange(n, dtype=float)})
    # a strong level shift in the last 10%: the chronological probe validates
    # entirely on the shifted regime (a shuffled probe would mix regimes)
    y = pd.Series(np.where(np.arange(n) < 450, 0.0, 100.0))
    pipe = Pipeline([
        ("prep", StandardScaler()),
        ("model", lightgbm.LGBMRegressor(n_estimators=400, random_state=0,
                                         n_jobs=1, verbose=-1)),
    ])
    fitted, best_n = fit_with_early_stopping(pipe, X, y, "regression", 0,
                                             chronological=True)
    assert best_n is not None and best_n >= 1
    # the refit pipeline saw the whole series, so it must know the shift level
    assert float(fitted.predict(X.tail(10)).mean()) > 50.0


def test_config_object_is_reusable_across_runs():
    """Column sanitization once rewrote config.target on the caller's config
    object, so a second run with the same config died with 'not found'."""
    df = make_classification_df().rename(columns={"y": "y:"})
    cfg = small_config(target="y:", task="classification")
    for _ in range(2):
        aml = AutoML(cfg)
        aml.load(df)
        assert aml.config.target == "y_"      # the run's own copy is sanitized
        assert cfg.target == "y:"             # the caller's object is untouched


def test_native_categorical_indices_point_at_the_ordinal_coded_columns():
    """HistGradientBoosting gets `categorical_features` as *positions*, derived
    by native_categorical_indices() from the FeatureSpec alone. (LightGBM is
    NOT wired to receive these — see the native_cat note in preprocess.py.)
    That is correct only while make_preprocessor appends numeric ->
    low-cardinality -> target-encoded -> text in exactly that order. Insert or
    reorder a branch and HistGB silently treats target-encoded floats as
    categories: no exception, no warning, just quietly worse models. This pins
    the index math that the wiring depends on.
    """
    from automl.preprocess import (FeatureSpec, make_preprocessor,
                                   native_categorical_indices)

    n = 300
    rs = np.random.RandomState(0)
    base = pd.DataFrame({
        "num1": rs.randn(n),
        "num2": rs.randn(n),
        "low": rs.choice(["a", "b", "c"], n),          # low cardinality
        "high": [f"id{i % 60}" for i in range(n)],     # above the threshold
    })
    y = pd.Series(rs.randn(n))

    # the text branch is appended last, so it must not shift the categorical
    # positions — check both with and without it
    for text_cols in ([], ["txt"]):
        X = base.copy()
        if text_cols:
            X["txt"] = "some free text here"
        spec = FeatureSpec(X, text_columns=text_cols, high_cardinality_thresh=10)
        assert spec.num_cols == ["num1", "num2"]
        assert spec.low_card_cols == ["low"]
        assert spec.high_card_cols == ["high"]

        out = make_preprocessor("native_cat", spec, "regression", 0, n).fit_transform(X, y)
        cat_idx = native_categorical_indices(spec)

        # the flagged positions carry integer codes at the source cardinality...
        for i, col in zip(cat_idx, spec.low_card_cols):
            vals = np.unique(np.asarray(out[:, i], dtype=float))
            assert len(vals) == X[col].nunique(), f"position {i} is not '{col}'"
            assert np.array_equal(vals, np.round(vals)), f"position {i} is not ordinal codes"

        # ...and the target-encoded high-cardinality column is not among them
        te_idx = len(spec.num_cols) + len(spec.low_card_cols)
        assert te_idx not in cat_idx
        assert len(np.unique(out[:, te_idx])) > X["low"].nunique()
