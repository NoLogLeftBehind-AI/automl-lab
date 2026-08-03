"""Model roster construction and early-stopping-aware fitting.

Roster principles:

- Diversity over depth: the leaderboard exists to discover which *family* suits
  the dataset; Optuna deepens the winners later.
- Objectives match the metric and target family: MAE-primary runs get
  absolute-error GBM objectives; Poisson/Tweedie/Gamma targets get matching
  deviance objectives; skewed positive targets get log1p-target siblings via
  TransformedTargetRegressor (scored on the original scale).
- Gradient boosting picks its own tree count via early stopping instead of
  burning Optuna trials on ``n_estimators``.
- Models run single-threaded inside joblib-parallel CV (nested parallelism can
  slow LightGBM ~1000x); final fits get all cores back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (ElasticNet, GammaRegressor, LogisticRegression,
                                  PoissonRegressor, Ridge, TweedieRegressor)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import AutoMLConfig
from .preprocess import FeatureSpec, make_preprocessor, native_categorical_indices
from .target import TargetAnalysis
from .utils import DecisionLog

OPTIONAL = {}
try:
    from xgboost import XGBClassifier, XGBRegressor
    OPTIONAL["xgboost"] = True
except ImportError:
    OPTIONAL["xgboost"] = False
try:
    import lightgbm
    from lightgbm import LGBMClassifier, LGBMRegressor
    OPTIONAL["lightgbm"] = True
except ImportError:
    OPTIONAL["lightgbm"] = False
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    OPTIONAL["catboost"] = True
except ImportError:
    OPTIONAL["catboost"] = False
try:
    import optuna  # noqa: F401
    OPTIONAL["optuna"] = True
except ImportError:
    OPTIONAL["optuna"] = False
try:
    import shap  # noqa: F401
    OPTIONAL["shap"] = True
except ImportError:
    OPTIONAL["shap"] = False

# families whose tree count is chosen by early stopping at fit time
ES_FAMILIES = ("XGBoost", "LightGBM", "CatBoost")


def _final_estimator(pipe):
    est = pipe
    while True:
        if hasattr(est, "steps"):
            est = est.steps[-1][1]
        elif isinstance(est, TransformedTargetRegressor):
            est = est.regressor
        else:
            return est


def family_of(name: str) -> str:
    return name.split("_")[0]


def build_roster(spec: FeatureSpec, ta: TargetAnalysis, config: AutoMLConfig,
                 log: DecisionLog, n_rows: int) -> dict:
    rng = config.random_state
    prep = lambda kind: make_preprocessor(kind, spec, ta.task, rng, n_rows)  # noqa: E731
    cat_idx = native_categorical_indices(spec)
    roster = {}

    if ta.task == "classification":
        roster["LogisticRegression"] = Pipeline([
            ("prep", prep("linear")),
            ("model", LogisticRegression(max_iter=2000, random_state=rng)),
        ])
        roster["RandomForest"] = Pipeline([
            ("prep", prep("tree")),
            ("model", RandomForestClassifier(n_estimators=300, random_state=rng, n_jobs=1)),
        ])
        roster["ExtraTrees"] = Pipeline([
            ("prep", prep("tree")),
            ("model", ExtraTreesClassifier(n_estimators=300, random_state=rng, n_jobs=1)),
        ])
        hgb_kw: dict = dict(random_state=rng)
        if cat_idx:
            hgb_kw["categorical_features"] = cat_idx
        if config.early_stopping:
            hgb_kw.update(early_stopping=True, validation_fraction=0.1,
                          n_iter_no_change=20, max_iter=1500)
        roster["HistGradientBoosting"] = Pipeline([
            ("prep", prep("native_cat")),
            ("model", HistGradientBoostingClassifier(**hgb_kw)),
        ])
        if OPTIONAL["xgboost"]:
            # eval_metric must match the class count: xgboost's elementwise
            # "logloss" rejects multi:softprob eval sets, which would kill the
            # early-stopping probe on every multiclass run
            roster["XGBoost"] = Pipeline([
                ("prep", prep("tree")),
                ("model", XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6,
                                        tree_method="hist",
                                        eval_metric="logloss" if ta.is_binary else "mlogloss",
                                        random_state=rng, n_jobs=1, verbosity=0)),
            ])
        if OPTIONAL["lightgbm"]:
            lgb_kw: dict = dict(n_estimators=400, learning_rate=0.1, random_state=rng,
                                n_jobs=1, verbose=-1)
            roster["LightGBM"] = Pipeline([
                ("prep", prep("native_cat")),
                ("model", LGBMClassifier(**lgb_kw)),
            ])
        if OPTIONAL["catboost"]:
            roster["CatBoost"] = Pipeline([
                ("prep", prep("tree")),
                ("model", CatBoostClassifier(iterations=400, learning_rate=0.1, depth=6,
                                             random_state=rng, verbose=0, thread_count=1,
                                             allow_writing_files=False)),
            ])
        # Imbalance: weighted variants COMPETE on the leaderboard rather than being
        # forced on every model — class weighting distorts the calibrated
        # probabilities that LogLoss (the primary metric) exists to protect.
        if ta.imbalanced:
            roster["LogisticRegression_weighted"] = Pipeline([
                ("prep", prep("linear")),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=rng)),
            ])
            if OPTIONAL["lightgbm"]:
                roster["LightGBM_weighted"] = Pipeline([
                    ("prep", prep("native_cat")),
                    ("model", LGBMClassifier(n_estimators=400, learning_rate=0.1,
                                             class_weight="balanced", random_state=rng,
                                             n_jobs=1, verbose=-1)),
                ])
            log.log("roster", "Imbalanced target -> class-weighted variants added "
                              "(they compete; weighting is not forced)")
    else:
        mae_primary = ta.primary_metric == "MAE"
        roster["Ridge"] = Pipeline([
            ("prep", prep("linear")), ("model", Ridge(random_state=rng)),
        ])
        roster["ElasticNet"] = Pipeline([
            ("prep", prep("linear")), ("model", ElasticNet(max_iter=5000, random_state=rng)),
        ])
        roster["RandomForest"] = Pipeline([
            ("prep", prep("tree")),
            ("model", RandomForestRegressor(n_estimators=300, random_state=rng, n_jobs=1)),
        ])
        roster["ExtraTrees"] = Pipeline([
            ("prep", prep("tree")),
            ("model", ExtraTreesRegressor(n_estimators=300, random_state=rng, n_jobs=1)),
        ])
        hgb_kw = dict(random_state=rng,
                      loss="absolute_error" if mae_primary else "squared_error")
        if cat_idx:
            hgb_kw["categorical_features"] = cat_idx
        if config.early_stopping:
            hgb_kw.update(early_stopping=True, validation_fraction=0.1,
                          n_iter_no_change=20, max_iter=1500)
        roster["HistGradientBoosting"] = Pipeline([
            ("prep", prep("native_cat")),
            ("model", HistGradientBoostingRegressor(**hgb_kw)),
        ])
        if OPTIONAL["xgboost"]:
            roster["XGBoost"] = Pipeline([
                ("prep", prep("tree")),
                ("model", XGBRegressor(n_estimators=400, learning_rate=0.1, max_depth=6,
                                       objective="reg:absoluteerror" if mae_primary
                                       else "reg:squarederror",
                                       tree_method="hist", random_state=rng,
                                       n_jobs=1, verbosity=0)),
            ])
        if OPTIONAL["lightgbm"]:
            roster["LightGBM"] = Pipeline([
                ("prep", prep("native_cat")),
                ("model", LGBMRegressor(n_estimators=400, learning_rate=0.1,
                                        objective="regression_l1" if mae_primary else "regression",
                                        random_state=rng, n_jobs=1, verbose=-1)),
            ])
        if OPTIONAL["catboost"]:
            roster["CatBoost"] = Pipeline([
                ("prep", prep("tree")),
                ("model", CatBoostRegressor(iterations=400, learning_rate=0.1, depth=6,
                                            loss_function="MAE" if mae_primary else "RMSE",
                                            random_state=rng, verbose=0, thread_count=1,
                                            allow_writing_files=False)),
            ])

        # ----- objective-family variants (Poisson / Tweedie / Gamma) ---------
        fam = ta.objective_family
        if fam == "poisson":
            roster["PoissonGLM"] = Pipeline([
                ("prep", prep("linear")), ("model", PoissonRegressor(max_iter=1000)),
            ])
            hgb_p = dict(hgb_kw)
            hgb_p["loss"] = "poisson"
            roster["HistGradientBoosting_poisson"] = Pipeline([
                ("prep", prep("native_cat")),
                ("model", HistGradientBoostingRegressor(**hgb_p)),
            ])
            if OPTIONAL["lightgbm"]:
                roster["LightGBM_poisson"] = Pipeline([
                    ("prep", prep("native_cat")),
                    ("model", LGBMRegressor(n_estimators=400, learning_rate=0.1,
                                            objective="poisson", random_state=rng,
                                            n_jobs=1, verbose=-1)),
                ])
        elif fam == "tweedie":
            roster["TweedieGLM"] = Pipeline([
                ("prep", prep("linear")),
                ("model", TweedieRegressor(power=1.5, max_iter=1000)),
            ])
            if OPTIONAL["lightgbm"]:
                roster["LightGBM_tweedie"] = Pipeline([
                    ("prep", prep("native_cat")),
                    ("model", LGBMRegressor(n_estimators=400, learning_rate=0.1,
                                            objective="tweedie", tweedie_variance_power=1.3,
                                            random_state=rng, n_jobs=1, verbose=-1)),
                ])
        elif fam == "gamma":
            roster["GammaGLM"] = Pipeline([
                ("prep", prep("linear")), ("model", GammaRegressor(max_iter=1000)),
            ])
            hgb_g = dict(hgb_kw)
            hgb_g["loss"] = "gamma"
            roster["HistGradientBoosting_gamma"] = Pipeline([
                ("prep", prep("native_cat")),
                ("model", HistGradientBoostingRegressor(**hgb_g)),
            ])

        # ----- log1p-target siblings (scored on the original scale) ----------
        if ta.log_target_candidate:
            for base in ("HistGradientBoosting", "LightGBM"):
                if base in roster:
                    base_pipe = roster[base]
                    roster[f"{base}_logy"] = Pipeline([
                        ("prep", clone(base_pipe.steps[0][1])),
                        ("model", TransformedTargetRegressor(
                            regressor=clone(base_pipe.steps[-1][1]),
                            func=np.log1p, inverse_func=np.expm1)),
                    ])

    missing = [lib for lib, ok in OPTIONAL.items() if not ok and lib in
               ("xgboost", "lightgbm", "catboost")]
    if missing:
        log.warn("roster", f"Not installed, families skipped: {missing}")
    log.log("roster", f"Roster ({len(roster)} models): {list(roster)}")
    return roster


# --------------------------------------------------------------------------
# Early-stopping-aware fitting
# --------------------------------------------------------------------------

def supports_early_stopping(pipe) -> bool:
    est = _final_estimator(pipe)
    return est.__class__.__name__ in (
        "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor",
        "CatBoostClassifier", "CatBoostRegressor",
    )


def fit_with_early_stopping(pipe, X: pd.DataFrame, y: pd.Series, task: str,
                            rng: int, rounds: int = 50, max_trees: int = 2000,
                            chronological: bool = False, log=None):
    """Fit a GBM pipeline with a 10% early-stopping split, then refit the whole
    training set at the discovered tree count.

    ``chronological=True`` (time-aware runs, where rows arrive time-sorted)
    validates on the chronological tail instead of a shuffled slice — a random
    split would let the probe train on rows later than its validation data.
    A ``log`` (DecisionLog), when given, records probe failures instead of
    letting them pass silently.

    The returned pipeline is a *plain* fitted sklearn pipeline with a concrete
    ``n_estimators`` — nothing about early stopping leaks into the artifact.
    Returns (fitted_pipeline, best_n_trees or None).

    Callers run this sequentially (never inside joblib-parallel CV), so the
    model gets all cores here; roster pipelines keep 1 thread for parallel CV.
    """
    from .utils import set_model_threads

    if not supports_early_stopping(pipe):
        fitted = clone(pipe)
        fitted.fit(X, y)
        return fitted, None

    if chronological:
        n_val = max(1, min(int(len(X) * 0.1), len(X) - 1))
        X_fit, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_fit, y_val = y.iloc[:-n_val], y.iloc[-n_val:]
    else:
        stratify = y if task == "classification" else None
        try:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X, y, test_size=0.1, random_state=rng, stratify=stratify)
        except ValueError:  # stratification impossible (tiny/rare classes) -> plain split
            X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.1, random_state=rng)

    probe = clone(pipe)
    set_model_threads(probe, -1)
    prep = probe.steps[0][1]
    model = probe.steps[-1][1]
    Xf = prep.fit_transform(X_fit, y_fit)   # prep sees the raw target, as in Pipeline.fit
    Xv = prep.transform(X_val)

    # log1p-target variants wrap the GBM in TransformedTargetRegressor: the
    # probe early-stops the inner GBM directly, so fit and eval targets must
    # both live on the transformed scale (mirroring what TTR.fit does).
    y_es_fit, y_es_val = y_fit, y_val
    if isinstance(model, TransformedTargetRegressor):
        y_es_fit = pd.Series(model.func(np.asarray(y_fit, dtype=float)), index=y_fit.index)
        y_es_val = pd.Series(model.func(np.asarray(y_val, dtype=float)), index=y_val.index)
        model = model.regressor

    name = model.__class__.__name__
    best_n = None
    try:
        if name.startswith("XGB"):
            model.set_params(n_estimators=max_trees, early_stopping_rounds=rounds)
            model.fit(Xf, y_es_fit, eval_set=[(Xv, y_es_val)], verbose=False)
            best_n = int(model.best_iteration) + 1
        elif name.startswith("LGBM"):
            model.set_params(n_estimators=max_trees)
            model.fit(Xf, y_es_fit, eval_set=[(Xv, y_es_val)],
                      callbacks=[lightgbm.early_stopping(rounds, verbose=False)])
            best_n = int(model.best_iteration_)
        elif name.startswith("CatBoost"):
            model.set_params(iterations=max_trees)
            model.fit(Xf, y_es_fit, eval_set=(Xv, y_es_val), early_stopping_rounds=rounds)
            best_n = int(model.get_best_iteration()) + 1
    except Exception as e:
        best_n = None  # fall back to the configured tree count
        if log is not None:
            log.warn("tuning", f"{name}: early-stopping probe failed "
                               f"({type(e).__name__}: {e}) — using the configured "
                               "tree count instead")

    fitted = clone(pipe)
    set_model_threads(fitted, -1)
    if best_n:
        est = fitted.steps[-1][1]
        if isinstance(est, TransformedTargetRegressor):
            est = est.regressor   # freeze the tree count on the inner GBM
        # small headroom: the final fit sees ~11% more data than the probe fit
        n_final = max(10, int(best_n * 1.1))
        if est.__class__.__name__.startswith("CatBoost"):
            est.set_params(iterations=n_final)
        else:
            est.set_params(n_estimators=n_final)
        if hasattr(est, "early_stopping_rounds"):
            est.set_params(early_stopping_rounds=None)
    fitted.fit(X, y)
    return fitted, best_n
