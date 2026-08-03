"""Stage-2 hyperparameter search with Optuna.

Fixes over a naive `study.best_value` approach:

- The **default configuration is always a candidate**: the roster defaults are
  scored with the exact same objective before the search, and tuning can only
  replace them if it *beats* them. (Optuna search spaces cannot even represent
  some defaults — e.g. RandomForest's ``max_depth=None`` — so without this
  guard, tuning could silently ship a strictly worse model.)
- **Early stopping inside the objective** for XGBoost/LightGBM/CatBoost:
  ``n_estimators`` is not searched; each fold picks its tree count against a
  validation split, which both improves accuracy and multiplies the effective
  trial budget.
- Trial failures are caught and pruned instead of killing the study.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_val_score

from .config import AutoMLConfig
from .models import OPTIONAL, family_of, fit_with_early_stopping, supports_early_stopping
from .utils import DecisionLog


def search_space(trial, family: str) -> dict:
    if family == "LogisticRegression":
        return {"model__C": trial.suggest_float("model__C", 1e-3, 100, log=True)}
    if family == "Ridge":
        return {"model__alpha": trial.suggest_float("model__alpha", 1e-3, 100, log=True)}
    if family == "ElasticNet":
        return {"model__alpha": trial.suggest_float("model__alpha", 1e-4, 10, log=True),
                "model__l1_ratio": trial.suggest_float("model__l1_ratio", 0.05, 0.95)}
    if family in ("PoissonGLM", "TweedieGLM", "GammaGLM"):
        return {"model__alpha": trial.suggest_float("model__alpha", 1e-4, 10, log=True)}
    if family in ("RandomForest", "ExtraTrees"):
        return {"model__n_estimators": trial.suggest_int("model__n_estimators", 200, 800),
                # None (unbounded) is the sklearn default and often the best choice —
                # a purely numeric range could never propose it
                "model__max_depth": trial.suggest_categorical(
                    "model__max_depth", [None, 6, 10, 14, 18, 24]),
                "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 20),
                "model__max_features": trial.suggest_categorical(
                    "model__max_features", ["sqrt", 0.3, 0.5, 0.7, 1.0])}
    if family == "HistGradientBoosting":
        return {"model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True),
                "model__max_leaf_nodes": trial.suggest_int("model__max_leaf_nodes", 15, 127),
                "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 5, 60),
                "model__l2_regularization": trial.suggest_float(
                    "model__l2_regularization", 1e-6, 1.0, log=True)}
    if family == "XGBoost":
        return {"model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True),
                "model__max_depth": trial.suggest_int("model__max_depth", 3, 10),
                "model__subsample": trial.suggest_float("model__subsample", 0.6, 1.0),
                "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.6, 1.0),
                "model__min_child_weight": trial.suggest_int("model__min_child_weight", 1, 10),
                "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 1.0, log=True),
                "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-8, 10.0, log=True)}
    if family == "LightGBM":
        return {"model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True),
                "model__num_leaves": trial.suggest_int("model__num_leaves", 15, 255),
                "model__min_child_samples": trial.suggest_int("model__min_child_samples", 5, 60),
                # subsample requires subsample_freq > 0 or LightGBM silently ignores it
                "model__subsample": trial.suggest_float("model__subsample", 0.6, 1.0),
                "model__subsample_freq": trial.suggest_int("model__subsample_freq", 1, 5),
                "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.6, 1.0),
                "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 1.0, log=True),
                "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-8, 10.0, log=True)}
    if family == "CatBoost":
        return {"model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True),
                "model__depth": trial.suggest_int("model__depth", 4, 10),
                "model__l2_leaf_reg": trial.suggest_float("model__l2_leaf_reg", 1.0, 30.0, log=True)}
    return {}


def _es_param_names(pipe) -> list:
    """Params the search must not set when early stopping picks the tree count."""
    if not supports_early_stopping(pipe):
        return []
    return ["model__n_estimators", "model__iterations"]


def remap_params_for_pipeline(params: dict, pipe) -> dict:
    """Adapt search-space keys to the pipeline's actual structure.

    Log-target roster variants wrap the estimator in TransformedTargetRegressor,
    so ``model__learning_rate`` must become ``model__regressor__learning_rate``
    — without this every tuning trial dies with 'Invalid parameter'.
    """
    from sklearn.compose import TransformedTargetRegressor
    if isinstance(pipe.steps[-1][1], TransformedTargetRegressor):
        return {k.replace("model__", "model__regressor__", 1): v
                for k, v in params.items()}
    return params


def cv_score(pipe, X: pd.DataFrame, y: pd.Series, cv, scorer_name: str,
             config: AutoMLConfig, ta, groups=None, use_es: bool = True) -> float:
    """Mean CV score (sklearn convention: higher is better).

    ES-capable GBMs are scored with a manual fold loop so each fold's tree count
    comes from early stopping; everything else uses plain cross_val_score.
    """
    if use_es and config.early_stopping and supports_early_stopping(pipe):
        scorer = check_scoring(pipe, scoring=scorer_name)
        scores = []
        for tr_idx, te_idx in cv.split(X, y, groups=groups):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            # time-aware runs feed time-sorted slices: the probe must validate
            # on the chronological tail, exactly like the finalize path
            fitted, _ = fit_with_early_stopping(pipe, X_tr, y_tr, ta.task,
                                                config.random_state,
                                                chronological=bool(config.time_column))
            scores.append(scorer(fitted, X.iloc[te_idx], y.iloc[te_idx]))
        return float(np.mean(scores))
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scorer_name,
                             n_jobs=config.n_jobs, groups=groups, error_score=np.nan)
    return float(np.nanmean(scores)) if not np.all(np.isnan(scores)) else float("-inf")


def tune_finalist(name: str, base_pipe, X: pd.DataFrame, y: pd.Series, cv,
                  ta, config: AutoMLConfig, log: DecisionLog, groups=None) -> dict:
    """Tune one finalist. Returns {'params', 'score', 'n_trials', 'improved'} where
    'score' is directly comparable to the default configuration's score (same folds,
    same data, same early-stopping behavior)."""
    scorer_name = ta.primary_scorer
    t0 = time.time()

    default_score = cv_score(base_pipe, X, y, cv, scorer_name, config, ta, groups)

    if not OPTIONAL["optuna"]:
        log.log("tuning", f"{name}: optuna not installed -> keeping defaults")
        return {"params": {}, "score": default_score, "n_trials": 0, "improved": False}

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    family = family_of(name)
    banned = set(_es_param_names(base_pipe)) if config.early_stopping else set()

    def objective(trial):
        params = {k: v for k, v in search_space(trial, family).items() if k not in banned}
        params = remap_params_for_pipeline(params, base_pipe)
        pipe = clone(base_pipe).set_params(**params)
        score = cv_score(pipe, X, y, cv, scorer_name, config, ta, groups)
        if not np.isfinite(score):
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=config.random_state))
    study.optimize(objective, n_trials=config.optuna_trials,
                   timeout=config.optuna_timeout, show_progress_bar=False,
                   catch=(Exception,))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_value = study.best_value if completed else float("-inf")

    if best_value > default_score:
        best_params = remap_params_for_pipeline(study.best_params, base_pipe)
        result = {"params": best_params, "score": float(best_value),
                  "n_trials": len(study.trials), "improved": True}
        log.log("tuning", f"{name}: tuned beats defaults "
                          f"({_disp(best_value, scorer_name):.4f} vs "
                          f"{_disp(default_score, scorer_name):.4f}, "
                          f"{len(completed)}/{len(study.trials)} trials, {time.time()-t0:.0f}s)")
    else:
        result = {"params": {}, "score": float(default_score),
                  "n_trials": len(study.trials), "improved": False}
        log.log("tuning", f"{name}: defaults kept — no trial beat them "
                          f"(best trial {_disp(best_value, scorer_name):.4f} vs default "
                          f"{_disp(default_score, scorer_name):.4f}, "
                          f"{len(completed)}/{len(study.trials)} trials, {time.time()-t0:.0f}s)")
    return result


def _disp(score: float, scorer_name: str) -> float:
    """Convert sklearn's higher-is-better score to display convention."""
    if not np.isfinite(score):
        return float("nan")
    return -score if scorer_name.startswith("neg_") else score
