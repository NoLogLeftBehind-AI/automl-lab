"""Explainability: permutation importance, partial dependence, SHAP.

Caveat stated up front (and in the report): permutation importance splits credit
between correlated features. The engine prunes pairs above the correlation
threshold, but features below it still share importance — read clusters, not
single rows, when features are related.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from .config import AutoMLConfig
from .models import OPTIONAL
from .target import TargetAnalysis
from .utils import DecisionLog

BLUE = "#4C72B0"


def compute_permutation_importance(model, X_holdout, y_holdout, ta: TargetAnalysis,
                                   config: AutoMLConfig) -> pd.DataFrame:
    perm = permutation_importance(model, X_holdout, y_holdout,
                                  scoring=ta.primary_scorer, n_repeats=5,
                                  random_state=config.random_state, n_jobs=config.n_jobs)
    return (pd.DataFrame({"feature": X_holdout.columns,
                          "importance": perm.importances_mean,
                          "std": perm.importances_std})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def fig_importance(imp: pd.DataFrame, champion_name: str):
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(top))))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color=BLUE)
    ax.set_title(f"Permutation importance — {champion_name} (holdout)")
    fig.tight_layout()
    return fig


def fig_partial_dependence(model, X_holdout, imp: pd.DataFrame, num_cols: list,
                           ta: TargetAnalysis, config: AutoMLConfig, champion_name: str,
                           log=None):
    top_numeric = [f for f in imp["feature"] if f in num_cols][:6]
    if not top_numeric:
        return None
    kw = {} if (ta.task == "regression" or ta.is_binary) else {"target": 1}
    try:
        X_pdp = X_holdout.copy()
        X_pdp[top_numeric] = X_pdp[top_numeric].astype(float)  # PDP grids are float-valued
        disp = PartialDependenceDisplay.from_estimator(
            model, X_pdp, features=top_numeric, n_cols=3,
            n_jobs=config.n_jobs, random_state=config.random_state, **kw)
        n_rows_fig = int(np.ceil(len(top_numeric) / 3))
        disp.figure_.set_size_inches(12, 3.2 * n_rows_fig)
        disp.figure_.suptitle(f"Partial dependence — {champion_name} (holdout)", y=1.02)
        disp.figure_.tight_layout()
        return disp.figure_
    except Exception as e:
        if log is not None:
            log.log("explain", f"Partial dependence skipped "
                               f"({type(e).__name__}: {e})")
        return None


def shap_explanations(model, X_holdout, ta: TargetAnalysis, config: AutoMLConfig,
                      log: DecisionLog, max_rows: int = 300):
    """Returns (fig, example_frame) or (None, None). Works on the innermost pipeline —
    threshold/calibration wrappers pass probabilities through untouched, but SHAP
    needs the raw model + transformed features."""
    if not OPTIONAL["shap"]:
        log.log("explain", "shap not installed -> prediction explanations skipped")
        return None, None
    import shap
    pipe = model
    # unwrap FixedThresholdClassifier / CalibratedClassifierCV to reach the pipeline
    for attr in ("estimator_", "estimator"):
        while hasattr(pipe, attr) and not hasattr(pipe, "steps"):
            pipe = getattr(pipe, attr)
    if not hasattr(pipe, "steps"):
        log.log("explain", "SHAP skipped: champion is not a plain pipeline "
                           "(calibrated ensembles explain poorly)")
        return None, None
    final_model = pipe.steps[-1][1]
    # shap 0.51's C extension segfaults (uncatchable) parsing multiclass CatBoost trees
    if final_model.__class__.__name__.startswith("CatBoost") and not ta.is_binary:
        log.log("explain", "SHAP skipped: known shap/CatBoost multiclass incompatibility "
                           "(segfault in shap's tree parser)")
        return None, None
    try:
        prep = pipe[:-1]
        Xs = X_holdout.sample(min(max_rows, len(X_holdout)), random_state=config.random_state)
        Xt_arr = prep.transform(Xs)
        Xt_arr = Xt_arr.toarray() if hasattr(Xt_arr, "toarray") else np.asarray(Xt_arr)
        try:
            names = list(prep.get_feature_names_out())
        except Exception:
            names = [f"f{i}" for i in range(Xt_arr.shape[1])]
        Xt = pd.DataFrame(Xt_arr, columns=names, index=Xs.index)

        explainer = shap.Explainer(final_model, Xt)
        sv = explainer(Xt)
        vals = np.asarray(sv.values)

        mean_abs = np.abs(vals).mean(axis=(0, 2) if vals.ndim == 3 else 0)
        order = np.argsort(mean_abs)[::-1][:15]
        top_names = np.array(Xt.columns)[order]

        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(order))))
        ax.barh(top_names[::-1], mean_abs[order][::-1], color=BLUE)
        ax.set_title("Mean |SHAP| (holdout sample, transformed feature space)")
        fig.tight_layout()

        row = vals[0] if vals.ndim == 2 else vals[0, :, -1]
        top_k = np.argsort(np.abs(row))[::-1][:8]
        example = pd.DataFrame({"feature": np.array(Xt.columns)[top_k],
                                "value": Xt.iloc[0].to_numpy()[top_k],
                                "shap_contribution": np.asarray(row)[top_k]}).round(4)
        return fig, example
    except Exception as e:
        log.log("explain", f"SHAP skipped ({type(e).__name__}: {e}) — some model/shap "
                           "combinations need a dedicated explainer")
        return None, None
