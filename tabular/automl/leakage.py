"""Target-leakage scan and correlated-feature pruning.

Two complementary detectors, because each has a blind spot:

1. |Spearman(feature, y)| — numeric features on regression targets only. Rank
   correlation has no model-capacity ceiling, so it flags noisy monotonic
   near-copies whose cross-validated single-feature tree score can sit below
   the model-scan threshold. (History: the depth-3 tree this scan originally
   used capped near R²≈0.96 and waved a literal copy of a continuous target
   past the 0.98 threshold — the regression test remembers.)
2. A single-feature depth-6 decision tree, cross-validated, for every feature —
   catches categorical proxies and non-monotonic leaks that correlation misses.

Univariate scans still can't see combinatorial leakage (price_per_sqft only
leaks price together with sqft); a post-modeling guardrail flags suspiciously
high holdout scores for exactly that reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import AutoMLConfig
from .target import TargetAnalysis
from .utils import DecisionLog


def _single_feature_pipeline(series: pd.Series, task: str, rng: int) -> Pipeline:
    if pd.api.types.is_numeric_dtype(series):
        prep = SimpleImputer(strategy="median")
    else:
        prep = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
    tree_cls = DecisionTreeRegressor if task == "regression" else DecisionTreeClassifier
    return Pipeline([("prep", prep), ("model", tree_cls(max_depth=6, random_state=rng))])


def scan_leakage(X: pd.DataFrame, y: pd.Series, ta: TargetAnalysis,
                 config: AutoMLConfig, log: DecisionLog,
                 text_columns: list) -> tuple[pd.DataFrame, list, dict]:
    """Returns (report, leaky_columns, target_association_scores)."""
    cap = min(len(X), config.leakage_sample_cap)
    rs = np.random.RandomState(config.random_state)
    idx = rs.choice(len(X), size=cap, replace=False)
    X_s, y_s = X.iloc[idx], y.iloc[idx]

    scoring = ("r2" if ta.task == "regression"
               else ("roc_auc" if ta.is_binary else "roc_auc_ovr"))
    model_thresh = (config.leakage_model_thresh_r2 if ta.task == "regression"
                    else config.leakage_model_thresh_auc)

    rows, assoc = [], {}
    scannable = [c for c in X.columns if c not in text_columns]
    for col in scannable:
        model_score, sp = np.nan, np.nan
        try:
            scores = cross_val_score(_single_feature_pipeline(X_s[col], ta.task, config.random_state),
                                     X_s[[col]], y_s, cv=3, scoring=scoring,
                                     n_jobs=config.n_jobs, error_score=np.nan)
            model_score = float(np.nanmean(scores))
        except Exception as e:
            log.warn("leakage", f"could not model-score '{col}': {e}")
        if ta.task == "regression" and pd.api.types.is_numeric_dtype(X_s[col]):
            mask = X_s[col].notna()
            if mask.sum() >= 10 and X_s.loc[mask, col].nunique() > 1:
                sp = float(abs(spearmanr(X_s.loc[mask, col], y_s[mask]).statistic))
        assoc[col] = np.nanmax([model_score, 0.0]) if not np.isnan(model_score) else np.nan

        flag_model = (not np.isnan(model_score)) and model_score >= model_thresh
        flag_sp = (not np.isnan(sp)) and sp >= config.leakage_spearman_thresh
        rows.append({"feature": col, "single_feature_score": model_score,
                     "abs_spearman_vs_target": sp,
                     "leakage_flag": bool(flag_model or flag_sp)})

    report = (pd.DataFrame(rows).set_index("feature")
              .sort_values("single_feature_score", ascending=False))

    protected = set(config.force_keep_columns)
    leaky = [c for c in report[report["leakage_flag"]].index if c not in protected]
    for col in leaky:
        r = report.loc[col]
        reasons = []
        if not np.isnan(r["single_feature_score"]) and r["single_feature_score"] >= model_thresh:
            reasons.append(f"single-feature {scoring}={r['single_feature_score']:.3f} >= {model_thresh}")
        if not np.isnan(r["abs_spearman_vs_target"]) and \
                r["abs_spearman_vs_target"] >= config.leakage_spearman_thresh:
            reasons.append(f"|Spearman vs target|={r['abs_spearman_vs_target']:.3f} "
                           f">= {config.leakage_spearman_thresh}")
        log.log("leakage", f"Dropping '{col}': {'; '.join(reasons)}")
    flagged_protected = [c for c in report[report["leakage_flag"]].index if c in protected]
    for col in flagged_protected:
        log.warn("leakage", f"'{col}' looks like leakage but is in force_keep_columns — kept")

    target_l = str(y.name).lower() if y.name else ""
    for col in scannable:
        if target_l and target_l in col.lower() and col not in leaky:
            log.warn("leakage", f"'{col}' contains the target name — verify it is known "
                                "at prediction time")
    if not leaky:
        log.log("leakage", "No leakage flags raised")
    return report, leaky, assoc


def prune_correlated(X: pd.DataFrame, assoc: dict, config: AutoMLConfig,
                     log: DecisionLog) -> list:
    """Greedy pairwise pruning: for each pair with |Spearman| above the threshold,
    drop the member with the weaker target association (NaN-safe tiebreak)."""
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    drops: list = []
    if len(num_cols) < 2:
        return drops
    corr = X[num_cols].corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = (upper.stack().dropna().reset_index()
             .set_axis(["f1", "f2", "abs_spearman"], axis=1)
             .sort_values("abs_spearman", ascending=False))
    pairs = pairs[pairs["abs_spearman"] >= config.correlation_thresh]

    protected = set(config.force_keep_columns)

    def score(col):  # NaN-safe: an unscoreable feature loses the tiebreak
        s = assoc.get(col, np.nan)
        return -np.inf if (s is None or (isinstance(s, float) and np.isnan(s))) else s

    for _, row in pairs.iterrows():
        f1, f2 = row["f1"], row["f2"]
        if f1 in drops or f2 in drops:
            continue
        weaker, stronger = (f1, f2) if score(f1) <= score(f2) else (f2, f1)
        if weaker in protected:
            weaker, stronger = stronger, weaker
        if weaker in protected:
            continue
        drops.append(weaker)
        log.log("correlation", f"Dropping '{weaker}' (|ρ|={row['abs_spearman']:.3f} with "
                               f"'{stronger}', weaker target association)")
    if not drops:
        log.log("correlation", "No feature pairs above the correlation threshold")
    return drops
