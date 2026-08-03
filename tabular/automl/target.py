"""Target analysis: task validation, metric selection, objective-family detection.

Distribution statistics that drive metric/objective choices are computed on the
*training partition only* — the holdout stays locked for everything except the
consistent label encoding needed to stratify the split itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import AutoMLConfig
from .utils import AutoMLError, DecisionLog, infer_task


class TargetAnalysis:
    task: str
    label_encoder: LabelEncoder | None = None
    classes: list | None = None
    n_classes: int = 0
    is_binary: bool = False
    imbalance_ratio: float = 1.0
    imbalanced: bool = False
    primary_metric: str = ""
    primary_scorer: str = ""
    secondary_scoring: dict | None = None
    objective_family: str = "standard"   # 'standard'|'poisson'|'tweedie'|'gamma'
    log_target_candidate: bool = False
    stats: dict | None = None


def encode_target(y_raw: pd.Series, config: AutoMLConfig, log: DecisionLog):
    """Resolve the task and encode labels. Returns (y, analysis, rows_to_keep_mask)."""
    ta = TargetAnalysis()
    ta.task = infer_task(y_raw) if config.task == "auto" else config.task
    if config.task == "auto":
        log.log("target", f"task='auto' -> inferred '{ta.task}' from the target "
                          f"({y_raw.nunique()} unique values, dtype {y_raw.dtype})")

    keep_mask = pd.Series(True, index=y_raw.index)

    if ta.task == "classification":
        counts = y_raw.value_counts()
        rare = counts[counts < config.min_class_members]
        if len(rare):
            if config.rare_class_policy == "error":
                raise AutoMLError(
                    f"Classes with fewer than {config.min_class_members} rows: "
                    f"{dict(rare)}. They cannot survive {config.cv_folds}-fold stratified CV. "
                    "Merge or drop them upstream, or set rare_class_policy='drop'."
                )
            keep_mask = ~y_raw.isin(rare.index)
            log.warn("target", f"Dropping {int((~keep_mask).sum())} rows of rare class(es) "
                               f"{list(rare.index)} (each below min_class_members="
                               f"{config.min_class_members}; they cannot survive stratified CV). "
                               "Set rare_class_policy='error' to stop instead.")
            y_raw = y_raw[keep_mask]
            counts = y_raw.value_counts()
        if y_raw.nunique() < 2:
            raise AutoMLError("Target has a single class after rare-class handling — nothing to learn.")

        ta.label_encoder = LabelEncoder()
        y = pd.Series(ta.label_encoder.fit_transform(y_raw), index=y_raw.index, name=y_raw.name)
        ta.classes = [str(c) for c in ta.label_encoder.classes_]
        ta.n_classes = len(ta.classes)
        ta.is_binary = ta.n_classes == 2
        ta.imbalance_ratio = float(counts.max() / counts.min())
        ta.imbalanced = ta.imbalance_ratio > 5
        ta.stats = {"class_counts": {str(k): int(v) for k, v in counts.items()},
                    "imbalance_ratio": round(ta.imbalance_ratio, 2)}
        return y, ta, keep_mask

    # ----- regression --------------------------------------------------------
    try:
        y = pd.to_numeric(y_raw, errors="raise").astype(float)
    except (ValueError, TypeError) as e:
        raise AutoMLError(
            f"Regression target '{y_raw.name}' is not numeric ({e}). "
            "If the target is categorical, run the classification template instead "
            "(or set task='classification')."
        ) from e
    if float(y.std()) == 0.0 or y.nunique() <= 1:
        raise AutoMLError("Regression target has zero variance — nothing to learn "
                          "(and every feature would look like leakage).")
    return y, ta, keep_mask


def analyze_target_distribution(y_train: pd.Series, ta: TargetAnalysis,
                                config: AutoMLConfig, log: DecisionLog) -> TargetAnalysis:
    """Metric + objective-family selection. Classification metrics are fixed by design;
    regression choices depend on the training target's shape."""
    if ta.task == "classification":
        ta.primary_metric, ta.primary_scorer = "LogLoss", "neg_log_loss"
        ta.secondary_scoring = {"roc_auc": "roc_auc" if ta.is_binary else "roc_auc_ovr",
                                "balanced_accuracy": "balanced_accuracy"}
        log.log("metric", "Primary metric: LogLoss (rewards calibrated probabilities)")
        if ta.imbalanced:
            if ta.is_binary:
                ta.secondary_scoring["pr_auc"] = "average_precision"
            log.log("metric", f"Imbalance ratio {ta.imbalance_ratio:.1f} > 5 -> class-weighted "
                              "model variants join the leaderboard and PR-AUC is reported. "
                              "Class weighting is NOT forced on every model: weighting distorts "
                              "probability calibration, which LogLoss exists to protect.")
        return ta

    y = np.asarray(y_train, dtype=float)
    skew = float(pd.Series(y).skew())
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    outlier_frac = float(((y < q1 - 3 * iqr) | (y > q3 + 3 * iqr)).mean()) if iqr > 0 else 0.0
    strictly_positive = bool((y > 0).all())
    non_negative = bool((y >= 0).all())
    is_integer = bool(np.allclose(y % 1, 0))
    zero_frac = float((y == 0).mean())

    ta.stats = {"mean": float(np.mean(y)), "median": float(np.median(y)),
                "std": float(np.std(y)), "skew": round(skew, 3),
                "outlier_frac": round(outlier_frac, 4), "zero_frac": round(zero_frac, 4),
                "strictly_positive": strictly_positive, "integer_valued": is_integer}

    if abs(skew) > 2 or outlier_frac > 0.01:
        ta.primary_metric, ta.primary_scorer = "MAE", "neg_mean_absolute_error"
        ta.secondary_scoring = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}
        log.log("metric", f"Skew={skew:.2f}, extreme outliers={outlier_frac:.2%} -> primary metric "
                          "MAE (robust). Gradient-boosting rosters switch to absolute-error "
                          "objectives so models optimize the metric they are ranked on.")
    else:
        ta.primary_metric, ta.primary_scorer = "RMSE", "neg_root_mean_squared_error"
        ta.secondary_scoring = {"mae": "neg_mean_absolute_error", "r2": "r2"}
        log.log("metric", f"Target reasonably symmetric (skew={skew:.2f}) -> primary metric RMSE")

    # Objective family: match the likelihood to the target's distribution.
    # Integer-valued alone doesn't make a count (prices in whole dollars are
    # integers too) — genuine counts have modest support.
    n_unique = int(pd.Series(y).nunique())
    count_like = (non_negative and is_integer
                  and n_unique <= min(100, max(10, int(0.05 * len(y)))))
    if count_like:
        ta.objective_family = "poisson"
        log.log("objective", f"Non-negative integer target with count-like support "
                             f"({n_unique} distinct values) -> Poisson-objective "
                             "variants join the roster")
    elif non_negative and not strictly_positive and zero_frac > 0.05 and skew > 1:
        ta.objective_family = "tweedie"
        log.log("objective", f"Zero-inflated non-negative target ({zero_frac:.0%} zeros) -> "
                             "Tweedie-objective variants join the roster")
    elif strictly_positive and skew > 2:
        ta.objective_family = "gamma"
        log.log("objective", "Strictly positive, highly skewed target -> Gamma-objective variants "
                             "join the roster")

    if strictly_positive and skew > 1:
        ta.log_target_candidate = True
        log.log("objective", "Strictly positive skewed target -> log1p-target model variants "
                             "join the leaderboard (TransformedTargetRegressor; scored on the "
                             "original scale, so directly comparable)")
    return ta
