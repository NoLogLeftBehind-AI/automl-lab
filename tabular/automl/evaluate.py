"""Champion finalization (calibration, operating point) and holdout evaluation.

Discipline rules enforced here:

- The tuned decision threshold is chosen on **out-of-fold training predictions**,
  never the holdout, and it ships *inside* the artifact via
  ``FixedThresholdClassifier`` — ``predict()`` applies it, so consumers cannot
  accidentally use the untuned 0.5 cutoff. Multiclass gets no threshold (None).
- Calibration is not just diagnosed but repaired: if an isotonic/sigmoid
  calibrator improves out-of-fold LogLoss by a meaningful margin, the champion
  is wrapped in ``CalibratedClassifierCV``. Calibration repair and threshold
  tuning currently cover **binary classification only**; multiclass ships
  argmax predictions with unrepaired probabilities (and the log says so).
- Every holdout metric passes explicit ``labels`` so a rare class missing from
  the holdout cannot crash the run.
- All metrics reported for the holdout come from a model that never saw the
  holdout; the optional refit-on-100%-of-rows happens strictly afterwards and
  is labeled as such in the metadata.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, median_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import FixedThresholdClassifier, cross_val_predict

from .config import AutoMLConfig
from .target import TargetAnalysis
from .utils import DecisionLog

BLUE, ORANGE = "#4C72B0", "#DD8452"


# --------------------------------------------------------------------------
# Champion finalization (classification)
# --------------------------------------------------------------------------

def finalize_classifier(champion_pipe, X_train: pd.DataFrame, y_train: pd.Series,
                        cv, ta: TargetAnalysis, config: AutoMLConfig, log: DecisionLog,
                        groups=None):
    """Returns (final_estimator_unfitted, threshold_or_None, calibration_info,
    oof_proba_or_None). OOF probabilities are computed only when a binary run
    needs them (calibration or threshold tuning) — they cost k champion refits."""
    estimator = clone(champion_pipe)
    calibration_info: dict = {"applied": False}
    threshold = None
    oof = None

    if ta.is_binary and (config.calibrate or config.tune_threshold):
        try:
            oof = cross_val_predict(clone(champion_pipe), X_train, y_train, cv=cv,
                                    method="predict_proba", n_jobs=config.n_jobs,
                                    groups=groups)
            yt = np.asarray(y_train)
        except ValueError:
            # splitters that don't cover every row (e.g. TimeSeriesSplit) can't produce
            # full OOF predictions — fall back to the last 20% of train as a tuning slice
            cut = int(len(X_train) * 0.8)
            probe = clone(champion_pipe).fit(X_train.iloc[:cut], y_train.iloc[:cut])
            oof = probe.predict_proba(X_train.iloc[cut:])
            yt = np.asarray(y_train.iloc[cut:])
            log.log("threshold", "CV splitter cannot produce out-of-fold predictions -> "
                                 "calibration/threshold tuned on the last 20% of train instead")
        p = oof[:, 1]

        # --- calibration check on OOF probabilities (half fit, half judge) ----
        # The diagnostic uses the same method that would ship (isotonic needs
        # data; sigmoid/Platt is the safe family below ~5000 rows), so the
        # apply/skip decision and the shipped calibrator can never disagree —
        # and the decision threshold below is tuned on the shipped scale.
        if config.calibrate:
            rs = np.random.RandomState(config.random_state)
            half = rs.permutation(len(p)) < len(p) // 2
            raw_ll = log_loss(yt[~half], np.clip(p[~half], 1e-15, 1 - 1e-15), labels=[0, 1])
            method = "isotonic" if len(X_train) > 5000 else "sigmoid"
            if method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(p[half], yt[half])
                cal_predict = iso.predict
            else:
                from sklearn.linear_model import LogisticRegression
                platt = LogisticRegression(C=1e6)   # plain Platt scaling, no shrinkage
                platt.fit(p[half].reshape(-1, 1), yt[half])

                def cal_predict(q, _m=platt):
                    return _m.predict_proba(np.asarray(q, dtype=float).reshape(-1, 1))[:, 1]
            cal_p = np.clip(cal_predict(p[~half]), 1e-6, 1 - 1e-6)
            cal_ll = log_loss(yt[~half], cal_p, labels=[0, 1])
            if cal_ll < raw_ll * 0.99:
                from sklearn.calibration import CalibratedClassifierCV
                estimator = CalibratedClassifierCV(estimator, method=method, cv=config.cv_folds)
                calibration_info = {"applied": True, "method": method,
                                    "oof_logloss_raw": float(raw_ll),
                                    "oof_logloss_calibrated": float(cal_ll)}
                log.log("calibration", f"OOF LogLoss improves {raw_ll:.4f} -> {cal_ll:.4f} "
                                       f"with {method} calibration -> champion wrapped in "
                                       "CalibratedClassifierCV")
                p = np.clip(cal_predict(p), 1e-6, 1 - 1e-6)  # threshold on the shipped scale
            else:
                log.log("calibration", f"Champion already well calibrated "
                                       f"(OOF LogLoss {raw_ll:.4f}, {method} would give "
                                       f"{cal_ll:.4f}) -> no calibration wrapper")

        # --- operating point on OOF (never the holdout) ----------------------
        if config.tune_threshold:
            grid = np.linspace(0.05, 0.95, 91)
            objective = (f1_score if config.threshold_objective == "f1"
                         else balanced_accuracy_score)
            vals = np.array([objective(yt, (p >= t).astype(int)) for t in grid])
            threshold = float(grid[int(vals.argmax())])
            estimator = FixedThresholdClassifier(estimator, threshold=threshold,
                                                 response_method="predict_proba")
            log.log("threshold", f"Decision threshold {threshold:.2f} maximizes "
                                 f"{config.threshold_objective} on out-of-fold train "
                                 "predictions; it ships INSIDE the artifact — predict() "
                                 "applies it automatically")
    elif not ta.is_binary:
        log.log("threshold", "Multiclass target -> no scalar threshold (predict() uses "
                             "argmax) and no calibration repair (binary-only); "
                             "metadata records optimal_threshold=null")
    return estimator, threshold, calibration_info, oof


# --------------------------------------------------------------------------
# Holdout evaluation
# --------------------------------------------------------------------------

def evaluate_classification(model, X_holdout, y_holdout, ta: TargetAnalysis,
                            config: AutoMLConfig, log: DecisionLog):
    labels = np.arange(ta.n_classes)
    proba = model.predict_proba(X_holdout)
    pred = model.predict(X_holdout)
    yh = np.asarray(y_holdout)

    metrics = {"LogLoss": float(log_loss(yh, proba, labels=labels)),
               "balanced_accuracy": float(balanced_accuracy_score(yh, pred))}
    try:
        if ta.is_binary:
            metrics["ROC_AUC"] = float(roc_auc_score(yh, proba[:, 1]))
        else:
            metrics["ROC_AUC"] = float(roc_auc_score(yh, proba, multi_class="ovr",
                                                     labels=labels, average="macro"))
    except ValueError as e:  # e.g. a class entirely absent from the holdout
        metrics["ROC_AUC"] = float("nan")
        log.warn("evaluate", f"ROC AUC unavailable on this holdout: {e}")
    if ta.is_binary:
        metrics["F1"] = float(f1_score(yh, pred, zero_division=0))
        metrics["precision"] = float(precision_score(yh, pred, zero_division=0))
        metrics["recall"] = float(recall_score(yh, pred, zero_division=0))
        metrics["Brier"] = float(brier_score_loss(yh, proba[:, 1]))

    class_names = ta.classes or []
    report = classification_report(yh, pred, labels=labels, target_names=class_names,
                                   zero_division=0)
    missing = [class_names[i] for i in labels if (yh == i).sum() == 0]
    if missing:
        log.warn("evaluate", f"Class(es) absent from the holdout: {missing} — "
                             "per-class metrics for them are undefined")
    return metrics, proba, pred, report


def evaluate_regression(model, X_holdout, y_holdout):
    pred = np.asarray(model.predict(X_holdout), dtype=float)
    yh = np.asarray(y_holdout, dtype=float)
    metrics = {"RMSE": float(np.sqrt(mean_squared_error(yh, pred))),
               "MAE": float(mean_absolute_error(yh, pred)),
               "median_AE": float(median_absolute_error(yh, pred)),
               "R2": float(r2_score(yh, pred))}
    return metrics, pred


def bootstrap_ci(y_holdout, pred, proba, ta: TargetAnalysis, config: AutoMLConfig,
                 point_metrics: dict, log: DecisionLog) -> pd.DataFrame:
    """95% bootstrap intervals for the headline metrics; resamples that lose a class
    are counted and reported, not silently swallowed."""
    yh = np.asarray(y_holdout)
    n = len(yh)
    if n < 20:
        log.warn("evaluate", "Holdout too small for meaningful bootstrap CIs — skipped")
        return pd.DataFrame()
    rs = np.random.RandomState(config.random_state)
    labels = np.arange(ta.n_classes) if ta.task == "classification" else None
    boot: dict[str, list] = {}
    skipped_auc = 0

    for _ in range(config.bootstrap_samples):
        b = rs.choice(n, size=n, replace=True)
        if ta.task == "classification":
            boot.setdefault("LogLoss", []).append(
                log_loss(yh[b], proba[b], labels=labels))
            boot.setdefault("balanced_accuracy", []).append(
                balanced_accuracy_score(yh[b], np.asarray(pred)[b]))
            present = np.unique(yh[b])
            if len(present) == ta.n_classes:
                if ta.is_binary:
                    boot.setdefault("ROC_AUC", []).append(roc_auc_score(yh[b], proba[b, 1]))
                else:
                    boot.setdefault("ROC_AUC", []).append(
                        roc_auc_score(yh[b], proba[b], multi_class="ovr",
                                      labels=labels, average="macro"))
            else:
                skipped_auc += 1
        else:
            p = np.asarray(pred, dtype=float)[b]
            yb = yh[b].astype(float)
            boot.setdefault("RMSE", []).append(float(np.sqrt(mean_squared_error(yb, p))))
            boot.setdefault("MAE", []).append(float(mean_absolute_error(yb, p)))
            boot.setdefault("R2", []).append(float(r2_score(yb, p)))

    if skipped_auc:
        log.warn("evaluate", f"{skipped_auc}/{config.bootstrap_samples} bootstrap resamples "
                             "lost a class and were excluded from the ROC AUC interval "
                             "(kept for the other metrics)")
    rows = {}
    for m, vals in boot.items():
        if len(vals) < 100:
            continue
        rows[m] = {"point": point_metrics.get(m, float(np.mean(vals))),
                   "ci_2.5%": float(np.percentile(vals, 2.5)),
                   "ci_97.5%": float(np.percentile(vals, 97.5)),
                   "n_resamples": len(vals)}
    return pd.DataFrame(rows).T.round(4)


# --------------------------------------------------------------------------
# Evaluation figures
# --------------------------------------------------------------------------

def fig_confusion_roc(yh, pred, proba, ta: TargetAnalysis, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = np.arange(ta.n_classes)
    cm = confusion_matrix(yh, pred, labels=labels)
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion matrix (holdout)")
    axes[0].set_xticks(range(ta.n_classes), ta.classes, rotation=45, ha="right")
    axes[0].set_yticks(range(ta.n_classes), ta.classes)
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("actual")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(ta.n_classes):
        for j in range(ta.n_classes):
            axes[0].text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
    if ta.is_binary and np.isfinite(metrics.get("ROC_AUC", np.nan)):
        fpr, tpr, _ = roc_curve(yh, proba[:, 1])
        axes[1].plot(fpr, tpr, color=BLUE, label=f"AUC = {metrics['ROC_AUC']:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[1].set_title("ROC curve (holdout)")
        axes[1].legend()
        axes[1].set_xlabel("FPR")
        axes[1].set_ylabel("TPR")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    return fig


def fig_regression_diagnostics(yh, pred):
    yh = np.asarray(yh, dtype=float)
    pred = np.asarray(pred, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(yh, pred, s=8, alpha=0.5, color=BLUE)
    lims = [min(yh.min(), pred.min()), max(yh.max(), pred.max())]
    axes[0].plot(lims, lims, "k--", lw=0.8)
    axes[0].set_title("Predicted vs actual (holdout)")
    axes[0].set_xlabel("actual")
    axes[0].set_ylabel("predicted")
    residuals = yh - pred
    axes[1].scatter(pred, residuals, s=8, alpha=0.5, color=BLUE)
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].set_title("Residuals vs predicted")
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("residual")
    fig.tight_layout()
    return fig


def fig_lift(yh, score_vals, task: str, n_bins: int = 10):
    """Lift chart; bins adapt so small holdouts don't produce empty deciles."""
    yh = np.asarray(yh, dtype=float)
    score_vals = np.asarray(score_vals, dtype=float)
    n_bins = int(max(2, min(n_bins, len(yh) // 5))) if len(yh) >= 10 else 0
    if n_bins < 2:
        return None
    order = np.argsort(score_vals)
    bins = np.array_split(order, n_bins)
    mean_pred = [float(score_vals[b].mean()) for b in bins]
    mean_act = [float(yh[b].mean()) for b in bins]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, n_bins + 1)
    ylab = "positive rate" if task == "classification" else "target"
    ax.plot(x, mean_act, "o-", color=ORANGE, label="mean actual")
    ax.plot(x, mean_pred, "s--", color=BLUE, label="mean predicted")
    ax.set_xticks(x)
    ax.set_xlabel(f"prediction bin (low → high, {n_bins} bins)")
    ax.set_ylabel(ylab)
    ax.set_title("Lift chart (holdout)")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_calibration(yh, proba_pos, threshold, config: AutoMLConfig):
    yh = np.asarray(yh, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        frac_pos, mean_pred = calibration_curve(yh, proba_pos, n_bins=10, strategy="quantile")
    except ValueError:
        plt.close(fig)
        return None
    brier = brier_score_loss(yh, proba_pos)
    ax.plot(mean_pred, frac_pos, "o-", color=BLUE)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    if threshold is not None:
        ax.axvline(threshold, color=ORANGE, ls="--", lw=0.8,
                   label=f"operating point = {threshold:.2f}")
        ax.legend()
    ax.set_title(f"Calibration (holdout) — Brier {brier:.4f}")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    fig.tight_layout()
    return fig
