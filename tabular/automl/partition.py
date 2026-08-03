"""Train/holdout partitioning and CV strategy.

Random shuffles silently inflate scores on temporal or grouped data, so the
partitioner is pluggable: set config.time_column for an out-of-time holdout
with expanding-window CV, or config.group_column to keep all rows of an entity
on one side of every split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     StratifiedGroupKFold, StratifiedKFold,
                                     TimeSeriesSplit, train_test_split)

from .config import AutoMLConfig
from .target import TargetAnalysis
from .utils import AutoMLError, DecisionLog, quantile_bins_for_stratification


class Partition:
    X_train: pd.DataFrame
    X_holdout: pd.DataFrame
    y_train: pd.Series
    y_holdout: pd.Series
    cv = None                     # splitter for model selection
    groups_train: np.ndarray | None = None   # aligned with X_train, for group-aware CV
    strategy: str = "random"


def make_partition(X: pd.DataFrame, y: pd.Series, ta: TargetAnalysis,
                   config: AutoMLConfig, log: DecisionLog) -> Partition:
    p = Partition()
    rng = config.random_state

    if config.time_column:
        col = config.time_column
        if col not in X.columns and not any(c.startswith(f"{col}__") for c in X.columns):
            raise AutoMLError(f"time_column '{col}' not found in the feature set.")
        # order by the raw column if present, else by its decomposed epoch_days
        order_col = col if col in X.columns else f"{col}__epoch_days"
        vals = X[order_col].to_numpy()
        n_nat = int(pd.isna(vals).sum())
        if n_nat:
            log.warn("partition", f"{n_nat} row(s) have missing/unparseable '{order_col}' "
                                  "values; they sort to the end of the timeline (and "
                                  "therefore into the holdout) — fix the timestamps "
                                  "upstream if that is not intended")
        # np.argsort places NaN/NaT last — unlike Series.argsort, whose pandas-2
        # behavior on missing values yields corrupt positional indices
        order = np.argsort(vals, kind="stable")
        n_holdout = max(1, int(len(X) * config.holdout_fraction))
        train_idx, hold_idx = order[:-n_holdout], order[-n_holdout:]
        p.X_train, p.X_holdout = X.iloc[train_idx], X.iloc[hold_idx]
        p.y_train, p.y_holdout = y.iloc[train_idx], y.iloc[hold_idx]
        p.cv = TimeSeriesSplit(n_splits=config.cv_folds)
        p.strategy = "time"
        log.log("partition", f"Out-of-time partition on '{order_col}': last "
                             f"{config.holdout_fraction:.0%} of rows held out; "
                             f"expanding-window CV ({config.cv_folds} splits). "
                             "Rows must be scored in time order for this to be valid.")
        return p

    if config.group_column:
        col = config.group_column
        if col not in X.columns:
            raise AutoMLError(f"group_column '{col}' not found in the feature set. "
                              "If it was auto-dropped (e.g. ID-like), add it to force_keep_columns.")
        groups = X[col]
        gss = GroupShuffleSplit(n_splits=1, test_size=config.holdout_fraction, random_state=rng)
        train_idx, hold_idx = next(gss.split(X, y, groups=groups))
        p.X_train, p.X_holdout = X.iloc[train_idx], X.iloc[hold_idx]
        p.y_train, p.y_holdout = y.iloc[train_idx], y.iloc[hold_idx]
        p.groups_train = groups.iloc[train_idx].to_numpy()
        # the grouping column itself must not be a feature (it's an entity key)
        p.X_train = p.X_train.drop(columns=[col])
        p.X_holdout = p.X_holdout.drop(columns=[col])
        if ta.task == "classification":
            p.cv = StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=rng)
        else:
            p.cv = GroupKFold(n_splits=config.cv_folds)
        p.strategy = "group"
        log.log("partition", f"Group-aware partition on '{col}': "
                             f"{groups.nunique()} groups, no group straddles train/holdout "
                             f"or any CV fold; '{col}' removed from the feature set.")
        return p

    # ----- default: shuffled split, stratified whenever possible -------------
    stratify = None
    if ta.task == "classification":
        stratify = y
    else:
        bins = quantile_bins_for_stratification(y)
        if bins is not None:
            stratify = bins
            log.log("partition", "Regression holdout stratified on target deciles "
                                 "(protects the tails on skewed targets)")
    p.X_train, p.X_holdout, p.y_train, p.y_holdout = train_test_split(
        X, y, test_size=config.holdout_fraction, stratify=stratify, random_state=rng)
    if ta.task == "classification":
        p.cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=rng)
    else:
        p.cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=rng)
    p.strategy = "random"
    log.log("partition", f"Train {len(p.X_train):,} rows / holdout {len(p.X_holdout):,} rows "
                         f"(locked until final evaluation); {config.cv_folds}-fold "
                         f"{'stratified ' if ta.task == 'classification' else ''}CV")
    return p


def fresh_cv(ta: TargetAnalysis, config: AutoMLConfig, seed_offset: int = 0):
    """CV splitter with different fold assignments — used for the de-biased final re-score."""
    seed = config.random_state + seed_offset
    if config.time_column:
        return TimeSeriesSplit(n_splits=config.cv_folds)
    if config.group_column:
        if ta.task == "classification":
            return StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
        return GroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
    if ta.task == "classification":
        return StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
    return KFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
