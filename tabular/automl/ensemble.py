"""Blenders: stacking and soft-voting over the tuned finalists.

Blends of the best models routinely top AutoML leaderboards; these candidates
join the final re-score and win only when they genuinely win.
Both are standard sklearn estimators, so they serialize like any pipeline.
"""
from __future__ import annotations

from sklearn.base import clone
from sklearn.ensemble import (StackingClassifier, StackingRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.linear_model import LogisticRegression, RidgeCV

from .config import AutoMLConfig
from .target import TargetAnalysis


def build_blenders(finalists: dict, ta: TargetAnalysis, config: AutoMLConfig) -> dict:
    """finalists: {name: unfitted pipeline with tuned params}. Returns blender candidates."""
    if len(finalists) < 2:
        return {}
    estimators = [(name, clone(pipe)) for name, pipe in finalists.items()]
    blenders = {}
    if ta.task == "classification":
        blenders[f"Stacked({len(estimators)})"] = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=config.random_state),
            cv=config.cv_folds, stack_method="predict_proba", n_jobs=1)
        blenders[f"Voting({len(estimators)})"] = VotingClassifier(
            estimators=[(n, clone(p)) for n, p in estimators], voting="soft", n_jobs=1)
    else:
        blenders[f"Stacked({len(estimators)})"] = StackingRegressor(
            estimators=estimators, final_estimator=RidgeCV(),
            cv=config.cv_folds, n_jobs=1)
        blenders[f"Voting({len(estimators)})"] = VotingRegressor(
            estimators=[(n, clone(p)) for n, p in estimators], n_jobs=1)
    return blenders
