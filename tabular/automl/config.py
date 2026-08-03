"""Configuration for an AutoML run.

Every automated decision the engine makes can be overridden here — automation
should never be a black box.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AutoMLConfig:
    # ----- Required ---------------------------------------------------------
    target: str = "target"
    task: str = "auto"                  # 'classification' | 'regression' | 'auto' (inferred)
    data_path: Optional[str] = None     # .csv or .parquet; may also pass a DataFrame to AutoML.load

    # ----- Columns ----------------------------------------------------------
    drop_columns: list = field(default_factory=list)    # excluded up front (IDs, keys, ...)
    force_keep_columns: list = field(default_factory=list)  # never auto-dropped

    # ----- Partitioning -----------------------------------------------------
    holdout_fraction: float = 0.20
    cv_folds: int = 5
    random_state: int = 42
    time_column: Optional[str] = None    # set -> out-of-time holdout + expanding-window CV
    group_column: Optional[str] = None   # set -> group-aware holdout + GroupKFold CV

    # ----- Data guardrails --------------------------------------------------
    max_missing_frac: float = 0.95       # drop features missing more than this
    near_constant_thresh: float = 0.995  # drop if one value covers >= this fraction
    id_unique_frac: float = 0.999        # drop if unique-ratio >= this (ID-like)
    high_cardinality_thresh: int = 50    # above this, categoricals get target encoding
    text_min_avg_len: int = 60           # strings at least this long on average -> TF-IDF text
    max_text_columns: int = 3            # vectorize at most this many text columns
    drop_duplicate_rows: bool = True     # duplicates straddling train/holdout inflate scores
    date_features: bool = True           # decompose datetime columns instead of dropping
    text_features: bool = True           # TF-IDF text columns instead of dropping
    rare_class_policy: str = "drop"      # 'drop' rows of classes below min_class_members | 'error'
    min_class_members: Optional[int] = None  # default: max(2 * cv_folds, 10)

    # ----- Leakage scan -----------------------------------------------------
    leakage_spearman_thresh: float = 0.98   # |Spearman(feature, y)| flags numeric leaks (regression)
    leakage_model_thresh_r2: float = 0.95   # single-feature depth-6 tree CV R^2 flag
    leakage_model_thresh_auc: float = 0.98  # single-feature depth-6 tree CV AUC flag
    leakage_sample_cap: int = 10_000
    correlation_thresh: float = 0.95        # |Spearman| between features -> prune the weaker twin

    # ----- Leaderboard ------------------------------------------------------
    stage1_sample_cap: int = 20_000     # stage-1 screening runs on at most this many rows
    n_finalists: int = 3                # models advancing to Optuna tuning
    optuna_trials: int = 30             # trials per finalist
    optuna_timeout: int = 600           # seconds per finalist (whichever comes first)
    ensemble: bool = True               # stack/blend the tuned finalists
    final_cv_seed_offset: int = 1000    # fresh folds for the de-biased final re-score
    champion_policy: str = "best_score"  # 'best_score': export the leaderboard winner;
                                         # 'parsimonious': export the deployment pick —
                                         # the simplest candidate within one standard error
                                         # of the best score (the pick is computed,
                                         # logged, and shipped in metadata either way)
    n_jobs: int = -1

    # ----- Modeling behavior ------------------------------------------------
    early_stopping: bool = True         # gradient boosting fits its own tree count
    calibrate: bool = True              # repair champion calibration when it helps (classification)
    tune_threshold: bool = True         # binary: ship a tuned operating point inside the artifact
    threshold_objective: str = "f1"     # 'f1' | 'balanced_accuracy' — swap for asymmetric costs

    # ----- Output -----------------------------------------------------------
    refit_on_full_data: bool = True     # after holdout scoring, refit champion on 100% of rows
    artifact_dir: str = "automl_artifacts"
    bootstrap_samples: int = 1000

    def __post_init__(self):
        if self.min_class_members is None:
            self.min_class_members = max(2 * self.cv_folds, 10)
        if self.task not in ("auto", "classification", "regression"):
            raise ValueError(f"task must be 'auto', 'classification' or 'regression', got {self.task!r}")
        if not 0.05 <= self.holdout_fraction <= 0.5:
            raise ValueError("holdout_fraction should be between 0.05 and 0.5")
        if self.time_column and self.group_column:
            raise ValueError("Set either time_column or group_column, not both")
        if self.rare_class_policy not in ("drop", "error"):
            raise ValueError(f"rare_class_policy must be 'drop' or 'error', "
                             f"got {self.rare_class_policy!r}")
        if self.threshold_objective not in ("f1", "balanced_accuracy"):
            raise ValueError(f"threshold_objective must be 'f1' or 'balanced_accuracy', "
                             f"got {self.threshold_objective!r}")
        if self.champion_policy not in ("best_score", "parsimonious"):
            raise ValueError(f"champion_policy must be 'best_score' or 'parsimonious', "
                             f"got {self.champion_policy!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def fast(cls, **overrides) -> "AutoMLConfig":
        """A preset for first runs and smoke tests: minutes, not tens of minutes."""
        defaults: dict = dict(cv_folds=3, optuna_trials=8, optuna_timeout=120,
                              n_finalists=2, bootstrap_samples=200,
                              stage1_sample_cap=10_000)
        defaults.update(overrides)
        return cls(**defaults)
