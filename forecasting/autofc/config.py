"""Configuration for an automated forecasting run."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ForecastConfig:
    # ----- Required ---------------------------------------------------------
    horizon: int = 14                   # steps ahead to forecast (in the data's frequency)
    target: str = "y"
    date_col: str = "ds"
    series_col: Optional[str] = None    # None -> single series
    data_path: Optional[str] = None     # .csv/.parquet; or pass a DataFrame to load()

    # ----- Data contract ----------------------------------------------------
    exog_cols: Optional[list] = None    # None -> every other column is an exogenous
                                        # regressor; MUST be known for future dates
    freq: Optional[str] = None          # pandas offset alias; None -> inferred
    seasonal_period: Optional[int] = None  # None -> detected from seasonal strength
    country_holidays: Optional[str] = None  # e.g. "US": holiday indicators for the
                                            # ML models, SARIMAX exog, and Prophet
    hierarchy: Optional[dict] = None    # {parent: [children]}; absent parents are
                                        # synthesized by summing their children
    reconciliation: str = "auto"        # 'auto'|'none'|'bottom_up'|'ols'|
                                        # 'wls_struct'|'mint_shrink' (needs hierarchy)

    # ----- Guardrails -------------------------------------------------------
    max_gap_frac: float = 0.10          # more missing timestamps than this -> series dropped
    min_series_length: Optional[int] = None  # default: max(3*season, 2*horizon+10, 30)
    duplicate_policy: str = "mean"      # 'mean' aggregate | 'error'

    # ----- Evaluation -------------------------------------------------------
    n_backtests: int = 3                # rolling-origin folds
    interval_level: float = 0.90        # prediction-interval coverage target

    # ----- Roster -----------------------------------------------------------
    enable_sarimax: bool = True
    enable_ets: bool = True
    enable_theta: bool = True
    enable_prophet: bool = True
    enable_ml: bool = True              # global LightGBM/XGBoost/HistGB lag models
    enable_catboost: bool = False       # off by default: little gain over LGBM/XGB
                                        # on lag features for the runtime it costs
    ensemble: bool = True               # blend of the top-3 models joins the leaderboard
    champion_policy: str = "best_score"  # 'best_score': the leaderboard winner ships;
                                         # 'parsimonious': the deployment pick ships —
                                         # fewest fitted models within one standard error of
                                         # the best mean MASE (computed and logged
                                         # either way)
    sarimax_max_obs: int = 1000         # order search + fits use at most this tail
    sarimax_large_season: int = 60      # above this, seasonality via Fourier exog terms

    # ----- Output -----------------------------------------------------------
    artifact_dir: str = "forecast_artifacts"
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self):
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not 0.5 <= self.interval_level < 1:
            raise ValueError("interval_level must be in [0.5, 1)")
        if self.n_backtests < 1:
            raise ValueError("n_backtests must be >= 1")
        if self.duplicate_policy not in ("mean", "error"):
            raise ValueError("duplicate_policy must be 'mean' or 'error'")
        if self.champion_policy not in ("best_score", "parsimonious"):
            raise ValueError(f"champion_policy must be 'best_score' or 'parsimonious', "
                             f"got {self.champion_policy!r}")
        if self.reconciliation not in ("auto", "none", "bottom_up", "ols",
                                       "wls_struct", "mint_shrink"):
            raise ValueError(f"unknown reconciliation method: {self.reconciliation!r}")
        if self.hierarchy:
            for parent, children in self.hierarchy.items():
                if not isinstance(children, (list, tuple)) or not children:
                    raise ValueError(f"hierarchy[{parent!r}] must be a non-empty list "
                                     "of child series names")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def fast(cls, **overrides) -> "ForecastConfig":
        defaults: dict = dict(n_backtests=2, enable_prophet=False, sarimax_max_obs=400)
        defaults.update(overrides)
        return cls(**defaults)
