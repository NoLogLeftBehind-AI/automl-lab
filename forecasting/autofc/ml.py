"""Global machine-learning forecasters: one gradient-boosting model trained
across ALL series on engineered lag / rolling / calendar features, forecasting
recursively one step at a time.

This is the approach that wins modern forecasting competitions (M5) when there
are many related series: the model borrows strength across the panel, and the
series identifier is just another categorical feature. On a single series it
degrades gracefully to an autoregressive GBM.

Leakage discipline: every rolling statistic is computed on values shifted by
one step, so no feature at time t can see y_t. ML models have no native
prediction intervals — the engine attaches backtest-residual quantiles and the
report labels them as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import ForecasterBase
from .utils import base_freq_alias

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def feature_plan(season: int, confirmed: list,
                 holiday_country: str | None = None) -> dict:
    """Which lags/windows/calendar features to build, given the detected
    seasonality and calendar configuration."""
    lags = sorted(set([1, 2, 3] + ([season, 2 * season] if season > 1 else [7])))
    windows = sorted(set([max(season, 3), max(2 * season, 7)])) if season > 1 else [7, 28]
    long_periods = [p for p in confirmed if p > 60]
    return {"lags": lags, "windows": windows, "long_periods": long_periods,
            "holiday_country": holiday_country}


def _calendar_features(idx: pd.DatetimeIndex, plan: dict, freq: str) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["dow"] = idx.dayofweek.astype(float)
    f["month"] = idx.month.astype(float)
    f["day"] = idx.day.astype(float)
    if base_freq_alias(freq) in ("h", "min"):
        f["hour"] = idx.hour.astype(float)
    # long confirmed cycles: a yearly one gets a smooth day-of-year position;
    # other long periods (e.g. weekly-in-hours) are served by the seasonal
    # lags above rather than being mislabeled as yearly
    if any(300 <= p <= 400 for p in plan["long_periods"]):
        doy = idx.dayofyear.astype(float)
        f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if plan.get("holiday_country"):
        from .calendars import holiday_flags
        f = pd.concat([f, holiday_flags(idx, plan["holiday_country"])], axis=1)
    return f


def build_features(frame: pd.DataFrame, plan: dict, freq: str,
                   exog_cols: list) -> pd.DataFrame:
    """Feature matrix for one series. y_lag*/roll* only ever see y up to t-1."""
    y = frame["y"]
    out = _calendar_features(frame.index, plan, freq)
    for lag in plan["lags"]:
        out[f"y_lag{lag}"] = y.shift(lag)
    shifted = y.shift(1)
    for w in plan["windows"]:
        out[f"roll_mean{w}"] = shifted.rolling(w).mean()
        out[f"roll_std{w}"] = shifted.rolling(w).std()
    for c in exog_cols:
        out[c] = frame[c]
    return out


class GlobalMLForecaster(ForecasterBase):
    native_intervals = False

    def __init__(self, name: str, make_model, requires: tuple = ()):
        self.name = name
        self._make_model = make_model
        # inference-time dependency of the wrapped regressor (HistGB is
        # scikit-learn and needs nothing extra)
        self.requires = requires

    def fit(self, panel, ctx):
        self._ctx = dict(ctx, exog_cols=panel.exog_cols)
        self._freq = panel.freq
        self._plan = feature_plan(ctx["season"], ctx["confirmed"],
                                  ctx.get("holiday_country"))

        X_parts, y_parts = [], []
        self._series_code = {sid: i for i, sid in enumerate(panel.frames)}
        self._history = {}
        for sid, frame in panel.frames.items():
            feats = build_features(frame, self._plan, panel.freq, panel.exog_cols)
            feats["series_code"] = float(self._series_code[sid])
            valid = feats.dropna().index
            X_parts.append(feats.loc[valid])
            y_parts.append(frame.loc[valid, "y"])
            keep = max(self._plan["lags"] + self._plan["windows"]) + 1
            self._history[sid] = {"y": frame["y"].iloc[-keep:].copy(),
                                  "end": frame.index[-1]}
        X = pd.concat(X_parts)
        y = pd.concat(y_parts)
        self._columns = list(X.columns)
        self._model = self._make_model(self._ctx["config"])
        self._model.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, horizon, future_exog=None):
        out = {}
        for sid, h in self._history.items():
            hist = h["y"].copy()
            idx = pd.date_range(h["end"], periods=horizon + 1, freq=self._freq)[1:]
            preds = []
            for step_date in idx:
                # history + one empty slot for the step being predicted
                frame = pd.DataFrame(
                    {"y": np.append(hist.to_numpy(), np.nan)},
                    index=hist.index.append(pd.DatetimeIndex([step_date])))
                for c in self._ctx["exog_cols"]:
                    if future_exog is None or sid not in future_exog:
                        raise ValueError(f"Series '{sid}' was trained with exogenous "
                                         "regressors — provide future_exog at predict time")
                    fe = future_exog[sid]
                    frame[c] = np.nan
                    frame.loc[step_date, c] = float(fe.loc[step_date, c])
                feats = build_features(frame, self._plan, self._freq,
                                       self._ctx["exog_cols"]).iloc[[-1]]
                feats["series_code"] = float(self._series_code[sid])
                yhat = float(self._model.predict(feats[self._columns].to_numpy())[0])
                preds.append(yhat)
                hist = pd.concat([hist, pd.Series([yhat], index=[step_date])])
            out[sid] = pd.DataFrame({"yhat": preds, "lo": np.nan, "hi": np.nan},
                                    index=idx)
        return out


# module-level factories (not lambdas) so fitted forecasters pickle cleanly
def _make_lgbm(cfg):
    return LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                         random_state=cfg.random_state, n_jobs=cfg.n_jobs, verbose=-1)


def _make_xgb(cfg):
    return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                        tree_method="hist", random_state=cfg.random_state,
                        n_jobs=cfg.n_jobs, verbosity=0)


def _make_histgb(cfg):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=500, random_state=cfg.random_state)


def _make_catboost(cfg):
    return CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                             random_state=cfg.random_state, verbose=0,
                             thread_count=cfg.n_jobs, allow_writing_files=False)


def make_ml_forecasters(config) -> list:
    """The ML roster, honoring availability and config switches."""
    models: list[ForecasterBase] = []
    if not config.enable_ml:
        return models
    if HAS_LGBM:
        models.append(GlobalMLForecaster("LightGBM_global", _make_lgbm, ("lightgbm",)))
    if HAS_XGB:
        models.append(GlobalMLForecaster("XGBoost_global", _make_xgb, ("xgboost",)))
    models.append(GlobalMLForecaster("HistGB_global", _make_histgb))
    if config.enable_catboost and HAS_CATBOOST:
        models.append(GlobalMLForecaster("CatBoost_global", _make_catboost, ("catboost",)))
    return models
