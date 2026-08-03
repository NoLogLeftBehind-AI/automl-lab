"""Statistical forecasters: auto-SARIMAX, ETS, Theta, Prophet.

All fit per series (parallelized with joblib). Design notes:

- **auto-SARIMAX** does a compact AIC search over a small order grid — the
  standard Box-Jenkins ranges cover most series, and a bounded grid keeps the
  runtime honest. ``d`` comes from an ADF test; seasonal ``D`` is set when the
  seasonal difference is materially flatter than the first difference
  (``mean|y_t - y_{t-m}| < 0.8 * mean|diff(y)|``) — an MAE-ratio rule, not an
  ACF test. Long seasonalities (yearly in daily data) are handled the
  standard way: Fourier terms as exogenous regressors instead of an
  intractable seasonal order. Orders are selected once per series and cached,
  so backtest refits only re-estimate coefficients — model *structure* is
  chosen once, honestly, and only parameters update per fold.
- **ETS / Theta** use statsmodels' state-space implementations with native
  prediction intervals; a per-series failure falls back to seasonal-naive for
  that series (logged) instead of killing the model family.
- **Prophet** is serialized via its own JSON functions (pickling fitted Stan
  backends is unreliable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .baselines import ForecasterBase, SeasonalNaiveForecaster
from .data import _offset_nanos
from .utils import base_freq_alias, silence_fit_warnings

try:
    from prophet import Prophet
    from prophet.serialize import model_from_json, model_to_json
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


def fourier_terms(idx: pd.DatetimeIndex, period_steps: float, K: int,
                  origin: pd.Timestamp, freq: str) -> pd.DataFrame:
    """Deterministic Fourier features (harmonic regression) for long seasonality.

    Step positions come from the offset's fixed duration (dividing a
    TimedeltaIndex by ``to_offset('D')`` raises on pandas 3, where Day is no
    longer a Tick); anchored offsets fall back to the index's median spacing.
    """
    nanos = _offset_nanos(freq)
    if nanos is None:
        diffs = np.diff(idx.asi8)
        nanos = float(np.median(diffs)) if len(diffs) else 0.0
    if not nanos:
        raise ValueError(f"cannot derive a fixed step length for freq {freq!r}")
    t = (idx - origin).asi8 / float(nanos)
    cols = {}
    for k in range(1, K + 1):
        cols[f"_fourier_sin{k}_{int(period_steps)}"] = np.sin(2 * np.pi * k * t / period_steps)
        cols[f"_fourier_cos{k}_{int(period_steps)}"] = np.cos(2 * np.pi * k * t / period_steps)
    return pd.DataFrame(cols, index=idx)


class AutoSarimaxForecaster(ForecasterBase):
    requires = ("statsmodels",)
    name = "AutoSARIMAX"

    def __init__(self):
        self._order_cache: dict[str, tuple] = {}   # persists across backtest refits

    # ---- order selection -------------------------------------------------
    def _select_orders(self, y: np.ndarray, m: int) -> tuple:
        from statsmodels.tsa.stattools import adfuller

        with silence_fit_warnings():
            try:
                d = 0 if adfuller(y, autolag="AIC")[1] < 0.05 else 1
            except Exception:
                d = 1
        D = 0
        if m > 1 and len(y) > 3 * m:
            seas_mae = np.mean(np.abs(y[m:] - y[:-m]))
            naive_mae = np.mean(np.abs(np.diff(y)))
            D = 1 if seas_mae < 0.8 * naive_mae else 0
        return d, D

    def _grid(self, m: int):
        pq = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]
        PQ = [(0, 0), (1, 0), (0, 1), (1, 1)] if m > 1 else [(0, 0)]
        return pq, PQ

    def _deterministic_exog(self, idx, origin, freq, det_cfg):
        """Engine-generated exog computable for any dates: Fourier terms for
        long seasonalities + holiday indicators."""
        parts = [fourier_terms(idx, p, det_cfg["K"], origin, freq)
                 for p in det_cfg["periods"]]
        if det_cfg.get("holiday_country"):
            from .calendars import holiday_flags
            parts.append(holiday_flags(idx, det_cfg["holiday_country"]))
        return pd.concat(parts, axis=1) if parts else None

    def _fit_one(self, sid, frame, ctx, det_cfg, cached_order):
        """Runs in a joblib worker: everything needed comes in as arguments and
        everything learned (orders included) goes back in the return value —
        worker-side mutation of ``self`` would be silently lost."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        cfg = ctx["config"]
        m = ctx["season"] if 1 < ctx["season"] <= cfg.sarimax_large_season else 0
        tail = frame.iloc[-cfg.sarimax_max_obs:]
        y = tail["y"]

        exog = tail[ctx["exog_cols"]] if ctx["exog_cols"] else None
        det = self._deterministic_exog(tail.index, frame.index[0], ctx["freq"], det_cfg)
        if det is not None:
            exog = pd.concat([exog, det], axis=1) if exog is not None else det

        order = cached_order
        if order is None:
            d, D = self._select_orders(y.to_numpy(), m)
            best, best_aic = None, np.inf
            pq, PQ = self._grid(m)
            for p, q in pq:
                for P, Q in PQ:
                    try:
                        with silence_fit_warnings():
                            res = SARIMAX(y, exog=exog, order=(p, d, q),
                                          seasonal_order=(P, D, Q, m) if m else (0, 0, 0, 0),
                                          enforce_stationarity=False,
                                          enforce_invertibility=False).fit(disp=False)
                        if res.aic < best_aic:
                            best, best_aic = (p, d, q, P, D, Q), res.aic
                    except Exception:
                        continue
            order = best or (1, 1, 1, 0, 0, 0)

        p, d, q, P, D, Q = order
        with silence_fit_warnings():
            res = SARIMAX(y, exog=exog, order=(p, d, q),
                          seasonal_order=(P, D, Q, m) if m else (0, 0, 0, 0),
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
        return sid, {"result": res, "end": frame.index[-1], "origin": frame.index[0],
                     "order": order, "has_user_exog": bool(ctx["exog_cols"])}

    def fit(self, panel, ctx):
        self._ctx = dict(ctx, exog_cols=panel.exog_cols)
        self._freq = panel.freq
        cfg = ctx["config"]
        self._det = {"periods": [p for p in ctx["confirmed"]
                                 if p > cfg.sarimax_large_season],
                     "K": 3, "holiday_country": ctx.get("holiday_country")}
        fitted = Parallel(n_jobs=cfg.n_jobs)(
            delayed(self._fit_one)(sid, f, self._ctx, self._det,
                                   self._order_cache.get(sid))
            for sid, f in panel.frames.items())
        self._state = dict(fitted)
        for sid, s in self._state.items():
            self._order_cache[sid] = s["order"]   # reused by backtest refits
        return self

    def predict(self, horizon, future_exog=None):
        alpha = 1 - self._ctx["level"]
        out = {}
        for sid, s in self._state.items():
            idx = pd.date_range(s["end"], periods=horizon + 1, freq=self._freq)[1:]
            exog = None
            if s["has_user_exog"]:
                if future_exog is None or sid not in future_exog:
                    raise ValueError(f"Series '{sid}' was trained with exogenous "
                                     "regressors — provide future_exog at predict time")
                exog = future_exog[sid].reindex(idx)[self._ctx["exog_cols"]]
            # phase continuity: same origin the training fit used
            det = self._deterministic_exog(idx, s["origin"], self._freq, self._det)
            if det is not None:
                exog = pd.concat([exog, det], axis=1) if exog is not None else det
            with silence_fit_warnings():
                fc = s["result"].get_forecast(steps=horizon, exog=exog)
                ci = fc.conf_int(alpha=alpha)
            out[sid] = pd.DataFrame({"yhat": np.asarray(fc.predicted_mean),
                                     "lo": ci.iloc[:, 0].to_numpy(),
                                     "hi": ci.iloc[:, 1].to_numpy()}, index=idx)
        return out


class _PerSeriesStatForecaster(ForecasterBase):
    """Shared machinery for ETS/Theta: per-series fit with seasonal-naive fallback."""
    requires = ("statsmodels",)

    def _fit_series(self, sid, frame, ctx):
        raise NotImplementedError

    def _predict_series(self, state, horizon, alpha, idx):
        raise NotImplementedError

    def fit(self, panel, ctx):
        self._ctx = ctx
        self._freq = panel.freq
        results = Parallel(n_jobs=ctx["config"].n_jobs)(
            delayed(self._safe_fit)(sid, f, ctx) for sid, f in panel.frames.items())
        self._state = {sid: s for sid, s in results}
        self._fallbacks = [sid for sid, s in self._state.items() if s.get("fallback")]
        if self._fallbacks:
            fb = SeasonalNaiveForecaster()
            sub = type(panel)()
            sub.freq, sub.exog_cols = panel.freq, panel.exog_cols
            sub.frames = {sid: panel.frames[sid] for sid in self._fallbacks}
            self._fallback_model = fb.fit(sub, ctx)
        return self

    def _safe_fit(self, sid, frame, ctx):
        try:
            with silence_fit_warnings():
                return sid, self._fit_series(sid, frame, ctx)
        except Exception as e:
            return sid, {"fallback": True, "error": f"{type(e).__name__}: {e}"}

    def predict(self, horizon, future_exog=None):
        alpha = 1 - self._ctx["level"]
        out = {}
        fallback_preds = (self._fallback_model.predict(horizon)
                          if self._fallbacks else {})
        for sid, s in self._state.items():
            if s.get("fallback"):
                out[sid] = fallback_preds[sid]
                continue
            idx = pd.date_range(s["end"], periods=horizon + 1, freq=self._freq)[1:]
            with silence_fit_warnings():
                out[sid] = self._predict_series(s, horizon, alpha, idx)
        return out


class ETSForecaster(_PerSeriesStatForecaster):
    name = "ETS"

    def _fit_series(self, sid, frame, ctx):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        y = frame["y"]
        m = ctx["season"]
        seasonal = "add" if (m > 1 and len(y) >= 2 * m + 10) else None
        model = ETSModel(y, error="add", trend="add", damped_trend=True,
                         seasonal=seasonal, seasonal_periods=m if seasonal else None)
        res = model.fit(disp=False)
        return {"result": res, "end": frame.index[-1], "n": len(y)}

    def _predict_series(self, s, horizon, alpha, idx):
        pred = s["result"].get_prediction(start=s["n"], end=s["n"] + horizon - 1)
        sf = pred.summary_frame(alpha=alpha)
        return pd.DataFrame({"yhat": sf["mean"].to_numpy(),
                             "lo": sf["pi_lower"].to_numpy(),
                             "hi": sf["pi_upper"].to_numpy()}, index=idx)


class ThetaForecaster(_PerSeriesStatForecaster):
    name = "Theta"

    def _fit_series(self, sid, frame, ctx):
        from statsmodels.tsa.forecasting.theta import ThetaModel

        y = frame["y"]
        m = ctx["season"]
        use_season = m > 1 and len(y) >= 2 * m + 10
        model = ThetaModel(y, period=m if use_season else 1,
                           deseasonalize=use_season)
        res = model.fit()
        return {"result": res, "end": frame.index[-1]}

    def _predict_series(self, s, horizon, alpha, idx):
        mean = s["result"].forecast(horizon)
        try:
            ci = s["result"].prediction_intervals(horizon, alpha=alpha)
            lo, hi = ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy()
        except Exception:
            lo = hi = np.full(horizon, np.nan)
        return pd.DataFrame({"yhat": mean.to_numpy(), "lo": lo, "hi": hi}, index=idx)


class ProphetForecaster(ForecasterBase):
    requires = ("prophet",)
    name = "Prophet"

    def _fit_one(self, sid, frame, ctx):
        from .utils import quiet_prophet_logs
        quiet_prophet_logs()
        df = frame.reset_index().rename(columns={frame.index.name or "index": "ds"})
        df = df.rename(columns={df.columns[0]: "ds"})[["ds", "y"] + ctx["exog_cols"]]
        # exact family match (the seasonality module's rule): prefix matching
        # would give minute data yearly-only seasonality via 'MIN' ~ 'M'
        fam = base_freq_alias(ctx["freq"] or "D")
        m = Prophet(interval_width=ctx["level"],
                    weekly_seasonality=fam in ("min", "h", "d", "b"),
                    yearly_seasonality=(len(frame) >= 400 if fam in ("d", "b")
                                        else fam in ("w", "m", "me", "ms")),
                    daily_seasonality=fam in ("min", "h"))
        if ctx.get("holiday_country"):
            m.add_country_holidays(country_name=ctx["holiday_country"])
        for c in ctx["exog_cols"]:
            m.add_regressor(c)
        with silence_fit_warnings():
            m.fit(df)
        return sid, {"model": m, "end": frame.index[-1]}

    def fit(self, panel, ctx):
        self._ctx = dict(ctx, exog_cols=panel.exog_cols)
        self._freq = panel.freq
        # Prophet holds the GIL through cmdstan anyway; loky processes work fine
        fitted = Parallel(n_jobs=ctx["config"].n_jobs)(
            delayed(self._fit_one)(sid, f, self._ctx) for sid, f in panel.frames.items())
        self._state = dict(fitted)
        return self

    def predict(self, horizon, future_exog=None):
        out = {}
        for sid, s in self._state.items():
            idx = pd.date_range(s["end"], periods=horizon + 1, freq=self._freq)[1:]
            future = pd.DataFrame({"ds": idx})
            for c in self._ctx["exog_cols"]:
                if future_exog is None or sid not in future_exog:
                    raise ValueError(f"Series '{sid}' was trained with exogenous "
                                     "regressors — provide future_exog at predict time")
                future[c] = future_exog[sid].reindex(idx)[c].to_numpy()
            fc = s["model"].predict(future)
            out[sid] = pd.DataFrame({"yhat": fc["yhat"].to_numpy(),
                                     "lo": fc["yhat_lower"].to_numpy(),
                                     "hi": fc["yhat_upper"].to_numpy()}, index=idx)
        return out

    # ---- pickle support: Prophet's Stan backend doesn't pickle reliably ----
    def __getstate__(self):
        state = self.__dict__.copy()
        if "_state" in state:
            state["_state"] = {sid: {"model_json": model_to_json(s["model"]),
                                     "end": s["end"]}
                               for sid, s in state["_state"].items()}
        return state

    def __setstate__(self, state):
        if "_state" in state:
            state["_state"] = {sid: {"model": model_from_json(s["model_json"]),
                                     "end": s["end"]}
                               for sid, s in state["_state"].items()}
        self.__dict__.update(state)
