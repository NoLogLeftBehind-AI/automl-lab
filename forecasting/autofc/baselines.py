"""Baseline forecasters: naive, seasonal naive, drift.

These are on the leaderboard for honesty, not decoration — on many real series
seasonal-naive is embarrassingly hard to beat, and a leaderboard that can't
show that is lying. MASE is defined relative to seasonal-naive for the same
reason. Interval formulas follow Hyndman & Athanasopoulos (fpp3, §5.5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class ForecasterBase:
    """Uniform interface: fit(panel, ctx) then predict(horizon, future_exog).

    ``ctx`` carries run context: {'season': int, 'confirmed': [int],
    'freq': str, 'level': float, 'config': ForecastConfig}.
    predict returns {series_id: DataFrame(index=future dates, columns
    yhat / lo / hi)} — lo/hi may be NaN for models without native intervals
    (the engine attaches backtest-residual intervals for those).
    """

    name = "base"
    native_intervals = True

    #: PyPI distributions this forecaster needs *at inference time*, beyond the
    #: core pandas/numpy/scipy/joblib/scikit-learn stack. Every third-party
    #: import in this package is lazy (``import autofc`` pulls none of them), so
    #: the exported artifact pins the union over its bundled models rather than
    #: whatever happened to be installed while training. Baselines are pure
    #: numpy/scipy and add nothing.
    requires: tuple = ()

    def fit(self, panel, ctx):
        raise NotImplementedError

    def predict(self, horizon, future_exog=None):
        raise NotImplementedError

    def _z(self, ctx):
        return float(norm.ppf(0.5 + ctx["level"] / 2))


class NaiveForecaster(ForecasterBase):
    name = "Naive"

    def fit(self, panel, ctx):
        self._ctx = ctx
        self._state = {}
        for sid, f in panel.frames.items():
            y = f["y"].to_numpy()
            sigma = float(np.std(np.diff(y))) if len(y) > 1 else 0.0
            self._state[sid] = {"last": float(y[-1]), "sigma": sigma,
                                "end": f.index[-1]}
        self._freq = panel.freq
        return self

    def predict(self, horizon, future_exog=None):
        z = self._z(self._ctx)
        out = {}
        for sid, s in self._state.items():
            idx = pd.date_range(s["end"], periods=horizon + 1, freq=self._freq)[1:]
            h = np.arange(1, horizon + 1)
            yhat = np.full(horizon, s["last"])
            half = z * s["sigma"] * np.sqrt(h)
            out[sid] = pd.DataFrame({"yhat": yhat, "lo": yhat - half, "hi": yhat + half},
                                    index=idx)
        return out


class SeasonalNaiveForecaster(ForecasterBase):
    name = "SeasonalNaive"

    def fit(self, panel, ctx):
        self._ctx = ctx
        self._m = max(1, ctx["season"])
        self._state = {}
        for sid, f in panel.frames.items():
            y = f["y"].to_numpy()
            m = self._m if len(y) > self._m else 1
            resid = y[m:] - y[:-m]
            self._state[sid] = {"tail": y[-m:], "m": m,
                                "sigma": float(np.std(resid)) if len(resid) else 0.0,
                                "end": f.index[-1]}
        self._freq = panel.freq
        return self

    def predict(self, horizon, future_exog=None):
        z = self._z(self._ctx)
        out = {}
        for sid, s in self._state.items():
            idx = pd.date_range(s["end"], periods=horizon + 1, freq=self._freq)[1:]
            h = np.arange(1, horizon + 1)
            yhat = np.array([s["tail"][(i - 1) % s["m"]] for i in h])
            k = np.floor((h - 1) / s["m"]) + 1          # completed seasonal cycles ahead
            half = z * s["sigma"] * np.sqrt(k)
            out[sid] = pd.DataFrame({"yhat": yhat, "lo": yhat - half, "hi": yhat + half},
                                    index=idx)
        return out


class DriftForecaster(ForecasterBase):
    name = "Drift"

    def fit(self, panel, ctx):
        self._ctx = ctx
        self._state = {}
        for sid, f in panel.frames.items():
            y = f["y"].to_numpy()
            n = len(y)
            slope = (y[-1] - y[0]) / (n - 1) if n > 1 else 0.0
            sigma = float(np.std(np.diff(y) - slope)) if n > 1 else 0.0
            self._state[sid] = {"last": float(y[-1]), "slope": slope, "n": n,
                                "sigma": sigma, "end": f.index[-1]}
        self._freq = panel.freq
        return self

    def predict(self, horizon, future_exog=None):
        z = self._z(self._ctx)
        out = {}
        for sid, s in self._state.items():
            idx = pd.date_range(s["end"], periods=horizon + 1, freq=self._freq)[1:]
            h = np.arange(1, horizon + 1)
            yhat = s["last"] + s["slope"] * h
            half = z * s["sigma"] * np.sqrt(h * (1 + h / max(s["n"] - 1, 1)))
            out[sid] = pd.DataFrame({"yhat": yhat, "lo": yhat - half, "hi": yhat + half},
                                    index=idx)
        return out
