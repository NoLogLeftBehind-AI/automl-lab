"""Rolling-origin backtesting and forecasting metrics.

The time-series analogue of cross-validation: K cutoffs stepping back from the
end of the panel by one horizon each; for every cutoff each model trains on
data up to the cutoff and forecasts the next H steps. No shuffling anywhere —
training data always strictly precedes the scored window.

Metrics (aggregated as the unweighted mean over series, so a large series
cannot hide bad forecasts on a small one):

- **MASE** — MAE scaled by the in-sample one-step seasonal-naive MAE of the
  training window; scale-free and comparable across series. < 1 means "beats
  seasonal-naive one-step-ahead". The primary leaderboard metric.
- **sMAPE**, **WAPE**, **RMSE**, **MAE** — the usual suspects, reported.
- **coverage** — the fraction of actuals inside the prediction interval;
  compare against the configured level to judge interval honesty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mase_scale(train_y: np.ndarray, m: int) -> float:
    m = m if (m and m >= 1 and len(train_y) > m) else 1
    return float(np.mean(np.abs(train_y[m:] - train_y[:-m])))


def fold_cutoffs(panel, horizon: int, n_backtests: int) -> list:
    end = max(f.index[-1] for f in panel.frames.values())
    offset = pd.tseries.frequencies.to_offset(panel.freq)
    return [end - offset * (horizon * k) for k in range(n_backtests, 0, -1)]


def score_predictions(preds: dict, panel, cutoff, horizon: int, season: int,
                      level: float) -> pd.DataFrame:
    """Per-series metrics for one fold. preds: {sid: DataFrame(yhat, lo, hi)}.

    Gap-interpolated points (y_was_filled) are excluded from the scored
    actuals — models are never scored against synthetic values. The MASE
    scale keeps the interpolated training series: dropping interior points
    would misalign the m-lag differences it is built from, and interpolation
    is bounded by max_gap_frac.
    """
    rows = []
    for sid, frame in panel.frames.items():
        actual_window = frame.loc[frame.index > cutoff].head(horizon)
        if "y_was_filled" in frame.columns and len(actual_window):
            actual_window = actual_window.loc[actual_window["y_was_filled"] < 0.5]
        if sid not in preds or not len(actual_window):
            continue
        p = preds[sid].reindex(actual_window.index)
        y, yhat = actual_window["y"].to_numpy(), p["yhat"].to_numpy()
        ok = ~np.isnan(yhat)
        if not ok.any():
            continue
        y, yhat = y[ok], yhat[ok]
        train_y = frame.loc[frame.index <= cutoff, "y"].to_numpy()
        scale = mase_scale(train_y, season)

        err = y - yhat
        denom_smape = (np.abs(y) + np.abs(yhat))
        smape = float(np.mean(2 * np.abs(err)[denom_smape > 0]
                              / denom_smape[denom_smape > 0])) if (denom_smape > 0).any() else np.nan
        row = {"series": sid,
               "MASE": float(np.mean(np.abs(err)) / scale) if scale > 0 else np.nan,
               "sMAPE": smape,
               "WAPE": float(np.sum(np.abs(err)) / np.sum(np.abs(y)))
               if np.sum(np.abs(y)) > 0 else np.nan,
               "RMSE": float(np.sqrt(np.mean(err ** 2))),
               "MAE": float(np.mean(np.abs(err)))}
        lo, hi = p["lo"].to_numpy()[ok], p["hi"].to_numpy()[ok]
        if not np.isnan(lo).all():
            inside = (y >= lo) & (y <= hi)
            row["coverage"] = float(np.mean(inside))
        else:
            row["coverage"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_scores(fold_frames: list) -> dict:
    """Mean over folds of the mean over series."""
    if not fold_frames:
        return {}
    per_fold = [f.drop(columns=["series"]).mean(numeric_only=True)
                for f in fold_frames if len(f)]
    if not per_fold:
        return {}
    agg = pd.concat(per_fold, axis=1).mean(axis=1)
    return {k: float(v) for k, v in agg.items()}


def residuals_by_step(fold_preds: list, panel, horizon: int,
                      series_id=None) -> list:
    """|error| per forecast step from the backtest folds, as a list of
    per-step lists — the quantile inputs for empirical intervals.

    fold_preds: [(cutoff, {sid: pred_frame}), ...]. With ``series_id`` the
    pool is restricted to that series, so interval widths respect per-series
    scale; without it errors pool across all series (only appropriate as a
    fallback when a single series has too few residuals). Interpolated
    actuals (y_was_filled) never contribute residuals — and each residual is
    attributed to its TRUE forecast step, so a dropped filled point inside a
    window cannot shift later residuals onto earlier steps.
    """
    steps: list[list[float]] = [[] for _ in range(horizon)]
    for cutoff, preds in fold_preds:
        for sid, frame in panel.frames.items():
            if sid not in preds or (series_id is not None and sid != series_id):
                continue
            window = frame.loc[frame.index > cutoff].head(horizon)
            actual = window
            if "y_was_filled" in frame.columns:
                actual = window.loc[window["y_was_filled"] < 0.5]
            p = preds[sid].reindex(actual.index)
            pos = {ts: k for k, ts in enumerate(window.index)}
            err = np.abs(actual["y"].to_numpy() - p["yhat"].to_numpy())
            for ts, e in zip(actual.index, err):
                if not np.isnan(e):
                    steps[pos[ts]].append(float(e))
    return steps


def attainable_level(n_residuals: int) -> float:
    """Highest two-sided level an empirical |error| quantile can support.

    A distribution-free band built from ``n`` residuals can do no better than
    "wider than all n of them", and a fresh draw exceeds the sample maximum
    with probability 1/(n+1) — so the ceiling is n/(n+1), regardless of the
    level requested. With the default 3 backtest folds a per-series band tops
    out at 0.75, not the configured 0.90; with ``n_backtests=1`` it is 0.50.

    This is the *ceiling*, not the achieved coverage: it says what the
    estimator cannot exceed in expectation, which is why the engine warns when
    the configured level sits above it rather than quietly labelling the band
    with a number it cannot reach.
    """
    return n_residuals / (n_residuals + 1) if n_residuals >= 1 else 0.0


def interval_half_widths(steps: list, level: float, horizon: int) -> np.ndarray:
    """Per-step half-widths from pooled |error| quantiles; monotone non-decreasing.

    See ``attainable_level`` — with few residuals per step the requested
    ``level`` is not reachable, and the running max below is what keeps the
    widths from shrinking rather than a fix for the sample-size ceiling.
    """
    halves = np.full(horizon, np.nan)
    last = None
    for i in range(horizon):
        vals = steps[i] if i < len(steps) else []
        if vals:
            q = float(np.quantile(vals, level))
            last = q if last is None else max(q, last)  # widths shouldn't shrink with h
        if last is not None:
            halves[i] = last
    return halves
