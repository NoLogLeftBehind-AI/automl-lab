#!/usr/bin/env python3
"""Benchmark the forecasting engine against Nixtla statsforecast and
AutoGluon-TimeSeries.

Protocol (mirrors the tabular benchmark): every system forecasts the SAME
rolling-origin folds of the same panel — the forecasting engine's own fold
cutoffs define the splits, its scoring code (MASE/sMAPE/WAPE/RMSE, unweighted
mean over series then folds) scores every system, and training data always
ends strictly at each cutoff. Challenger metrics are computed here from raw
fold predictions; the engine's row is its own leaderboard champion — produced
by the same scoring code on the same folds (see the README's methodology note).

Dataset: daily electricity load for the 8 ERCOT regions (2016–2021), horizon
28 days, 3 backtest folds — the same panel and fold protocol as the demo
notebook (base regions only: no synthesized total, hierarchy, or holidays).

Systems:
    autofc            the forecasting engine (its leaderboard champion)
    sf-autoarima      statsforecast AutoARIMA (season_length = 7)
    sf-autoets        statsforecast AutoETS
    sf-autotheta      statsforecast AutoTheta
    ag-ts-medium      AutoGluon-TS presets='medium_quality', 240s/fold
    ag-ts-best        AutoGluon-TS presets='best_quality',   600s/fold

Note: this environment cannot reach Hugging Face, so AutoGluon-TS runs
WITHOUT its pretrained Chronos models — a genuine handicap for it, stated in
the README rather than hidden.

Usage:
    python benchmark.py --system sf-autoarima
    python benchmark.py --all
    python benchmark.py --table
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "forecasting"))

from autofc import AutoForecast, ForecastConfig, from_wide  # noqa: E402
from autofc.backtest import (aggregate_scores, fold_cutoffs,  # noqa: E402
                             score_predictions)

RESULTS = HERE / "results" / "results.json"
ERCOT_URL = ("https://raw.githubusercontent.com/ourownstory/neuralprophet-data/"
             "main/datasets/multivariate/load_ercot_regions.csv")
HORIZON, N_FOLDS, SEASON = 28, 3, 7


def load_panel():
    """The exact panel the forecasting demo uses, via the engine's own guardrails."""
    cache = HERE / "results" / "_ercot.parquet"
    if cache.exists():
        raw = pd.read_parquet(cache)
    else:
        raw = pd.read_csv(ERCOT_URL)
        raw.to_parquet(cache)
    raw["ds"] = pd.to_datetime(raw["ds"])
    daily = raw.set_index("ds").resample("D").mean().reset_index()
    daily = daily[(daily.ds >= "2016-01-01") & (daily.ds < "2021-09-01")]
    df = from_wide(daily, "ds")
    fc = AutoForecast(ForecastConfig(horizon=HORIZON, series_col="series",
                                     n_backtests=N_FOLDS))
    fc.log.verbose = False
    fc.load(df)
    if fc.season_info["primary"] != SEASON:
        raise SystemExit(
            f"engine detected primary seasonal period {fc.season_info['primary']} "
            f"but challengers are MASE-scaled with SEASON={SEASON} — align them "
            "before benchmarking, or the two sides score on different scales")
    return fc


def panel_to_long(panel, cutoff) -> pd.DataFrame:
    rows = []
    for sid, f in panel.frames.items():
        g = f.loc[:cutoff, ["y"]].reset_index(names="ds")
        g["unique_id"] = sid
        rows.append(g[["unique_id", "ds", "y"]])
    return pd.concat(rows, ignore_index=True)


def score_fold_predictions(fc, all_fold_preds) -> dict:
    frames = []
    cutoffs = fold_cutoffs(fc.panel, HORIZON, N_FOLDS)
    for cutoff, preds in zip(cutoffs, all_fold_preds):
        frames.append(score_predictions(preds, fc.panel, cutoff, HORIZON,
                                        SEASON, 0.9))
    return aggregate_scores(frames)


# ---------------------------------------------------------------- systems ---

def run_autofc(args) -> dict:
    fc = load_panel()
    t0 = time.time()
    fc.backtest()
    fit_s = time.time() - t0
    row = fc.leaderboard.iloc[0]
    return {"MASE": float(row["MASE"]), "sMAPE": float(row["sMAPE"]),
            "WAPE": float(row["WAPE"]), "RMSE": float(row["RMSE"]),
            "fit_seconds": round(fit_s, 1),
            "champion": fc.champion_name,
            "note": "leaderboard champion over its full roster"}


def run_statsforecast(args, model_name: str) -> dict:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

    maker = {"AutoARIMA": lambda: AutoARIMA(season_length=SEASON),
             "AutoETS": lambda: AutoETS(season_length=SEASON),
             "AutoTheta": lambda: AutoTheta(season_length=SEASON)}[model_name]
    fc = load_panel()
    cutoffs = fold_cutoffs(fc.panel, HORIZON, N_FOLDS)
    t0 = time.time()
    all_preds = []
    for cutoff in cutoffs:
        train = panel_to_long(fc.panel, cutoff)
        sf = StatsForecast(models=[maker()], freq="D", n_jobs=-1)
        out = sf.forecast(df=train, h=HORIZON)
        col = [c for c in out.columns if c not in ("unique_id", "ds")][0]
        preds = {}
        for sid, g in out.groupby("unique_id"):
            preds[str(sid)] = pd.DataFrame(
                {"yhat": g[col].to_numpy(), "lo": np.nan, "hi": np.nan},
                index=pd.DatetimeIndex(g["ds"]))
        all_preds.append(preds)
    fit_s = time.time() - t0
    agg = score_fold_predictions(fc, all_preds)
    return {**{k: round(v, 4) for k, v in agg.items()
               if k in ("MASE", "sMAPE", "WAPE", "RMSE")},
            "fit_seconds": round(fit_s, 1)}


def run_autogluon_ts(args, presets: str, time_limit_per_fold: int) -> dict:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    fc = load_panel()
    cutoffs = fold_cutoffs(fc.panel, HORIZON, N_FOLDS)
    t0 = time.time()
    all_preds, champions = [], []
    for cutoff in cutoffs:
        train = panel_to_long(fc.panel, cutoff).rename(
            columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"})
        ts_train = TimeSeriesDataFrame.from_data_frame(train)
        workdir = tempfile.mkdtemp(prefix="agts_")
        predictor = TimeSeriesPredictor(prediction_length=HORIZON,
                                        eval_metric="MASE", path=workdir,
                                        verbosity=1)
        predictor.fit(ts_train, presets=presets, time_limit=time_limit_per_fold,
                      excluded_model_types=["Chronos"])  # no HF access here
        out = predictor.predict(ts_train)
        preds = {}
        for sid in out.index.get_level_values(0).unique():
            g = out.loc[sid]
            preds[str(sid)] = pd.DataFrame(
                {"yhat": g["mean"].to_numpy(),
                 "lo": g["0.05"].to_numpy() if "0.05" in g else np.nan,
                 "hi": g["0.95"].to_numpy() if "0.95" in g else np.nan},
                index=pd.DatetimeIndex(g.index))
        all_preds.append(preds)
        champions.append(str(predictor.model_best))
        shutil.rmtree(workdir, ignore_errors=True)
    fit_s = time.time() - t0
    agg = score_fold_predictions(fc, all_preds)
    return {**{k: round(v, 4) for k, v in agg.items()
               if k in ("MASE", "sMAPE", "WAPE", "RMSE")},
            "fit_seconds": round(fit_s, 1),
            "champion": max(set(champions), key=champions.count),
            "note": "Chronos excluded (no Hugging Face access in this environment)"}


SYSTEMS = {
    "autofc": run_autofc,
    "sf-autoarima": lambda a: run_statsforecast(a, "AutoARIMA"),
    "sf-autoets": lambda a: run_statsforecast(a, "AutoETS"),
    "sf-autotheta": lambda a: run_statsforecast(a, "AutoTheta"),
    "ag-ts-medium": lambda a: run_autogluon_ts(a, "medium_quality", 240),
    "ag-ts-best": lambda a: run_autogluon_ts(a, "best_quality", 600),
}


def save_result(system: str, payload: dict):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    versions = {}
    from importlib.metadata import version as _dist_version
    for lib in ("statsforecast", "autogluon.timeseries"):
        try:
            # __import__("autogluon.timeseries") returns the namespace package,
            # which has no __version__ — read the distribution metadata instead
            versions[lib] = _dist_version(lib)
        except Exception:
            pass
    payload["_meta"] = {"when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "horizon": HORIZON, "n_folds": N_FOLDS, **versions}
    data[system] = payload
    RESULTS.write_text(json.dumps(data, indent=2))


def print_table():
    data = json.loads(RESULTS.read_text())
    print("| System | MASE | WAPE | RMSE | wall-clock |")
    print("|---|---|---|---|---|")
    order = ["autofc", "sf-autoarima", "sf-autoets", "sf-autotheta",
             "ag-ts-medium", "ag-ts-best"]
    for s in order:
        if s not in data:
            continue
        r = data[s]
        champ = f" ({r['champion']})" if "champion" in r else ""
        print(f"| {s}{champ} | {r['MASE']:.3f} | {r['WAPE']:.3%} | "
              f"{r['RMSE']:,.0f} | {r['fit_seconds']/60:.1f} min |")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=list(SYSTEMS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    if args.table:
        print_table()
        return
    systems = [args.system] if args.system else (list(SYSTEMS) if args.all else None)
    if not systems:
        ap.error("pass --system, --all, or --table")
    for s in systems:
        print(f"\n{'='*70}\nRUN {s}\n{'='*70}", flush=True)
        t0 = time.time()
        result = SYSTEMS[s](args)
        save_result(s, result)
        print(f"-> {json.dumps({k: v for k, v in result.items() if k != '_meta'})}"
              f"  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
