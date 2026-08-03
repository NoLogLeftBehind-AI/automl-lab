"""Tests for the autofc forecasting engine: guardrails, backtest discipline,
and end-to-end artifact integrity. Prophet is disabled in most tests for
speed; one end-to-end test exercises it when installed."""
import importlib.util
import json
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest

from autofc import AutoForecast, ForecastConfig, ForecastError


def make_panel_df(n_series=3, n=300, freq="D", seed=0, amp=20.0, noise=3.0):
    rs = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq=freq)
    rows = []
    for i in range(n_series):
        base = 50 + 30 * i
        weekly = amp * np.sin(2 * np.pi * dates.dayofweek / 7)
        y = base + np.linspace(0, 10, n) + weekly + rs.randn(n) * noise
        rows.append(pd.DataFrame({"ds": dates, "series": f"s{i}", "y": y}))
    return pd.concat(rows, ignore_index=True)


# Exports land in a temp dir, never in the working tree: a `pytest` run on a
# fresh clone must not leave a `forecast_artifacts/` folder behind for the
# person evaluating the repo to clean up.
_TEST_ARTIFACTS = tempfile.mkdtemp(prefix="autofc_tests_")


def fast_config(**kw):
    defaults = dict(horizon=7, series_col="series", n_backtests=2,
                    enable_prophet=False, enable_ml=True, sarimax_max_obs=250,
                    n_jobs=2, artifact_dir=_TEST_ARTIFACTS)
    defaults.update(kw)
    return ForecastConfig(**defaults)


# ---------------------------------------------------------------- guardrails

def test_missing_columns_error():
    fc = AutoForecast(fast_config(target="sales"))
    with pytest.raises(ForecastError, match="not found"):
        fc.load(make_panel_df())


def test_unparseable_dates_error():
    df = make_panel_df()
    df["ds"] = "not-a-date-" + df.index.astype(str)
    with pytest.raises(ForecastError, match="datetimes"):
        AutoForecast(fast_config()).load(df)


def test_non_numeric_target_error():
    df = make_panel_df()
    df["y"] = "high"
    with pytest.raises(ForecastError, match="not numeric"):
        AutoForecast(fast_config()).load(df)


def test_duplicate_timestamps_aggregated_and_logged():
    df = make_panel_df()
    df = pd.concat([df, df.head(10)], ignore_index=True)  # duplicates in s0
    fc = AutoForecast(fast_config())
    fc.load(df)
    assert any("duplicate timestamps" in r["decision"] for r in fc.log.records)


def test_duplicate_timestamps_error_policy():
    df = make_panel_df()
    df = pd.concat([df, df.head(10)], ignore_index=True)
    with pytest.raises(ForecastError, match="duplicate"):
        AutoForecast(fast_config(duplicate_policy="error")).load(df)


def test_gaps_interpolated_within_threshold():
    df = make_panel_df()
    df = df.drop(df[(df.series == "s0")].index[50:60])  # 10 missing days
    fc = AutoForecast(fast_config())
    fc.load(df)
    assert any("time interpolation" in r["decision"] for r in fc.log.records)
    # regular grid restored
    f = fc.panel.frames["s0"]
    assert (f.index[1:] - f.index[:-1] == pd.Timedelta("1D")).all()


def test_too_gappy_series_dropped():
    df = make_panel_df(n_series=2)
    s0 = df[df.series == "s0"]
    rs = np.random.RandomState(0)
    keep = rs.rand(len(s0)) > 0.4          # ~40% of days randomly missing
    df = pd.concat([s0[keep], df[df.series == "s1"]])
    fc = AutoForecast(fast_config())
    fc.load(df)
    assert "s0" in fc.panel.dropped
    assert "s1" in fc.panel.frames


def test_constant_series_dropped():
    df = make_panel_df(n_series=2)
    df.loc[df.series == "s0", "y"] = 42.0
    fc = AutoForecast(fast_config())
    fc.load(df)
    assert "s0" in fc.panel.dropped and "constant" in fc.panel.dropped["s0"]


def test_all_series_dropped_is_clear_error():
    df = make_panel_df(n_series=1, n=20)  # far too short
    with pytest.raises(ForecastError):
        AutoForecast(fast_config()).load(df)


def test_horizon_too_large_for_history_errors():
    df = make_panel_df(n=80)
    with pytest.raises(ForecastError, match="Reduce horizon"):
        AutoForecast(fast_config(horizon=30, n_backtests=3)).load(df)


def test_single_series_without_series_col():
    df = make_panel_df(n_series=1).drop(columns=["series"])
    fc = AutoForecast(fast_config(series_col=None))
    fc.load(df)
    assert fc.panel.series_ids == ["series_0"]


def test_wide_format_load():
    df = make_panel_df(n_series=3)
    wide = df.pivot(index="ds", columns="series", values="y").reset_index()
    fc = AutoForecast(fast_config())
    fc.load(wide, wide=True)
    assert set(fc.panel.series_ids) == {"s0", "s1", "s2"}


def test_equivalent_frequency_labels_are_unified():
    """Regression test for a CI-only failure: pandas 3 names a one-day offset
    '24h' where pandas 2 says 'D', so a gappy series (frequency from the
    modal-delta fallback) disagreed with complete ones and killed the panel."""
    from autofc.data import _canonical_fixed_alias, _unify_freqs
    assert _canonical_fixed_alias(pd.Timedelta("1 day")) == "D"
    assert _canonical_fixed_alias(pd.Timedelta("2h")) == "2h"
    assert _unify_freqs({"s0": "24h", "s1": "D"}) == "D"
    assert _unify_freqs({"s0": "h", "s1": "D"}) is None          # genuine mismatch
    assert _unify_freqs({"s0": "W-SUN", "s1": "D"}) is None      # anchored vs fixed


def test_weekly_seasonality_detected():
    fc = AutoForecast(fast_config())
    fc.load(make_panel_df())
    assert fc.season_info["primary"] == 7


def test_no_seasonality_on_noise():
    rs = np.random.RandomState(1)
    dates = pd.date_range("2022-01-01", periods=300, freq="D")
    df = pd.DataFrame({"ds": dates, "y": rs.randn(300).cumsum() + 100})
    fc = AutoForecast(fast_config(series_col=None))
    fc.load(df)
    assert fc.season_info["primary"] == 1


def test_long_period_seasonality_detected():
    """Regression test: the original differenced-ACF detector missed even the
    yearly cycle in daily electricity load; the strength detector must confirm
    a yearly cycle AND reject it on a random walk of the same length."""
    rs = np.random.RandomState(4)
    n = 1400
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    t = np.arange(n)
    y = (1000 + 200 * np.sin(2 * np.pi * t / 365.25)
         + 40 * np.sin(2 * np.pi * (t % 7) / 7) + rs.randn(n) * 30)
    df = pd.DataFrame({"ds": dates, "y": y})
    fc = AutoForecast(fast_config(series_col=None))
    fc.load(df)
    assert 7 in fc.season_info["confirmed"]
    assert 365 in fc.season_info["confirmed"]
    assert fc.season_info["primary"] == 7    # short period drives SARIMAX/ETS

    rw = pd.DataFrame({"ds": dates, "y": rs.randn(n).cumsum() + 500})
    fc2 = AutoForecast(fast_config(series_col=None))
    fc2.load(rw)
    assert 365 not in fc2.season_info["confirmed"]


# ---------------------------------------------------- backtest discipline ---

def test_backtest_no_future_leakage():
    """A model fit at a cutoff must never see rows after it: the panel handed
    to fit() is truncated, so training-set maxima stay behind every cutoff."""
    fc = AutoForecast(fast_config())
    fc.load(make_panel_df())
    from autofc.backtest import fold_cutoffs
    cutoffs = fold_cutoffs(fc.panel, fc.config.horizon, fc.config.n_backtests)
    for cutoff in cutoffs:
        train = fc.panel.truncate(cutoff)
        for sid, f in train.frames.items():
            assert f.index.max() <= cutoff


def test_leaderboard_and_champion():
    fc = AutoForecast(fast_config())
    fc.load(make_panel_df())
    lb = fc.backtest()
    assert "MASE" in lb.columns
    assert lb["MASE"].is_monotonic_increasing
    # strong weekly signal: something must beat plain Naive decisively
    naive = float(lb[lb.model == "Naive"]["MASE"].iloc[0])
    assert float(lb["MASE"].iloc[0]) < naive
    assert fc.champion_name == lb.iloc[0]["model"]


def test_seasonal_naive_beats_naive_on_seasonal_data():
    fc = AutoForecast(fast_config())
    fc.load(make_panel_df(amp=30, noise=1.0))
    lb = fc.backtest()
    snaive = float(lb[lb.model == "SeasonalNaive"]["MASE"].iloc[0])
    naive = float(lb[lb.model == "Naive"]["MASE"].iloc[0])
    assert snaive < naive


def test_near_random_walk_warning(tmp_path):
    rs = np.random.RandomState(2)
    dates = pd.date_range("2021-01-01", periods=400, freq="D")
    df = pd.DataFrame({"ds": dates, "y": rs.randn(400).cumsum() + 500})
    art = tmp_path / "artifacts"
    fc = AutoForecast(fast_config(series_col=None, enable_ml=False,
                                  enable_sarimax=False, enable_theta=False,
                                  enable_ets=False, ensemble=False,
                                  artifact_dir=str(art)))
    fc.run(df)
    # a baseline wins on a random walk and the engine must SAY so — the old
    # version of this assertion passed even when the warning could never fire
    # (no SeasonalNaive row exists on non-seasonal data)
    msgs = [r["decision"] for r in fc.log.records]
    assert any("won the leaderboard" in m or "barely improves" in m for m in msgs)
    # ... and the shareable report must show the banner, not only the console
    # log: its filter once matched a string no log site ever emitted
    html_text = (art / "report.html").read_text()
    assert "class='warn'" in html_text


# ------------------------------------------------------------- end to end ---

def test_end_to_end_with_artifacts(tmp_path):
    art = tmp_path / "artifacts"
    fc = AutoForecast(fast_config(artifact_dir=str(art)))
    fc.run(make_panel_df())

    meta = json.loads((art / "metadata.json").read_text())
    assert meta["horizon"] == 7
    assert meta["n_series"] == 3
    assert len(meta["leaderboard"]) >= 5
    # the parsimony deployment pick ships in the metadata and names a real row
    assert meta["deployment_recommendation"]["model"] in [
        r["model"] for r in meta["leaderboard"]]

    fcst = pd.read_csv(art / "forecast.csv")
    assert len(fcst) == 3 * 7
    assert {"series", "ds", "yhat", "lo", "hi"} <= set(fcst.columns)
    # intervals must exist for the champion (native or residual-based)
    assert fcst["lo"].notna().all()

    # bundle round-trip: reload models and forecast again
    bundle = joblib.load(art / "model.joblib")
    preds = bundle["models"][bundle["blend_members"][0]].predict(7)
    assert len(preds) == 3

    # generated script runs from the artifact folder alone
    out = tmp_path / "script_fcst.csv"
    res = subprocess.run([sys.executable, str(art / "forecast.py"),
                          "--output", str(out), "--horizon", "10"],
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr
    script_fcst = pd.read_csv(out)
    assert len(script_fcst) == 3 * 10

    # report exists and carries the leaderboard
    html = (art / "report.html").read_text()
    assert "Leaderboard" in html and "MASE" in html


def test_exog_regressors_flow_through(tmp_path):
    rs = np.random.RandomState(3)
    dates = pd.date_range("2022-01-01", periods=300, freq="D")
    promo = (rs.rand(300) < 0.2).astype(float)
    y = 100 + 25 * promo + 5 * np.sin(2 * np.pi * dates.dayofweek / 7) + rs.randn(300)
    df = pd.DataFrame({"ds": dates, "y": y, "promo": promo})
    fc = AutoForecast(fast_config(series_col=None, enable_ml=False,
                                  enable_theta=False, enable_ets=False,
                                  ensemble=False, artifact_dir=str(tmp_path / "a")))
    fc.load(df)
    fc.backtest()
    future_idx = fc.panel.future_index(fc.config.horizon)
    fut = {"series_0": pd.DataFrame({"promo": np.ones(len(future_idx))}, index=future_idx)}
    fcst = fc.forecast(future_exog=fut)
    assert len(fcst) == fc.config.horizon
    # SARIMAX must demand exog when trained with it
    with pytest.raises(ValueError, match="future_exog"):
        fc.roster["AutoSARIMAX"].predict(7, None)


# ------------------------------------------------- holidays & hierarchy ---

def test_holiday_flags_mark_real_holidays():
    from autofc.calendars import holiday_flags
    idx = pd.date_range("2023-12-20", "2024-01-05", freq="D")
    f = holiday_flags(idx, "US")
    assert f.loc["2023-12-25", "is_holiday"] == 1.0        # Christmas
    assert f.loc["2024-01-01", "is_holiday"] == 1.0        # New Year
    assert f.loc["2023-12-24", "is_day_before_holiday"] == 1.0
    assert f.loc["2023-12-26", "is_day_after_holiday"] == 1.0
    assert f.loc["2023-12-28", "is_holiday"] == 0.0


def test_holiday_features_flow_into_ml_and_sarimax():
    df = make_panel_df()
    fc = AutoForecast(fast_config(country_holidays="US"))
    fc.load(df)
    assert fc.holiday_country == "US"
    from autofc.ml import build_features, feature_plan
    plan = feature_plan(7, [7], "US")
    feats = build_features(fc.panel.frames["s0"], plan, "D", [])
    assert "is_holiday" in feats.columns
    # SARIMAX carries holiday indicators as deterministic exog and still runs
    from autofc.statistical import AutoSarimaxForecaster
    m = AutoSarimaxForecaster().fit(fc.panel, fc._ctx())
    preds = m.predict(7)
    assert len(preds) == 3


def test_hierarchy_parent_synthesized_and_reconciled(tmp_path):
    """End to end: parent synthesized by summation, reconciliation applied,
    final forecasts coherent (children sum exactly to the parent)."""
    df = make_panel_df(n_series=3)
    cfg = fast_config(hierarchy={"TOTAL": ["s0", "s1", "s2"]},
                      artifact_dir=str(tmp_path / "a"))
    fc = AutoForecast(cfg)
    fc.run(df)
    assert "TOTAL" in fc.panel.frames
    assert fc.reconciliation_method in ("bottom_up", "ols", "wls_struct", "mint_shrink")
    p = fc.final_forecast
    total = p["TOTAL"]["yhat"].to_numpy()
    kids = sum(p[s]["yhat"].to_numpy() for s in ("s0", "s1", "s2"))
    np.testing.assert_allclose(total, kids, rtol=1e-8)
    # report + metadata carry the reconciliation story
    meta = json.loads((tmp_path / "a" / "metadata.json").read_text())
    assert meta["reconciliation_method"] == fc.reconciliation_method
    assert meta["reconciliation_backtest"]


def test_hierarchy_with_missing_children_errors():
    df = make_panel_df(n_series=2)
    fc = AutoForecast(fast_config(hierarchy={"TOTAL": ["s0", "s1", "nope"]}))
    with pytest.raises(ForecastError, match="not in the panel|missing"):
        fc.run(df)


def test_multilevel_s_matrix():
    from autofc.reconcile import resolve_hierarchy
    hierarchy = {"TOTAL": ["A", "B"], "A": ["A1", "A2"]}
    series = ["TOTAL", "A", "B", "A1", "A2"]
    ordered, leaves, S = resolve_hierarchy(hierarchy, series)
    assert leaves == ["B", "A1", "A2"]
    lp = {s: i for i, s in enumerate(leaves)}
    total_row = S[ordered.index("TOTAL")]
    assert total_row[lp["B"]] == 1 and total_row[lp["A1"]] == 1 and total_row[lp["A2"]] == 1
    a_row = S[ordered.index("A")]
    assert a_row[lp["A1"]] == 1 and a_row[lp["A2"]] == 1 and a_row[lp["B"]] == 0


def test_bottom_up_reconciliation_exact():
    from autofc.reconcile import reconcile_predictions, resolve_hierarchy
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    preds = {"TOTAL": pd.DataFrame({"yhat": [100.0, 100, 100], "lo": np.nan, "hi": np.nan}, index=idx),
             "a": pd.DataFrame({"yhat": [30.0, 30, 30], "lo": np.nan, "hi": np.nan}, index=idx),
             "b": pd.DataFrame({"yhat": [50.0, 50, 50], "lo": np.nan, "hi": np.nan}, index=idx)}
    ordered, leaves, S = resolve_hierarchy({"TOTAL": ["a", "b"]}, list(preds))
    rec = reconcile_predictions(preds, ordered, S, "bottom_up", leaves=leaves)
    np.testing.assert_allclose(rec["TOTAL"]["yhat"], [80.0, 80, 80])  # leaves kept
    np.testing.assert_allclose(rec["a"]["yhat"], [30.0, 30, 30])


def test_ml_forecaster_pickles_after_fit():
    """Regression test: fitted global ML forecasters must survive pickling
    (a lambda factory once made the ERCOT demo's artifact export crash)."""
    import pickle
    fc = AutoForecast(fast_config(enable_sarimax=False, enable_theta=False,
                                  enable_ets=False, ensemble=False))
    fc.load(make_panel_df())
    from autofc.ml import make_ml_forecasters
    model = make_ml_forecasters(fc.config)[0]
    model.fit(fc.panel, fc._ctx())
    restored = pickle.loads(pickle.dumps(model))
    preds = restored.predict(5)
    assert len(preds) == 3


@pytest.mark.skipif(importlib.util.find_spec("prophet") is None,
                    reason="prophet not installed")
def test_prophet_pickles_via_json_roundtrip(tmp_path):
    df = make_panel_df(n_series=2, n=250)
    fc = AutoForecast(fast_config(enable_prophet=True, enable_ml=False,
                                  enable_sarimax=False, enable_theta=False,
                                  enable_ets=False, ensemble=False,
                                  artifact_dir=str(tmp_path / "a")))
    fc.load(df)
    fc.backtest()
    prophet = fc.roster["Prophet"]
    ctx = fc._ctx()
    prophet.fit(fc.panel, ctx)
    import pickle
    restored = pickle.loads(pickle.dumps(prophet))
    preds = restored.predict(7)
    assert len(preds) == 2


# ------------------------------------------------- review regression tests ---

def test_bottom_up_single_child_parent_not_double_counted():
    """Leaf detection by unit row-sums once matched single-child parents too,
    double-counting the branch (TOTAL came out 133 instead of 93)."""
    from autofc.reconcile import reconcile_predictions, resolve_hierarchy
    idx = pd.date_range("2024-01-01", periods=2, freq="D")

    def mk(v):
        return pd.DataFrame({"yhat": [float(v)] * 2, "lo": np.nan, "hi": np.nan},
                            index=idx)

    preds = {"TOTAL": mk(100), "A": mk(40), "B": mk(55), "A1": mk(38)}
    ordered, leaves, S = resolve_hierarchy({"TOTAL": ["A", "B"], "A": ["A1"]},
                                           list(preds))
    assert set(leaves) == {"B", "A1"}
    rec = reconcile_predictions(preds, ordered, S, "bottom_up", leaves=leaves)
    np.testing.assert_allclose(rec["A1"]["yhat"], 38.0)
    np.testing.assert_allclose(rec["A"]["yhat"], 38.0)
    np.testing.assert_allclose(rec["TOTAL"]["yhat"], 93.0)


def test_seasonality_candidates_frequency_matching():
    """'min' must never prefix-match the monthly entry, and multiplied
    frequencies scale their candidate periods."""
    from autofc.seasonality import _candidates_for_freq
    assert _candidates_for_freq("D") == [7, 365]
    assert _candidates_for_freq("W-SUN") == [52]
    assert _candidates_for_freq("M") == [12]
    assert _candidates_for_freq("min") == [60, 1440]
    assert _candidates_for_freq("T") == [60, 1440]
    assert _candidates_for_freq("30min") == [2, 48]
    assert _candidates_for_freq("2h") == [12, 84]
    assert _candidates_for_freq(None) == []


def test_filled_gap_points_are_not_scored():
    """Interpolated actuals must never be scored as real ones."""
    from autofc.backtest import score_predictions
    df = make_panel_df(n_series=1, n=120)
    df = df[df.ds != "2022-02-10"]          # punch a hole -> interpolated at load
    fc = AutoForecast(fast_config())
    fc.load(df)
    frame = fc.panel.frames["s0"]
    assert frame["y_was_filled"].sum() == 1
    filled_ts = frame.index[frame["y_was_filled"] > 0][0]
    cutoff = filled_ts - pd.Timedelta(days=3)
    idx = pd.date_range(cutoff + pd.Timedelta(days=1), periods=7, freq="D")
    p = pd.DataFrame({"yhat": frame["y"].reindex(idx).to_numpy().copy(),
                      "lo": np.nan, "hi": np.nan}, index=idx)
    p.loc[filled_ts, "yhat"] = 1e9          # absurd prediction at the filled point
    scores = score_predictions({"s0": p}, fc.panel, cutoff, 7, 7, 0.9)
    assert float(scores.iloc[0]["MAE"]) < 1.0   # the absurd point was excluded


def test_requirements_pin_only_the_bundled_models_libraries(tmp_path):
    """`autofc` imports every model library lazily, so an artifact only needs
    what its own champion uses. Pinning the whole roster instead shipped
    statsmodels and three GBMs to artifacts that touch none of them.
    """
    from autofc.artifacts import pinned_requirements
    from autofc.baselines import NaiveForecaster, SeasonalNaiveForecaster
    from autofc.ml import make_ml_forecasters
    from autofc.statistical import AutoSarimaxForecaster

    def names(models, holidays=None):
        return {r.split("==")[0] for r in pinned_requirements(models, holidays)}

    core = {"pandas", "numpy", "scipy", "joblib", "scikit-learn"}

    # baselines are pure numpy/scipy — nothing else may be pinned
    assert names({"Naive": NaiveForecaster(), "SN": SeasonalNaiveForecaster()}) == core
    # a statsmodels champion pins statsmodels and no GBM
    sarimax = names({"AutoSARIMAX": AutoSarimaxForecaster()})
    assert sarimax == core | {"statsmodels"}
    # holiday features are regenerated at predict time, so they are a real dep
    assert "holidays" in names({"Naive": NaiveForecaster()}, "US")

    # a global-GBM champion pins its own library only
    ml = {m.name: m for m in make_ml_forecasters(fast_config())}
    if "LightGBM_global" in ml:
        got = names({"LightGBM_global": ml["LightGBM_global"]})
        assert "lightgbm" in got
        assert "xgboost" not in got and "catboost" not in got
        assert "statsmodels" not in got
    # HistGB is scikit-learn: no extra pin at all
    assert names({"HistGB_global": ml["HistGB_global"]}) == core

    # and the real export writes the same thing
    fc = AutoForecast(fast_config(artifact_dir=str(tmp_path / "art"))).run(
        make_panel_df(n_series=3))
    text = (tmp_path / "art" / "requirements.txt").read_text()
    pinned = {ln.split("==")[0] for ln in text.splitlines()
              if ln.strip() and not ln.startswith("#")}
    expected = set()
    for model in fc._final_models.values():
        expected |= set(getattr(model, "requires", ()))
    for lib in ("statsmodels", "lightgbm", "xgboost", "catboost", "prophet"):
        if lib in pinned:
            assert lib in expected, f"{lib} pinned but no bundled model needs it"


def test_residual_band_coverage_ceiling_is_computed_and_warned():
    """A distribution-free band from n residuals cannot beat n/(n+1) coverage.

    With the default 3 backtest folds a per-series 90% band tops out near 75%,
    and n_backtests=1 tops out at 50%. The engine must surface that ceiling
    rather than ship a band labelled 90% that cannot reach it.
    """
    from autofc.backtest import attainable_level

    assert attainable_level(1) == pytest.approx(0.50)
    assert attainable_level(3) == pytest.approx(0.75)     # the shipped default
    assert attainable_level(9) == pytest.approx(0.90)     # 9 folds reach 90%
    assert attainable_level(0) == 0.0

    df = make_panel_df(n_series=3)
    fc = AutoForecast(fast_config(n_backtests=2, interval_level=0.90))
    fc.run(df)

    if fc.interval_method[fc.champion_name] == "backtest-residual":
        ceiling = fc._interval_attainable[fc.champion_name]
        assert ceiling < 0.90, "a 2-fold residual band cannot support 90%"
        log = " ".join(r["decision"] for r in fc.log.records)
        assert "caps attainable coverage" in log
        assert fc.metadata["interval_attainable_level"] == pytest.approx(ceiling)


def test_residual_intervals_respect_series_scale():
    """Per-series residual pools: a small series must not inherit a large
    series' interval widths (or vice versa)."""
    from autofc.backtest import interval_half_widths, residuals_by_step
    from autofc.data import Panel
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    panel = Panel()
    panel.freq = "D"
    rs = np.random.RandomState(0)
    panel.frames = {
        "small": pd.DataFrame({"y": 10 + rs.randn(40)}, index=idx),
        "big": pd.DataFrame({"y": 1000 + 100 * rs.randn(40)}, index=idx),
    }
    cutoff = idx[29]
    preds = {}
    for sid, f in panel.frames.items():
        win = f.index[f.index > cutoff]
        offset = 1.0 if sid == "small" else 100.0
        preds[sid] = pd.DataFrame({"yhat": f.loc[win, "y"].to_numpy() + offset,
                                   "lo": np.nan, "hi": np.nan}, index=win)
    fold_preds = [(cutoff, preds)]
    h_small = interval_half_widths(
        residuals_by_step(fold_preds, panel, 10, series_id="small"), 0.9, 10)
    h_big = interval_half_widths(
        residuals_by_step(fold_preds, panel, 10, series_id="big"), 0.9, 10)
    assert np.nanmax(h_small) < 5
    assert np.nanmin(h_big) > 50


def test_forecast_cli_template_reconciliation_and_per_series_bands():
    """The generated CLI must warn about unreproduced reconciliation and use
    per-series interval widths."""
    from autofc.artifacts import FORECAST_SRC
    # the warning message is wrapped across adjacent string literals in the
    # template source, so drop the quotes and collapse whitespace before
    # asserting on the message text
    flat = " ".join(FORECAST_SRC.replace('"', ' ').split())
    assert "reconciliation_method" in FORECAST_SRC
    assert "does not reproduce" in flat
    assert "halves.get(sid)" in FORECAST_SRC


def test_fourier_terms_survive_daily_freq():
    """Dividing a TimedeltaIndex by to_offset('D') raises on pandas 3, where
    Day is no longer a Tick — the long-seasonality Fourier path must not
    depend on it (its failure was silently absorbed as a leaderboard dropout)."""
    from autofc.statistical import fourier_terms
    idx = pd.date_range("2020-01-01", periods=800, freq="D")
    f = fourier_terms(idx, 365.25, K=2, origin=idx[0], freq="D")
    assert f.shape == (800, 4)
    assert np.isfinite(f.to_numpy()).all()
    # one year apart lands at (nearly) the same phase
    assert abs(f.iloc[0, 0] - f.iloc[365, 0]) < 0.05


def test_base_freq_alias_families():
    """Frequency-family decisions must exact-match the base alias — prefix
    matching classified 'min' as monthly and dropped features for '2h'."""
    from autofc.calendars import holidays_supported_for
    from autofc.utils import base_freq_alias
    assert base_freq_alias("30min") == "min"
    assert base_freq_alias("T") == "min"
    assert base_freq_alias("2h") == "h"
    assert base_freq_alias("W-SUN") == "w"
    assert base_freq_alias("MIN") == "min"
    assert base_freq_alias(None) == ""
    assert holidays_supported_for("2h")
    assert holidays_supported_for("30min")
    assert not holidays_supported_for("MS")
    assert not holidays_supported_for("W-SUN")


def test_residuals_attributed_to_true_forecast_step():
    """A filled point dropped inside a scored window must not shift later
    residuals onto earlier steps (that inflated early-step interval widths
    and lost the largest late-step residuals)."""
    from autofc.backtest import residuals_by_step
    from autofc.data import Panel
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    frame = pd.DataFrame({"y": np.zeros(30), "y_was_filled": 0.0}, index=idx)
    window = idx[20:27]                              # the 7-step window
    frame.loc[window[1], "y_was_filled"] = 1.0       # step 2 is synthetic
    panel = Panel()
    panel.freq = "D"
    panel.frames = {"s": frame}
    errs = np.array([1.0, 99.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    p = pd.DataFrame({"yhat": -errs, "lo": np.nan, "hi": np.nan}, index=window)
    steps = residuals_by_step([(idx[19], {"s": p})], panel, horizon=7)
    assert steps[1] == []                    # the filled step contributes nothing
    assert steps[2] == [3.0]                 # step 3 stays step 3
    assert steps[6] == [7.0]                 # the largest residual is not lost


def test_parsimony_pick_prefers_fewest_models_within_noise():
    """The deployment pick is the candidate with the fewest fitted models
    within one standard error of the best mean MASE — a three-model blend must
    earn its ops surface with a margin the backtest folds can resolve."""
    from autofc.core import parsimony_pick
    blend = "Blend(XGBoost_global, LightGBM_global, Prophet)"
    lb = pd.DataFrame([
        {"model": blend, "MASE": 0.94},
        {"model": "XGBoost_global", "MASE": 0.96},
        {"model": "SeasonalNaive", "MASE": 1.10},
    ])
    # only the best model's folds are known -> documented sd/sqrt(k) fallback.
    # blend's margin (0.02) inside the band (0.03/sqrt(2)) -> single model
    fold_mase = {blend: [0.91, 0.97]}
    assert parsimony_pick(lb, fold_mase, ["a", "b", "c"]) == "XGBoost_global"
    # tight folds: the blend's win is real -> the blend stays
    fold_mase = {blend: [0.935, 0.945]}
    assert parsimony_pick(lb, fold_mase, ["a", "b", "c"]) == blend
    # a single model on top needs no defending
    lb2 = pd.DataFrame([
        {"model": "XGBoost_global", "MASE": 0.93},
        {"model": blend, "MASE": 0.94},
    ])
    assert parsimony_pick(lb2, {"XGBoost_global": [0.91, 0.95]},
                          ["a", "b", "c"]) == "XGBoost_global"


def test_parsimony_band_uses_paired_fold_differences():
    """With both models' folds known the band is the SE of their per-fold
    *differences*, not either model's marginal spread.

    Backtest folds differ wildly in difficulty (a holiday week, a demand
    spike) and that difficulty hits every model at once. Banding on the
    marginal spread therefore discards blend wins that are perfectly
    consistent fold to fold — exactly the evidence that should keep a blend.
    """
    from autofc.core import one_se_band, parsimony_pick
    blend = "Blend(a, b, c)"
    lb = pd.DataFrame([{"model": blend, "MASE": 0.94},
                       {"model": "XGBoost_global", "MASE": 0.96}])

    # wide marginal spread, but the blend wins by exactly 0.02 on every fold
    blend_folds = [0.88, 1.00, 0.91, 0.97]
    fold_mase = {blend: blend_folds,
                 "XGBoost_global": [f + 0.02 for f in blend_folds]}
    assert np.std(blend_folds) > 0.04                      # marginal: looks noisy
    assert one_se_band(fold_mase, blend, "XGBoost_global") == pytest.approx(0, abs=1e-12)
    assert parsimony_pick(lb, fold_mase, ["a", "b", "c"]) == blend

    # same mean margin, erratic per-fold differences -> simpler model ships
    fold_mase = {blend: blend_folds,
                 "XGBoost_global": [f + d for f, d in
                                    zip(blend_folds, [0.10, -0.07, 0.06, -0.01])]}
    assert one_se_band(fold_mase, blend, "XGBoost_global") > 0.02
    assert parsimony_pick(lb, fold_mase, ["a", "b", "c"]) == "XGBoost_global"


def test_config_object_is_reusable_for_wide_input():
    """Wide input once rewrote config.series_col on the caller's config
    object, making the config single-use."""
    df = make_panel_df(n_series=3)
    wide = df.pivot(index="ds", columns="series", values="y").reset_index()
    cfg = fast_config(series_col=None)
    for _ in range(2):
        fc = AutoForecast(cfg)
        fc.load(wide, wide=True)
        assert fc.config.series_col == "series"   # the run's own copy
        assert cfg.series_col is None             # the caller's object
