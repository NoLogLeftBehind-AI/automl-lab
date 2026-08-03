"""The AutoForecast orchestrator.

    fc = AutoForecast(ForecastConfig(horizon=28, series_col="series"))
    fc.load(df)          # panel guardrails + seasonality detection
    fc.backtest()        # rolling-origin evaluation of the whole roster
    fc.forecast()        # refit champion on all data -> final H-step forecast
    fc.export()          # artifact folder: model, metadata, script, report

or ``fc.run(df)`` for the full sequence.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from .backtest import (aggregate_scores, attainable_level, fold_cutoffs,
                       interval_half_widths, residuals_by_step, score_predictions)
from .baselines import DriftForecaster, NaiveForecaster, SeasonalNaiveForecaster
from .config import ForecastConfig
from .data import build_panel, from_wide, load_dataframe
from .ml import make_ml_forecasters
from .seasonality import detect_seasonality
from .statistical import (HAS_PROPHET, AutoSarimaxForecaster, ETSForecaster,
                          ProphetForecaster, ThetaForecaster)
from .utils import DecisionLog, ForecastError

BLUE, ORANGE = "#4C72B0", "#DD8452"


def one_se_band(fold_mase: dict, best_name: str, cand_name: str) -> float:
    """Standard error of the (candidate - best) MASE difference across folds.

    The classic one-standard-error rule uses a standard *error*, sd/sqrt(k),
    not the marginal fold-to-fold standard *deviation* — which is ~sqrt(k)
    times wider. Every model is backtested on the same rolling-origin cutoffs,
    so the fold scores are paired: fold difficulty (a hard month, a holiday
    week) is common to both models and cancels in the difference. Falls back to
    the best model's sd/sqrt(k), then to its marginal sd, when the paired
    arrays are unavailable.
    """
    bf, cf = fold_mase.get(best_name), fold_mase.get(cand_name)
    if bf and cf and len(bf) == len(cf) > 1:
        d = np.asarray(cf, dtype=float) - np.asarray(bf, dtype=float)
        d = d[~np.isnan(d)]
        if len(d) > 1:
            return float(np.std(d, ddof=1) / np.sqrt(len(d)))
    if bf and len(bf) > 1:
        return float(np.std(bf) / np.sqrt(len(bf)))
    return float(np.std(bf or [0.0]))


def parsimony_pick(lb: pd.DataFrame, fold_mase: dict, blend_members: list) -> str:
    """The deployment pick: the candidate with the fewest fitted models whose
    mean MASE sits within one standard error of the best.

    The classic one-standard-error rule: a margin the backtest folds cannot
    resolve does not buy the extra refit/monitoring surface a blend carries.
    Ties break toward the lower MASE (``lb`` is MASE-sorted, best first).
    """
    best = lb.iloc[0]

    def n_models(name: str) -> int:
        return max(len(blend_members), 1) if name.startswith("Blend(") else 1

    keep = [m for m in lb["model"]
            if float(lb.loc[lb["model"] == m, "MASE"].iloc[0]) - float(best["MASE"])
            <= one_se_band(fold_mase, str(best["model"]), str(m))]
    within = lb[lb["model"].isin(keep)]
    min_c = min(n_models(m) for m in within["model"])
    return next(m for m in within["model"] if n_models(m) == min_c)


class AutoForecast:
    def __init__(self, config: ForecastConfig):
        # private copy: wide input rewrites config.series_col, and the caller's
        # config object must stay reusable across runs
        self.config = replace(config)
        self.log = DecisionLog()
        self._t0 = time.time()
        self.figures: dict = {}
        self.fold_predictions: dict = {}       # model -> [(cutoff, preds), ...]
        self.interval_method: dict = {}        # model -> 'native' | 'backtest-residual'
        # model -> coverage ceiling n/(n+1) of its residual bands (residual
        # methods only); None-free, populated when bands are built
        self._interval_attainable: dict = {}
        self._fold_mase: dict = {}             # model -> per-fold mean MASE (parsimony rule)

    # ------------------------------------------------------------------ load
    def load(self, df: pd.DataFrame | None = None, wide: bool = False) -> pd.DataFrame:
        if df is None:
            df = load_dataframe(self.config)
        if wide:
            df = from_wide(df, self.config.date_col)
            self.config.series_col = "series"
            self.log.log("data", "Wide input melted to long format "
                                 "(one row per series x timestamp)")
        # first pass without the seasonal hint, then re-check length guardrail
        self.panel = build_panel(df, self.config, self.log)
        self._synthesize_hierarchy_parents()
        self._resolve_holiday_country()
        self.season_info = detect_seasonality(self.panel, self.config, self.log)
        m = self.season_info["primary"]
        min_len = self.config.min_series_length or max(3 * m, 2 * self.config.horizon + 10, 30)
        for sid in list(self.panel.frames):
            if len(self.panel.frames[sid]) < min_len:
                self.panel.dropped[sid] = f"shorter than {min_len} (3 seasonal cycles)"
                self.log.warn("data", f"Series '{sid}' dropped: needs >= {min_len} points "
                                      f"for seasonal period {m}")
                del self.panel.frames[sid]
        if not self.panel.frames:
            raise ForecastError("No series long enough for the detected seasonality; "
                                "set config.seasonal_period or provide more history.")
        need = self.config.horizon * (self.config.n_backtests + 1)
        shortest = min(len(f) for f in self.panel.frames.values())
        if shortest <= need:
            raise ForecastError(
                f"horizon={self.config.horizon} with n_backtests={self.config.n_backtests} "
                f"needs > {need} points per series; shortest usable series has {shortest}. "
                "Reduce horizon/n_backtests or provide more history.")
        return self.panel.summary()

    # ----------------------------------------------------------------- roster
    def _build_roster(self) -> dict:
        cfg = self.config
        m = self.season_info["primary"]
        roster = {"Naive": NaiveForecaster(), "Drift": DriftForecaster()}
        if m > 1:
            roster["SeasonalNaive"] = SeasonalNaiveForecaster()
        if cfg.enable_ets:
            roster["ETS"] = ETSForecaster()
        if cfg.enable_theta:
            roster["Theta"] = ThetaForecaster()
        if cfg.enable_sarimax:
            roster["AutoSARIMAX"] = AutoSarimaxForecaster()
        if cfg.enable_prophet:
            if HAS_PROPHET:
                roster["Prophet"] = ProphetForecaster()
            else:
                self.log.warn("roster", "prophet not installed -> skipped "
                                        "(pip install prophet)")
        for fc in make_ml_forecasters(cfg):
            roster[fc.name] = fc
        self.log.log("roster", f"Roster ({len(roster)} models): {list(roster)}")
        return roster

    def _resolve_holiday_country(self):
        from .calendars import HAS_HOLIDAYS, holidays_supported_for
        self.holiday_country = None
        country = self.config.country_holidays
        if not country:
            return
        if not HAS_HOLIDAYS:
            self.log.warn("calendar", "country_holidays set but the 'holidays' package "
                                      "is not installed -> holiday features skipped")
            return
        if not holidays_supported_for(self.panel.freq):
            self.log.warn("calendar", "country_holidays set but holiday calendars need "
                                      f"sub-weekly data (daily/business-daily/hourly/"
                                      f"minute); frequency '{self.panel.freq}' is "
                                      "unsupported -> holiday features skipped")
            return
        self.holiday_country = country
        self.log.log("calendar", f"Holiday calendar enabled ({country}): day-of/"
                                 "before/after indicators feed the ML models and "
                                 "SARIMAX; Prophet uses its built-in holidays")

    def _synthesize_hierarchy_parents(self):
        """Parents missing from the panel are built by summing their children."""
        hier = self.config.hierarchy
        if not hier:
            return
        remaining = {p: kids for p, kids in hier.items()
                     if p not in self.panel.frames}
        if remaining and self.panel.exog_cols:
            raise ForecastError("Synthesizing hierarchy parents is not supported "
                                "together with exogenous columns (their parent "
                                "values would be ambiguous) — provide the parent "
                                "series in the data.")
        progress = True
        while remaining and progress:
            progress = False
            for parent, children in list(remaining.items()):
                if all(c in self.panel.frames for c in children):
                    idx = self.panel.frames[children[0]].index
                    for c in children[1:]:
                        idx = idx.intersection(self.panel.frames[c].index)
                    y = sum(self.panel.frames[c].loc[idx, "y"] for c in children)
                    frame = pd.DataFrame({"y": y, "y_was_filled": 0.0}, index=idx)
                    self.panel.frames[parent] = frame
                    self.log.log("hierarchy", f"Synthesized parent '{parent}' as the "
                                              f"sum of {len(children)} children "
                                              f"({len(idx):,} common timestamps)")
                    del remaining[parent]
                    progress = True
        if remaining:
            missing = {p: [c for c in kids if c not in self.panel.frames]
                       for p, kids in remaining.items()}
            raise ForecastError(f"Cannot synthesize hierarchy parents — missing "
                                f"children: {missing}")

    def _ctx(self):
        return {"season": self.season_info["primary"],
                "confirmed": self.season_info["confirmed"],
                "freq": self.panel.freq, "level": self.config.interval_level,
                "holiday_country": self.holiday_country,
                "config": self.config}

    def _future_exog_from_panel(self, cutoff, horizon) -> dict | None:
        """During backtests the 'future' exog values are the recorded ones."""
        if not self.panel.exog_cols:
            return None
        out = {}
        for sid, f in self.panel.frames.items():
            window = f.loc[f.index > cutoff].head(horizon)
            out[sid] = window[self.panel.exog_cols]
        return out

    # --------------------------------------------------------------- backtest
    def backtest(self) -> pd.DataFrame:
        cfg = self.config
        self.roster = self._build_roster()
        ctx = self._ctx()
        cutoffs = fold_cutoffs(self.panel, cfg.horizon, cfg.n_backtests)
        self.log.log("backtest", f"Rolling-origin backtests: {cfg.n_backtests} fold(s), "
                                 f"horizon {cfg.horizon}, cutoffs "
                                 f"{[str(c.date() if hasattr(c, 'date') else c) for c in cutoffs]}")

        rows = []
        for name, model in self.roster.items():
            t0 = time.time()
            fold_frames, fold_preds = [], []
            try:
                for cutoff in cutoffs:
                    train = self.panel.truncate(cutoff)
                    model.fit(train, ctx)
                    preds = model.predict(cfg.horizon,
                                          self._future_exog_from_panel(cutoff, cfg.horizon))
                    fold_preds.append((cutoff, preds))
                    fold_frames.append(score_predictions(
                        preds, self.panel, cutoff, cfg.horizon,
                        self.season_info["primary"], cfg.interval_level))
            except Exception as e:
                self.log.warn("backtest", f"{name} failed and is off the leaderboard: "
                                          f"{type(e).__name__}: {e}")
                continue
            self.fold_predictions[name] = fold_preds
            self.interval_method[name] = ("native" if model.native_intervals
                                          else "backtest-residual")
            fallbacks = getattr(model, "_fallbacks", None)
            if fallbacks:
                self.log.warn("backtest", f"{name}: per-series fit failed on "
                                          f"{len(fallbacks)} series -> seasonal-naive "
                                          f"fallback used for {sorted(fallbacks)[:5]}"
                                          f"{' ...' if len(fallbacks) > 5 else ''} "
                                          "(its leaderboard row includes those series)")
            agg = aggregate_scores(fold_frames)
            if not agg or np.isnan(agg.get("MASE", np.nan)):
                self.log.warn("backtest", f"{name}: no scorable predictions — skipped")
                continue
            self._fold_mase[name] = [float(f["MASE"].mean()) for f in fold_frames if len(f)]
            rows.append({"model": name, **{k: round(v, 4) for k, v in agg.items()},
                         "time_s": round(time.time() - t0, 1)})
            print(f"  {name:<18} MASE={agg['MASE']:.3f}  WAPE={agg['WAPE']:.3%}  "
                  f"({time.time()-t0:.1f}s)")

        if not rows:
            raise ForecastError("Every model failed backtesting — see the decision log.")

        # ---- blend of the top 3 (computed from stored fold predictions) -----
        lb = pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)
        if self.config.ensemble and len(lb) >= 3:
            top3 = lb["model"].head(3).tolist()
            fold_frames = []
            blend_folds = []
            for i, cutoff in enumerate(cutoffs):
                preds = {}
                for sid in self.panel.frames:
                    members = [self.fold_predictions[nm][i][1][sid]
                               for nm in top3 if sid in self.fold_predictions[nm][i][1]]
                    if not members:
                        continue
                    yhat = np.mean([mm["yhat"].to_numpy() for mm in members], axis=0)
                    preds[sid] = pd.DataFrame({"yhat": yhat, "lo": np.nan, "hi": np.nan},
                                              index=members[0].index)
                blend_folds.append((cutoff, preds))
                fold_frames.append(score_predictions(
                    preds, self.panel, cutoff, cfg.horizon,
                    self.season_info["primary"], cfg.interval_level))
            agg = aggregate_scores(fold_frames)
            if agg and not np.isnan(agg.get("MASE", np.nan)):
                name = f"Blend({', '.join(top3)})"
                self.fold_predictions[name] = blend_folds
                self.interval_method[name] = "backtest-residual"
                self._fold_mase[name] = [float(f["MASE"].mean())
                                         for f in fold_frames if len(f)]
                self._blend_members = top3
                lb = pd.concat([lb, pd.DataFrame([{"model": name,
                                                   **{k: round(v, 4) for k, v in agg.items()},
                                                   "time_s": 0.0}])], ignore_index=True)
                lb = lb.sort_values("MASE").reset_index(drop=True)
                print(f"  {name:<18} MASE={agg['MASE']:.3f}")

        lb.index += 1
        self.leaderboard = lb
        best_name = lb.iloc[0]["model"]
        pick = parsimony_pick(lb, self._fold_mase, getattr(self, "_blend_members", []))
        self.deployment_pick = pick
        if pick != best_name:
            band = one_se_band(self._fold_mase, str(best_name), str(pick))
            pick_mase = float(lb.loc[lb["model"] == pick, "MASE"].iloc[0])
            self.log.log("backtest",
                         f"Deployment pick: {pick} (MASE {pick_mase:.3f}) — within one "
                         f"standard error of the paired fold differences ({band:.3f}) "
                         f"of {best_name} "
                         f"({float(lb.iloc[0]['MASE']):.3f}), with fewer moving parts "
                         "to refit and monitor. "
                         + ("champion_policy='parsimonious' -> the pick ships as champion"
                            if cfg.champion_policy == "parsimonious" else
                            "The leaderboard stays accuracy-ranked; set "
                            "champion_policy='parsimonious' to ship the pick instead"))
        else:
            self.log.log("backtest",
                         "Deployment pick: the leaderboard winner itself — no candidate "
                         "with fewer models sits within one standard error of its MASE")
        self.champion_name = pick if cfg.champion_policy == "parsimonious" else best_name
        champ_mase = float(lb.loc[lb["model"] == self.champion_name, "MASE"].iloc[0])
        self.log.log("backtest", f"Champion: {self.champion_name} (mean MASE={champ_mase:.3f})")

        baseline_names = ("SeasonalNaive", "Naive", "Drift")
        baselines = lb[lb["model"].isin(baseline_names)]
        if len(baselines):
            bar_row = baselines.iloc[0]         # lb is MASE-sorted: best baseline
            if self.champion_name in baseline_names:
                self.log.warn("backtest",
                              f"A baseline ({self.champion_name}) won the leaderboard — "
                              "the data is likely near-random-walk, and no complex model "
                              "earned its keep here.")
            elif champ_mase > 0.95 * float(bar_row["MASE"]):
                self.log.warn("backtest",
                              f"The champion barely improves on {bar_row['model']} "
                              f"(MASE {champ_mase:.3f} vs {float(bar_row['MASE']):.3f}). "
                              "This is common on near-random-walk data — consider whether "
                              "a complex model earns its keep here.")
        return lb

    # --------------------------------------------------------------- forecast
    def forecast(self, future_exog: dict | None = None) -> pd.DataFrame:
        """Refit the champion on the full panel and produce the final forecast.

        ``future_exog``: {series_id: DataFrame indexed by the future dates with
        the exogenous columns} — required if the run uses exogenous regressors.
        """
        cfg = self.config
        ctx = self._ctx()
        name = self.champion_name

        if name.startswith("Blend("):
            members = self._blend_members
            member_preds = []
            self._final_models = {}
            for nm in members:
                model = self.roster[nm]
                model.fit(self.panel, ctx)
                self._final_models[nm] = model
                member_preds.append(model.predict(cfg.horizon, future_exog))
            preds = {}
            for sid in self.panel.frames:
                ms = [mp[sid] for mp in member_preds if sid in mp]
                yhat = np.mean([m["yhat"].to_numpy() for m in ms], axis=0)
                preds[sid] = pd.DataFrame({"yhat": yhat, "lo": np.nan, "hi": np.nan},
                                          index=ms[0].index)
        else:
            model = self.roster[name]
            model.fit(self.panel, ctx)
            self._final_models = {name: model}
            preds = model.predict(cfg.horizon, future_exog)

        # models without native intervals get empirical backtest-residual bands,
        # computed per series so widths respect each series' own error scale
        # (an 8x-larger parent series must not share bands with its children)
        if self.interval_method[name] == "backtest-residual":
            pooled_steps = residuals_by_step(self.fold_predictions[name], self.panel,
                                             cfg.horizon)
            pooled = interval_half_widths(pooled_steps, cfg.interval_level, cfg.horizon)
            n_pooled_fallback, min_n = 0, None
            for sid, p in preds.items():
                own = residuals_by_step(self.fold_predictions[name], self.panel,
                                        cfg.horizon, series_id=sid)
                if sum(len(s) for s in own) >= cfg.horizon:
                    halves, used = interval_half_widths(own, cfg.interval_level,
                                                        cfg.horizon), own
                else:
                    halves, used = pooled, pooled_steps
                    n_pooled_fallback += 1
                counts = [len(s) for s in used[:cfg.horizon] if s]
                if counts:
                    min_n = min(counts) if min_n is None else min(min_n, min(counts))
                p["lo"] = p["yhat"].to_numpy() - halves[: len(p)]
                p["hi"] = p["yhat"].to_numpy() + halves[: len(p)]
            # a distribution-free band from n residuals cannot exceed n/(n+1)
            # coverage however high the configured level is — say so rather
            # than ship a band labelled with a number it cannot reach
            if min_n is not None:
                attainable = attainable_level(min_n)
                self._interval_attainable[name] = attainable
                if attainable < cfg.interval_level - 1e-9:
                    self.log.warn(
                        "forecast",
                        f"{name}: {cfg.interval_level:.0%} residual bands are built from "
                        f"as few as {min_n} residual(s) per step "
                        f"({cfg.n_backtests} backtest fold(s)), which caps attainable "
                        f"coverage at ~{attainable:.0%}. The bands are the widest the "
                        "residuals support, but treat the nominal level as an upper "
                        "bound — raise n_backtests to close the gap.")
            self.log.log("forecast", f"{name} has no native intervals -> "
                                     f"{cfg.interval_level:.0%} bands from per-series "
                                     "backtest residual quantiles"
                                     + (f" ({n_pooled_fallback} series with too few "
                                        "residuals used the pooled widths)"
                                        if n_pooled_fallback else ""))

        # ---- hierarchical reconciliation -------------------------------------
        self.reconciliation_report = None
        self.reconciliation_method = None
        if cfg.hierarchy and cfg.reconciliation != "none":
            from .reconcile import (coherency_gap, evaluate_reconciliation,
                                    fold_error_matrix, reconcile_predictions,
                                    resolve_hierarchy, shrunk_covariance)
            ordered, leaves, S = resolve_hierarchy(cfg.hierarchy, self.panel.series_ids)
            self.reconciliation_report = evaluate_reconciliation(self)
            rep = self.reconciliation_report
            candidates = rep[rep["method"] != "none"]
            if cfg.reconciliation == "auto":
                method = candidates.iloc[0]["method"] if len(candidates) else "bottom_up"
            else:
                method = cfg.reconciliation
            gap_before = coherency_gap(preds, cfg.hierarchy)
            W = None
            if method == "mint_shrink":
                errs = fold_error_matrix(self.fold_predictions[name], self.panel,
                                         ordered, cfg.horizon)
                W = shrunk_covariance(errs)
            preds = reconcile_predictions(preds, ordered, S, method, W, leaves=leaves)
            self.reconciliation_method = method
            base_row = rep[rep["method"] == "none"]
            chosen_row = rep[rep["method"] == method]
            note = ""
            if len(base_row) and len(chosen_row):
                b, c = float(base_row.iloc[0]["MASE"]), float(chosen_row.iloc[0]["MASE"])
                note = (f"; backtest MASE {b:.3f} -> {c:.3f}"
                        + (" (coherence costs a little accuracy here)" if c > b else ""))
            self.log.log("reconcile", f"Applied '{method}' reconciliation "
                                      f"(selected {'automatically' if cfg.reconciliation == 'auto' else 'by config'}); "
                                      f"pre-reconciliation coherency gap was "
                                      f"{gap_before:.2%} of the parent level{note}. "
                                      "Intervals are translated with the point "
                                      "adjustment (widths unchanged).")

        self.final_forecast = preds
        self.figures["Forecast"] = self._plot_forecast(preds)
        rows = []
        for sid, p in preds.items():
            f = p.copy()
            f.insert(0, "series", sid)
            rows.append(f.reset_index(names="ds"))
        self.forecast_frame = pd.concat(rows, ignore_index=True)
        self.log.log("forecast", f"Final forecast: {cfg.horizon} steps x "
                                 f"{len(preds)} series by {name}")
        return self.forecast_frame

    def _plot_forecast(self, preds: dict, tail_points: int | None = None):
        # Imported here rather than at module scope so that plotting stays off
        # the `import autofc` path. The exported artifact bundles this package
        # and pins its own requirements (artifacts.pinned_requirements), which
        # cover the modelling stack only — a module-scope matplotlib import
        # would make unpickling the champion fail in an environment built from
        # the artifact's own requirements.txt.
        import matplotlib.pyplot as plt

        sids = list(preds)
        n = len(sids)
        ncols = min(2, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 2.8 * nrows),
                                 squeeze=False)
        m = self.season_info["primary"]
        tail = tail_points or max(6 * m, 4 * self.config.horizon)
        for ax, sid in zip(axes.flat, sids):
            hist = self.panel.frames[sid]["y"].iloc[-tail:]
            p = preds[sid]
            ax.plot(hist.index, hist, color=BLUE, lw=0.9, label="history")
            ax.plot(p.index, p["yhat"], color=ORANGE, lw=1.3, label="forecast")
            if not p["lo"].isna().all():
                ax.fill_between(p.index, p["lo"], p["hi"], color=ORANGE, alpha=0.2,
                                label=f"{self.config.interval_level:.0%} interval")
            ax.set_title(str(sid), fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in axes.flat[n:]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=8)
        fig.suptitle(f"{self.champion_name} — {self.config.horizon}-step forecast",
                     y=1.005)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ export
    def export(self):
        import matplotlib.pyplot as plt   # lazy — see _plot_forecast

        from .artifacts import export_artifacts
        runtime = time.time() - self._t0
        self.log.log("export", f"Total runtime: {runtime/60:.1f} min")
        art_dir = export_artifacts(self)
        for fig in self.figures.values():
            if fig is not None:
                plt.close(fig)
        return art_dir

    # --------------------------------------------------------------------- run
    def run(self, df: pd.DataFrame | None = None, wide: bool = False,
            future_exog: dict | None = None):
        self.load(df, wide=wide)
        self.backtest()
        self.forecast(future_exog)
        self.export()
        return self
