"""Self-contained HTML report for a forecasting run — same design language as
the tabular engine's report: leaderboard, backtest metrics, forecast plots,
and the full decision audit, with figures embedded as base64 PNGs."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import pandas as pd

_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 2rem auto;
       max-width: 1000px; color: #1a1a2e; line-height: 1.45; padding: 0 1rem; }
h1 { border-bottom: 3px solid #4C72B0; padding-bottom: .3rem; }
h2 { color: #274472; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .2rem; }
table { border-collapse: collapse; margin: .8rem 0; font-size: .9rem; }
th, td { border: 1px solid #ccc; padding: .35rem .6rem; text-align: left; }
th { background: #eef2f8; }
tr:nth-child(even) { background: #f7f9fc; }
img { max-width: 100%; height: auto; border: 1px solid #eee; margin: .5rem 0; }
.badge { display: inline-block; background: #4C72B0; color: white; border-radius: 4px;
         padding: .15rem .55rem; margin-right: .4rem; font-size: .85rem; }
.warn { background: #fff3cd; border-left: 4px solid #DD8452; padding: .6rem .9rem; margin: .8rem 0; }
.note { color: #555; font-size: .88rem; }
"""


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df is None or not len(df):
        return "<p class='note'>—</p>"
    return df.head(max_rows).to_html(border=0, float_format=lambda v: f"{v:.4f}")


def build_forecast_report(path: Path, fc, meta: dict) -> Path:
    attainable = meta.get("interval_attainable_level")
    # a residual band from n residuals cannot exceed n/(n+1) coverage — when
    # the nominal level sits above that ceiling the badge says so, so the
    # report never advertises a level the estimator cannot reach
    capped = (f", capped at ~{attainable:.0%} by fold count"
              if attainable is not None and attainable < meta["interval_level"] - 1e-9
              else "")
    badges = [f"champion: {meta['champion']}", f"horizon: {meta['horizon']}",
              f"freq: {meta['freq']}", f"series: {meta['n_series']}",
              f"intervals: {meta['interval_level']:.0%} "
              f"({meta['interval_method']}{capped})"]
    parts = [f"<html><head><meta charset='utf-8'>"
             f"<title>Forecast report — {html.escape(meta['champion'])}</title>",
             f"<style>{_CSS}</style></head><body>",
             "<h1>AutoML forecast report</h1>",
             "".join(f"<span class='badge'>{html.escape(b)}</span>" for b in badges),
             f"<p class='note'>Generated {meta['created_utc']} · statsmodels "
             f"{meta['versions']['statsmodels']} · python {meta['versions']['python']}</p>"]

    # the stable fragments core.py logs for both near-random-walk outcomes:
    # "barely improves on <baseline>" and "A baseline (...) won the leaderboard"
    warns = [r for r in meta["decision_log"]
             if "barely improves on" in r["decision"]
             or "won the leaderboard" in r["decision"]]
    if warns:
        msg = warns[0]["decision"].removeprefix("WARNING: ")
        parts.append(f"<div class='warn'><b>⚠</b> {html.escape(msg)}</div>")

    parts.append("<h2>Leaderboard (rolling-origin backtests)</h2>")
    parts.append(_table(fc.leaderboard))
    parts.append("<p class='note'>MASE &lt; 1 beats one-step seasonal-naive; 'coverage' "
                 f"should sit near {meta['interval_level']:.0%} for honest intervals. "
                 "Scores are the unweighted mean over series, then over folds.</p>")
    rec = meta.get("deployment_recommendation") or {}
    if rec.get("model") and rec["model"] != meta["champion"]:
        parts.append(f"<p class='note'><b>Deployment pick: {html.escape(str(rec['model']))}"
                     "</b> — within one standard error of the leaderboard winner, with fewer "
                     "moving parts to refit and monitor. The exported champion follows "
                     "champion_policy; set it to 'parsimonious' to ship the pick.</p>")
    elif rec.get("applied"):
        parts.append("<p class='note'><b>Champion selected with "
                     "champion_policy='parsimonious'</b>: the fewest fitted models within "
                     "one standard error of the best mean MASE.</p>")

    if getattr(fc, "reconciliation_report", None) is not None:
        parts.append("<h2>Hierarchical reconciliation</h2>")
        parts.append(_table(fc.reconciliation_report))
        parts.append(f"<p class='note'>Champion backtest scores after each "
                     f"reconciliation method; '{meta['reconciliation_method']}' was "
                     "applied to the final forecast, so children sum exactly to "
                     "their parents. Interval widths are unchanged by "
                     "reconciliation (centers move with the point adjustment).</p>")

    for name, fig in fc.figures.items():
        if fig is None:
            continue
        parts.append(f"<h2>{html.escape(name)}</h2>")
        parts.append(f"<img alt='{html.escape(name)}' "
                     f"src='data:image/png;base64,{_fig_to_b64(fig)}'>")

    parts.append("<h2>Panel</h2>")
    parts.append(_table(fc.panel.summary().reset_index(), max_rows=100))
    if meta["dropped_series"]:
        parts.append("<p class='note'>Dropped series: "
                     + html.escape(str(meta["dropped_series"])) + "</p>")

    parts.append("<h2>Decision audit</h2>")
    parts.append(_table(fc.log.to_frame(), max_rows=200))
    parts.append("</body></html>")

    path = Path(path)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
