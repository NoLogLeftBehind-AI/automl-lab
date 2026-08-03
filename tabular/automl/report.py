"""Self-contained HTML model report — the shareable deliverable.

Everything a reviewer needs without re-running the notebook: dataset summary,
guardrail decisions, leaderboard, holdout metrics with bootstrap CIs, the
evaluation and explainability figures, and the full decision audit. Figures are
embedded as base64 PNGs, so the file has zero external dependencies.
"""
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


def build_report(path: Path, *, title: str, meta: dict, leaderboard: pd.DataFrame,
                 holdout_metrics: dict, ci: pd.DataFrame, importance: pd.DataFrame,
                 figures: dict, decision_log: pd.DataFrame, profile: pd.DataFrame,
                 classification_report_text: str | None = None,
                 runtime_s: float | None = None) -> Path:
    m = meta
    badges = [f"task: {m['task']}", f"champion: {m['champion']}",
              f"metric: {m['primary_metric']}"]
    if runtime_s:
        badges.append(f"runtime: {runtime_s/60:.1f} min")
    parts = [f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
             f"<style>{_CSS}</style></head><body>",
             f"<h1>{html.escape(title)}</h1>",
             "".join(f"<span class='badge'>{html.escape(b)}</span>" for b in badges),
             f"<p class='note'>Generated {m['created_utc']} · python {m['versions']['python']}"
             f" · scikit-learn {m['versions']['sklearn']}</p>"]

    if m.get("suspected_leakage"):
        parts.append("<div class='warn'><b>⚠ Suspected leakage:</b> holdout performance is "
                     "near-perfect. Real problems rarely score this high — verify the top "
                     "features are genuinely available at prediction time.</div>")

    parts.append("<h2>Holdout performance</h2>")
    hm = pd.DataFrame([holdout_metrics]).T.rename(columns={0: "value"}).round(4)
    parts.append(_table(hm))
    if ci is not None and len(ci):
        parts.append("<p class='note'>95% bootstrap confidence intervals "
                     f"({m['config']['bootstrap_samples']} resamples):</p>")
        parts.append(_table(ci))
    parts.append(f"<p class='note'>{html.escape(m['holdout_metrics_note'])}</p>")
    if classification_report_text:
        parts.append(f"<pre>{html.escape(classification_report_text)}</pre>")

    parts.append("<h2>Leaderboard</h2>")
    parts.append(_table(leaderboard))
    parts.append(f"<p class='note'>{html.escape(m['cv_score_basis'])}</p>")
    rec = m.get("deployment_recommendation") or {}
    if rec.get("model") and rec["model"] != m["champion"]:
        parts.append(f"<p class='note'><b>Deployment pick: {html.escape(rec['model'])}</b> "
                     "— within one standard error of the leaderboard winner and simpler to "
                     "deploy, retrain, and explain. The exported champion follows "
                     "champion_policy; set it to 'parsimonious' to ship the pick.</p>")
    elif rec.get("applied"):
        parts.append("<p class='note'><b>Champion selected with "
                     "champion_policy='parsimonious'</b>: the simplest candidate within "
                     "one standard error of the best final score.</p>")

    for name, fig in figures.items():
        if fig is None:
            continue
        parts.append(f"<h2>{html.escape(name)}</h2>")
        parts.append(f"<img alt='{html.escape(name)}' src='data:image/png;base64,{_fig_to_b64(fig)}'>")

    if importance is not None and len(importance):
        parts.append("<h2>Feature importance (top 20)</h2>")
        parts.append(_table(importance.head(20)))
        parts.append("<p class='note'>Permutation importance splits credit between correlated "
                     "features — read related features as a group.</p>")

    parts.append("<h2>Data profile</h2>")
    parts.append(_table(profile.reset_index(), max_rows=80))

    parts.append("<h2>Decision audit</h2>")
    parts.append("<p class='note'>Every automated decision, in order:</p>")
    parts.append(_table(decision_log, max_rows=200))

    parts.append("</body></html>")
    path = Path(path)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
