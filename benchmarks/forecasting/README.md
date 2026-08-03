# Forecasting Benchmarks — the forecasting engine vs statsforecast and AutoGluon-TS

How does the [forecasting engine](../../forecasting/) stack up
against the strongest open-source forecasting libraries? Same philosophy as
[the tabular benchmark](../tabular/): identical splits, shared
scoring code, honest reading.

## Methodology

- **Identical folds.** The engine's own rolling-origin cutoffs define the
  3 backtest windows (28 days each) on the daily ERCOT 8-region panel
  (2016–2021) — the same raw panel and fold protocol as the forecasting demo,
  but run on the **8 base regions only**: no synthesized total series, no
  hierarchical reconciliation, no holiday calendars, so every system competes
  on exactly the same inputs. (This is why the engine's numbers here are not
  directly comparable to the demo notebook's, which adds all three.) Every
  system trains strictly before each cutoff and forecasts the same 28 days.
- **Shared scoring.** Challengers are scored by the engine's metric code
  (MASE scaled to one-step seasonal-naive, unweighted mean over series then
  folds) from their raw fold predictions. The engine's row is its own
  leaderboard champion — produced by that same scoring code on those same
  folds, not an independent re-computation; the harness asserts both sides
  use the same MASE season.
- **Systems at their defaults**: the engine's leaderboard champion;
  Nixtla `statsforecast` AutoARIMA / AutoETS / AutoTheta (season_length=7,
  the library's celebrated fast auto-models); AutoGluon-TimeSeries
  `medium_quality` (240 s/fold) and `best_quality` (600 s/fold).
- **One stated handicap:** this environment cannot reach Hugging Face, so
  AutoGluon-TS runs **without its pretrained Chronos models** — a genuine
  disadvantage for AutoGluon (Chronos is a headline feature of its 1.5
  release). Treat its numbers as a floor, not a ceiling.
- Hardware: 4 CPU cores, no GPU.

Reproduce:

```bash
pip install -r requirements.txt
python benchmark.py --all       # ~1h on 4 cores
python benchmark.py --table     # compact table from results.json (the README
                                # table is formatted from the same file)
```

Raw results (metrics, fit times, run timestamps, champions) are committed in
[`results/results.json`](results/results.json).

## Results

Daily ERCOT load, 8 regions, 28-day horizon, 3 rolling-origin folds, 4 CPU cores:

| System | MASE ↓ | WAPE ↓ | RMSE ↓ | wall-clock |
|---|---|---|---|---|
| **Forecasting engine** — Blend(XGBoost, Prophet, LightGBM) | **0.940** | **6.4%** | **541** | 1.5 min |
| AutoGluon-TS `best_quality` (no Chronos) | 1.046 | 7.5% | 588 | 20.9 min |
| AutoGluon-TS `medium_quality` (no Chronos) | 1.167 | 8.5% | 659 | 7.2 min |
| statsforecast AutoARIMA | 1.268 | 9.5% | 805 | 3.2 min |
| statsforecast AutoTheta | 1.320 | 9.8% | 768 | 0.1 min |
| statsforecast AutoETS | 1.321 | 9.8% | 776 | 0.2 min |

## Reading the numbers honestly

**The engine wins this benchmark — with three caveats stated before any
celebration:**

1. **AutoGluon-TS is handicapped here.** Its pretrained Chronos models — a
   headline strength of the 1.5 release — couldn't download in this
   environment. Its numbers are a floor. On open infrastructure, expect it
   closer, possibly ahead.
2. **The engine's champion enjoys a mild selection advantage**: its blend was
   *chosen* on the same folds scored here, while every challenger fields one
   fixed configuration. The honest check: the engine's individual roster
   members score 0.96 (XGBoost-global) and 1.01 (Prophet) on these folds —
   still ahead of every challenger — so the win survives removing the
   advantage. Caveat on that check itself: those two numbers are quoted from
   the run's console leaderboard; `results.json` commits only the champion
   row, so they are not independently checkable from this repo until a re-run
   commits the full leaderboard (queued, alongside the tabular
   harness-scorer routing).
3. **One dataset is a demonstration, not a study.** This panel plays to the
   engine's strengths: 8 related series (global GBMs borrow strength across
   them), a dominant yearly cycle (Prophet and day-of-year features), a
   28-day horizon. statsforecast's per-series statistical auto-models are
   built for a different regime — short horizons, enormous series counts,
   millisecond fits — and its 6-second AutoTheta fit is its own kind of
   impressive.

**Why the result makes sense anyway:** the engine's edge is *roster breadth
with honest selection* — it fields per-series statistical models AND global
gradient boosting AND Prophet, backtests them identically, and blends the
winners. None of the challengers spans that space in one call: statsforecast
is statistical-only; AutoGluon-TS (without Chronos) leans on local models and
deep learning that need more data or compute to shine. Breadth-plus-discipline
is exactly the thesis of this project, and here it beat both a faster
specialist and a bigger generalist.

**Where the challengers still win:** statsforecast at scale (thousands of
series, its numba-compiled models are untouchable on speed-per-series) and
probabilistic rigor; AutoGluon-TS wherever Chronos/deep models can load, on
messier panels, and with GPU budgets. The engine's counter-offer remains the
same as the tabular engine's: an auditable decision log, honest coverage
reporting, hierarchical reconciliation, and a codebase you can read in an
afternoon.

**Provenance of `results.json`.** These rows were produced on 2026-07-12. The
forecasting engine has changed since — most consequentially the parsimony
deployment pick and a correction to the band it uses — so the committed
numbers describe the engine as of that date, not as it stands today. The
comparison itself is unaffected in one specific sense: every row reports its
system's *leaderboard champion*, and `champion_policy` defaults to
accuracy-ranked, so the deployment-pick work does not move the MASE column.
Re-running `benchmark.py --all` at the current commit is still the only way to
get rows generated by the current engine. (The same note, with its own dates,
is on the [tabular benchmark](../tabular/README.md#reading-the-numbers-honestly).)
