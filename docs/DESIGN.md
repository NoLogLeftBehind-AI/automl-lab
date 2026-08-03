# Design decisions

The judgment calls behind the AutoML engines in this repository, and the
failure modes each one prevents. Everything here is enforced in code, and
the highest-risk behaviors carry regression tests — this page collects the
*why* in one place.

## Tabular engine (`tabular/`)

### The final leaderboard is re-scored on fresh folds

Optuna's `best_value` is the maximum over N adaptive trials on the same folds
used for selection — an optimistically biased estimate (winner's curse). And
when stage-1 screening runs on a capped sample, its scores aren't comparable
to full-data scores at all. So no number produced during search is ever used
for ranking: every candidate that reaches the final leaderboard — each
finalist in its chosen configuration, plus the blenders — gets one plain
cross-validation pass on *fresh folds*, and the champion comes from that
column only.

Note what that does and does not de-bias. It removes the winner's curse from
the *cross-family* comparison, which is where champion selection happens. It
does **not** de-bias the within-family tuned-vs-defaults gate: `tune_finalist`
compares Optuna's best-of-N against the default configuration on the *search*
folds, and only the winner of that comparison is carried into the fresh-fold
pass. So a tuned configuration that beat its own defaults by noise on the
search folds ships those tuned parameters, and the fresh-fold column scores
one configuration per finalist rather than both. Carrying both variants
forward would close it; that is not implemented.

Three further residual caveats are disclosed rather than hidden. Time-aware runs have
no reshuffled-fold variant — `TimeSeriesSplit` is chronological — so their
re-score removes the mixed sample-vs-full basis but necessarily reuses the
selection folds, which means the tuning search's selection bias is *not*
fully removed there (the decision log and the artifact's score-basis field
say so on every such run, and the leaderboard labels drop the "fresh-fold"
tag). Frozen GBM tree counts are chosen by early stopping on the full
training partition before the final CV pass re-uses those rows as test folds
— the same class of caveat as the leakage scan running once outside the CV
loop, with the locked holdout as the backstop for both. And while the
engine's own early-stopping probe validates on the chronological tail for
time-aware runs, sklearn-internal splits (HistGradientBoosting's validation
fraction, calibration and stacking folds) stay shuffled/stratified regardless
of time or group structure — that affects internal selection only; outer
folds and the locked holdout remain clean.

### The leaderboard is a measurement; the deployment pick is a judgment

Blends usually top AutoML leaderboards, and all three committed demos
reproduce the classic result. Winning the leaderboard is not the same as
earning deployment, though: a blend is three models to retrain and monitor
instead of one, a larger artifact, and lost explanations (SHAP is skipped for
non-plain pipelines — the demos' own decision logs record it). So the engine
separates the two roles. The leaderboard stays accuracy-ranked — blends must
still earn their rows honestly — and a parsimony rule computes the
**deployment pick**: the simplest candidate within one standard error of the
best score, ordered single model < voting < stacking. The pick lands in the
decision log, the report, and the artifact metadata on every run;
`champion_policy="parsimonious"` exports it as the champion. The forecasting
engine applies the same rule with fitted-model count as the complexity order.

The rule is conditional, not dogmatic, and on the committed demos it now
returns three different answers — which is the behaviour you want from a
statistic rather than a habit:

| Demo | Leaderboard winner | Deployment pick | Why |
|---|---|---|---|
| Adult | Voting(3), LogLoss 0.2751 | **Voting(3)** | the 0.0020 lead over default XGBoost (0.2771) is outside the paired band — the blend earned it |
| Diamonds | Stacked(3), RMSE 570.67 | **Voting(3)** (571.32) | 0.64 RMSE apart and indistinguishable, so the simpler *blend* ships; the single models (LightGBM_logy 578.73) are a resolvable 8 RMSE back and do not qualify |
| ERCOT | Blend(3), MASE 0.9612 | **Blend(3)** | beats Prophet (0.9995) on every fold; 0.038 lead vs a 0.011 band |

Diamonds is the case that shows the rule is doing real work: it steps
stacking → voting, a genuine simplification across an unresolvable margin,
without pretending that the 8-RMSE gap down to a single model is also noise.

The band is the actual one-standard-error rule of CART's pruning and glmnet's
`lambda.1se`, and getting there took a correction worth recording. The first
implementation banded on the marginal fold-to-fold standard *deviation*, which
is ~√k times wider than the standard *error* the cited rule uses — it would
have waved through margins the folds can resolve perfectly well. Worse, it
threw away the fact that every candidate is scored on the *same* splitter: the
fold scores are paired, so the right scale is the spread of their per-fold
*differences*, where fold difficulty is common to both models and cancels.
Both engines now band on `sd(differences)/√k`.

The correction changed the shipped answer on **all three** demos, which is the
point. The clearest case is ERCOT: the old band was 0.159 — wide enough to
swallow the blend's entire 0.038 MASE lead over Prophet — so the rule handed
deployment to Prophet. The blend actually beats Prophet on *every* fold, and
the paired band is 0.011: the lead is 3.4× the noise. Adult moved the same way,
and Diamonds moved from a single model to the simpler of the two blends. Every
one of those old picks was a simpler model than the evidence supported, because
a band ~√k too wide is biased in exactly one direction. A rule that discards
real evidence is not a conservative rule, it is a wrong one — and the tell was
that it reached the same conclusion on every dataset, which a statistic
sensitive to the data should not do.

### Tuning can never ship worse than defaults

The roster defaults are scored with the exact same objective before the
search, and the search result replaces them only by beating them. This isn't
paranoia: some defaults aren't even representable in bounded search spaces
(RandomForest's `max_depth=None`), so unbounded trust in `study.best_params`
can silently ship a strictly worse model — we measured it happening.

### Class weighting is a candidate, not a policy

LogLoss is the primary classification metric because calibrated probabilities
are what production consumers need. Class weighting distorts calibration by
construction, so on imbalanced data weighted variants *join the leaderboard*
rather than being forced onto every model. If weighting genuinely helps, it
wins; if it just reshuffles probability mass, the LogLoss column says so.

### Two leakage detectors, because each has a blind spot

A shallow decision tree has a hard R² ceiling on continuous targets — an
8-leaf tree caps near R² ≈ 0.96, so a tree-based scan with a 0.98 threshold
lets a *literal copy of the target* through (the original notebook this engine
replaced did exactly that). Rank correlation has no ceiling but misses
categorical proxies and non-monotonic leaks. On regression both run and either
can flag. The Spearman scan is gated to numeric features on regression targets,
though, so **classification runs one detector, not two** — including the Adult
demo, where the tree scan is the whole defence. That is the right gate (rank
correlation against encoded class labels is not meaningful) but it means the
"two detectors" property is a regression property, and the classification blind
spot is the tree's alone. Combinatorial leakage stays out of reach of any
univariate scan either way, so a near-perfect-holdout guardrail backstops both
after modeling.

### Early stopping owns the tree count

Searching `n_estimators` wastes trials rediscovering what a validation fold
gives for free. GBMs fit with early stopping; the discovered tree count is
frozen into a plain pipeline so nothing about early stopping leaks into the
artifact.

### The artifact is pure scikit-learn

Preprocessing lives inside the serialized pipeline; stateless preparation
(date decomposition, renames, text NaN-fill) is a recipe in the metadata that
the generated `predict.py` replays. Consequence: inference environments need
no code from this repository — a strong deployment property that constrains
several design choices upstream (e.g. no custom transformer classes).

## Forecasting engine (`forecasting/`)

### MASE is the primary metric, and seasonal-naive always competes

Scale-free (comparable across series), and defined relative to the one
forecast everyone secretly loses to. When the champion barely improves on
seasonal-naive the engine says so out loud — near-random-walk data is common,
and pretending a complex model earned its keep there is the most common
self-deception in applied forecasting.

### Seasonality is tested, not assumed

Calendar candidates (weekly/yearly for daily data) are confirmed by
bias-corrected seasonal strength: the variance share explained by phase means
after detrending, shrunk by the ~1/k baseline that k observed cycles explain
spuriously. The bias correction matters — a random walk scores ~0.27 raw at a
yearly candidate with 3 cycles of data. (The first implementation used ACF on
the differenced series; differencing erases long-period signal and it missed
even the yearly cycle in electricity load. The regression test remembers.)

The correction's limit, stated because the code cannot back the stronger
claim: `m/n` is the expected explained share under an **i.i.d.** null, and the
detrended residual of an integrated series is serially correlated, so its null
strength is both higher and more variable than `m/n`. The correction therefore
shrinks — but does not eliminate — spurious long-period confirmations on
random-walk-like data. Calibrating the threshold against a block-bootstrap or
phase-shuffle null is the right fix and is not implemented; the single-seed
regression test pins one draw, not a false-positive rate.

Detection runs once, on the full panel — a deliberate structural choice:
the seasonal period is a global property shared by every fold, and detecting
it per fold would let folds disagree on the roster itself. The folds stay
blind to their scored windows for everything the models *learn*; the
calendar-candidate choice is the one shared input.

### Rolling-origin backtests only

Training data always ends strictly before the scored window. There is no
shuffled CV anywhere in the forecasting engine, and per-series scores are
averaged so one large series cannot hide bad forecasts on a small one.

### Intervals must be earned

Coverage is measured in the backtests and printed next to the configured
level. Models without native intervals get empirical residual bands — labeled
as such, never dressed up as parametric confidence.

"Earned" has to include the sample size behind the band, and this is where the
first implementation over-promised. A distribution-free band built from `n`
residuals per step is at best "wider than all `n` of them", and a fresh draw
exceeds the sample maximum with probability `1/(n+1)` — so its coverage
ceiling is `n/(n+1)` no matter what level was configured. At the default 3
backtest folds a per-series band labelled 90% tops out near **75%**; at
`n_backtests=1` it tops out at 50%. The engine now computes that ceiling,
warns when the configured level exceeds it, ships it in the artifact metadata
as `interval_attainable_level`, and annotates the report badge. Two things it
still does not do: pool residuals across steps after scale normalisation
(which would trade the ceiling for an exchangeability assumption), and measure
these bands' realised coverage — impossible on the residuals they were fitted
on, which is why the leaderboard shows `NaN` rather than a flattering number.

### Reconciliation is evaluated, then applied

Bottom-up / OLS / WLS / MinT are scored on the champion's own backtest folds
before anything touches the final forecast, with MinT's error covariance
estimated leave-fold-out so it can't flatter itself. On the ERCOT demo this
showed bottom-up *improving* accuracy 2.2% while enforcing coherence, MinT a
close second at 1.2% — and OLS/WLS hurting, which is exactly why the choice
is measured instead of assumed.

## Benchmarks (`benchmarks/`)

- **Identical partitions**: challengers train on exactly the rows the engine's
  deterministic split defines, and are scored by the harness from their raw
  predictions; each engine's row is its own evaluation through the same
  shared metric code on the same partition/folds (both benchmark READMEs
  state this precisely).
- **Matched optimization targets and budgets** where the system allows it.
- **Handicaps are disclosed, not hidden**: AutoGluon-TS ran without Chronos
  (no Hugging Face access in the benchmark environment); the engine's
  fold-selected champion advantage is quantified by also reporting its
  individual members.
- **A win on one dataset is a demonstration, not a study** — both benchmark
  READMEs say so above the fold.
- **Point estimates, not resolved differences**: one seeded run per system and
  no paired significance test, so the tables state an ordering rather than a
  proven gap. The tabular benchmark README names the test that *would* settle
  it (a paired bootstrap over per-row holdout differences) and says plainly
  that it is not computed here.

## What this project deliberately leaves out

This document covers the decisions that were *made*. The decisions about what
not to build — experiment tracking, orchestration, tabular deep learning,
conformal intervals, serving hardening, scale, fairness auditing — are their
own document, with the triage criterion stated up front and a reason attached
to every omission:

**→ [SCOPE.md](SCOPE.md)**

The short version: this is a self-funded project built on personal time, so
the scoping is part of the engineering. The hours went where measurable
honesty lives — partitioning, leakage, selection bias, interval coverage,
benchmark discipline — and the omissions are the pieces a production team
fills with standard tooling.
