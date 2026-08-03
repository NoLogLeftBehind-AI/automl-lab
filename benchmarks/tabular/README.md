# AutoML Benchmarks — the tabular engine vs AutoGluon

How close does a readable, single-repo AutoML engine get to the strongest
open-source AutoML system? This folder measures it, on the same public
datasets the [tabular engine](../../tabular/) demos use.

## Methodology

- **Identical partitions.** The engine's deterministic load/split (seed 42,
  duplicates removed) defines the train and holdout rows; every challenger is
  fit on exactly those training rows and scored on exactly that holdout.
  Challenger metrics are computed by `benchmark.py` from raw holdout
  predictions; the engine's row reports its own holdout evaluation — the same
  sklearn metric functions on the identical partition. (Routing the engine's
  raw predictions through the harness scorer too is queued for the next
  re-run; the committed numbers were produced as described here.)
- **Identical rows, but not identical feature sets — and the asymmetry runs
  against the engine.** Each system starts from the same partition and then
  applies its own pipeline, which is the honest end-to-end comparison. The
  consequence on Diamonds is worth stating plainly: the engine's correlation
  pruner drops `carat`, `x`, and `z` (each |ρ| > 0.98 with `y`, which scored a
  stronger single-feature target association), so **the engine competes on 6
  features while both challengers see all 9 — including `carat`, the strongest
  single predictor of diamond price.** Adult is unaffected. The cost of that
  choice is measured in [the ablation below](#ablation-the-pruner-picks-the-wrong-twin-on-diamonds):
  it is most of the Diamonds gap.
- **Matched optimization targets.** AutoGluon is told to optimize the same
  metric the engine selected (LogLoss / RMSE) — and the harness fails fast if
  a new dataset flips the engine to a different primary metric — so nobody
  wins by optimizing something easier.
- **Two AutoGluon operating points.** `medium_quality` (its fast default) and
  `best_quality` with a fixed 1300 s budget, matched to the engine's own
  Adult fit time in this benchmark run (22.4 min). On Diamonds the engine's
  run went ~24% longer (26.8 vs 21.6 min) — an imbalance in the engine's
  favor, disclosed here, and AutoGluon won that dataset anyway.
- **The two wall-clocks are not like for like, and the asymmetry costs the
  engine.** AutoGluon's timer wraps `predictor.fit()` alone; the engine's wraps
  the whole `AutoML.run()` — screening, tuning, the fresh-fold re-score, plus
  permutation importance, PDP, SHAP, the HTML report and the artifact export.
  Several minutes of every engine row is deliverable-building that AutoGluon
  is not charged for, so "time-matched" means matched *budgets*, not matched
  work. Timing to `evaluate()` would make the comparison tighter; the
  committed numbers are the conservative version.
- **Hardware:** 4 CPU cores, 15 GB RAM — no GPU for anybody.

Reproduce with:

```bash
pip install -r requirements.txt
python benchmark.py --all            # ~2h on 4 cores
python benchmark.py --table          # compact table from results.json (the
                                     # README tables are formatted from the same file)
```

Raw results (metrics, fit times, library versions, champions) are committed in
[`results/results.json`](results/results.json).

## Results

**Adult census income** — binary classification, optimize LogLoss
(39,032 train / 9,758 holdout rows)

| System | LogLoss ↓ | ROC AUC ↑ | fit time | champion |
|---|---|---|---|---|
| Base-rate baseline (predicts the training prior) | 0.5504 | 0.5000 | — | — |
| XGBoost, defaults | 0.2721 | 0.9306 | 0.3 s | — |
| **Tabular engine** | 0.2658 | 0.9343 | 22.4 min | Voting(3) |
| AutoGluon `medium_quality` | 0.2642 | 0.9349 | 2.5 min | WeightedEnsemble_L2 (11 models) |
| AutoGluon `best_quality` (time-matched) | **0.2636** | **0.9353** | 21.6 min | WeightedEnsemble_L2 (12 models) |

**Diamonds** — regression, optimize RMSE
(43,035 train / 10,759 holdout rows)

| System | RMSE ↓ | MAE ↓ | R² ↑ | fit time | champion |
|---|---|---|---|---|---|
| Mean-predictor baseline | 3,997 | 3,041 | 0.000 | — | — |
| XGBoost, defaults | 553.7 | 277.9 | 0.9808 | 0.6 s | — |
| **Tabular engine** | 538.3 | 280.8 | 0.9819 | 26.8 min | Stacked(3) |
| AutoGluon `medium_quality` | 520.0 | 256.2 | 0.9831 | 5.6 min | WeightedEnsemble_L2 (9 models) |
| AutoGluon `best_quality` (time-matched) | **516.0** | **255.6** | **0.9833** | 21.6 min | WeightedEnsemble_L3 (13 models) |

## Ablation: the pruner picks the wrong twin on Diamonds

`carat` and `y` are near-duplicates (|ρ| = 0.996), so the correlation pruner
keeps exactly one. It kept `y`, because `y` scored the stronger single-feature
target association. Two runs of the same engine, same seed, same settings,
differing only in `force_keep_columns`, measure what that cost
(`python ablation_carat.py` → [`results/ablation_carat.json`](results/ablation_carat.json)):

| Diamonds run | features kept | RMSE ↓ | vs AutoGluon best |
|---|---|---|---|
| Default pruning (keeps `y`) | cut, color, clarity, depth, table, **y** | 538.4 | +4.33% |
| `force_keep_columns=["carat"]` | **carat**, cut, color, clarity, depth, table | **520.0** | **+0.78%** |

**18.3 RMSE — 82% of the engine's Diamonds gap to AutoGluon — is the pruner's
choice of twin, not AutoGluon's ensembling.** Both runs train on six features,
so this is not a feature-count handicap; it is one automated decision, made on
a univariate criterion, going the wrong way. `y` is a diamond's width in
millimetres and `carat` is its weight — a domain expert picks `carat` without
hesitating, and the single-feature association score does not encode that.

The honest reading cuts both ways. It shrinks the accuracy gap this benchmark
reports, and it is also the sharpest known limitation of the engine's pruning
heuristic: greedy pairwise pruning on single-feature target association is
cheap and usually harmless, but when it is wrong it is expensive, and nothing
downstream catches it. Keeping both twins and letting the model sort them out
would avoid this; it would also give up the collinearity control the rule
exists for. The rule stays, the failure is documented, and
`force_keep_columns` is the escape hatch — surfaced in the decision log on
every run so the choice is visible rather than silent.

The headline table above is left as the default pipeline produced it. Reporting
the better number as if it were the default would be exactly the kind of
flattering evaluation this project is about avoiding.

## Reading the numbers honestly

**AutoGluon wins on raw accuracy — say it plainly.** Time-matched
`best_quality` finishes ahead of the engine by **0.8%** on Adult LogLoss and
**4.3%** on Diamonds RMSE (the engine's score expressed as a percentage above
AutoGluon's — the reference it is being measured against). Even its
fast `medium_quality` preset edges the engine out in a fraction of the
wall-clock. That is the expected result: AutoGluon is the
product of years of engineering, and its multi-layer stacking over bagged
out-of-fold predictions is a fundamentally stronger ensembling scheme than the
engine's stack/vote-of-three-finalists. That explanation holds for Adult. It
does **not** hold for Diamonds: the ablation above shows 82% of that 4.3% is
the engine's own correlation pruner discarding `carat`, and with `carat` kept
the gap is 0.78% — the same order as Adult's. Attributing the Diamonds gap to
AutoGluon's stacking would have been wrong, and only measuring it showed that.

**The engine holds its ground where it was designed to.** It beats default
XGBoost on the primary metric on both datasets, finishes 0.8% and 4.3%
behind AutoGluon's best, and its champions are the same *kind* of model
AutoGluon crowns (ensembles of diverse tuned GBMs) — the leaderboard
machinery reaches the same conclusions, with less firepower behind them.

**Where the wall-clock goes is a design choice.** AutoGluon spends its budget
fitting more models; the engine deliberately spends much of its 22–27 min on
selection honesty — a two-stage leaderboard, a de-biased fresh-fold re-score
of every candidate, out-of-fold calibration and threshold work — and on
producing its deliverables (report, scoring script, drift reference).

**What you get for that 0.8–4.3%:** a pure-scikit-learn artifact with no
AutoML framework dependency at inference time (an AutoGluon deployment
imports AutoGluon and ships its whole model store), a generated batch scorer
and PSI drift checker, a self-contained HTML model report, a decision log of
every automated choice, and an engine small enough to read in an afternoon.

Both sides of that trade are now measured — the deployment side in
[`benchmarks/deployment/`](../deployment/), and it does not all go our way:

| | Engine artifact | AutoGluon |
|---|---|---|
| Deployable unit | **4.88 MB** folder | model store + framework |
| Inference closure (benchmark env) | **21 pkgs / 209 MB** | 124 pkgs / 4,527 MB |
| Inference closure (base install) | 21 pkgs / 209 MB | **36 pkgs / 162 MB** |
| Scoring 1,000 rows | **58 ms** p50 (0.058 ms/row) | not measured |
| Imports the training framework? | **no** | yes |

Against the environment this benchmark actually ran in, the artifact is 21.7×
lighter; against a *base* `autogluon.tabular` install it is still 1.3× heavier,
because that base carries no gradient-boosting library while this champion needs
two. Both rows are in the table. The claim that survives on every axis is the
coupling one: the artifact unpickles with zero imports from this repository, so
it outlives the engine that produced it. Whether that is worth 0.8–4.3% of a
metric is a per-project call — but it is now a call made against numbers on both
sides.

**Reproducibility.** Neither side is bit-reproducible across runs. AutoGluon
1.5 exposes no fit-level random seed, and the engine — although seeded
end-to-end — trains its gradient-boosting models multithreaded (`n_jobs=-1`),
which is not bit-deterministic — so the engine is seed-reproducible only when
code, library versions, and hardware are all held fixed.

Held fixed, it reproduces exactly. The `default_pruning` arm of
[the ablation](#ablation-the-pruner-picks-the-wrong-twin-on-diamonds) is the
demo notebook's Diamonds configuration re-run on 2026-08-01, and it returned
RMSE 538.3501, MAE 279.2727, R² 0.98186 — matching the
[committed 2026-07-30 report](../../tabular/examples/report_regression.html) to
four decimals on every metric. That is one paired re-run on one dataset, not a
replication study, but it is a measurement rather than an assertion.

**Provenance of `results.json`.** The committed benchmark rows were produced
on 2026-07-11 against scikit-learn 1.7.2. The demo reports in
[`tabular/examples/`](../../tabular/examples/) were re-executed on 2026-07-30
against scikit-learn 1.9.0, after eight commits to `tabular/automl/` —
including a log-target early-stopping fix and the parsimony deployment pick.
So the small Diamonds difference between them (538.33 vs 538.35) is a
*version* difference, not seed noise — the exact-match re-run above is what
seed noise actually looks like here. Re-running `benchmark.py --all` at the
current commit is the only way to get challenger rows that match the engine as
it stands today.

**Measurement precision — the gaps above are not established as significant.**
Each row is a single run of a single seed, and the engine's own 95% bootstrap
confidence intervals on the holdout are wide next to the gaps being discussed
(Adult LogLoss 0.2658, CI [0.2564, 0.2761]; Diamonds RMSE 538.4, CI
[514.8, 560.4] — both in the committed
[demo reports](../../tabular/examples/)). Overlapping intervals are not
evidence of a tie either: every system here is scored on the *same* holdout
rows, so the correct test is a paired bootstrap over per-row differences,
which is much tighter than the marginal intervals and is **not** computed in
this repo. Read the tables as point estimates with the ordering stated
plainly — AutoGluon ahead on both datasets — and not as resolved differences.
Multi-seed replication and a paired test are the two cheapest things that
would settle it; both are listed as deliberate omissions in
[SCOPE.md](../../docs/SCOPE.md).

## What this does and doesn't show

- AutoGluon is a multi-year, multi-maintainer project with neural models,
  sophisticated multi-layer stacking, and heavy engineering for robustness at
  scale. Matching or approaching it on clean tabular data does **not** make
  this engine a general replacement for it.
- Two datasets is a demonstration, not a study. The point is the *harness*:
  identical partitions, matched metrics, matched budgets — add a dataset by
  adding five lines to `DATASETS` in `benchmark.py`.
- What the engine buys instead of leaderboard supremacy: a fully auditable
  decision log, a pure-sklearn artifact with no framework dependency at
  inference time, generated scoring/drift/report deliverables, and a codebase
  small enough to read in an afternoon.
