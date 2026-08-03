# Scope — what this project includes, what it leaves out, and why

This is a self-funded project built on personal time, alongside a full-time
job. That constraint is not a disclaimer bolted onto the end; it is the
design input that shaped everything below. A portfolio project that tried to
be a complete ML platform would be a worse portfolio project — it would spend
its hours on integration work that demonstrates familiarity with other
people's tools, and it would still be a worse platform than the tools it
integrated.

So the scope was triaged against a single criterion:

> **Signal about modeling judgment per hour of work.**

Everything on the "in" list is something where a wrong decision quietly
produces a number that looks fine and isn't. Everything on the "out" list is
either standard integration work, or something whose absence changes nothing
about the judgment on display.

This document is the cross-cutting triage. Local caveats — the ones that
belong next to the code that has them — live with their components:
[tabular limitations](../tabular/README.md#limitations),
[forecasting limitations](../forecasting/README.md#limitations),
[serving scope](../serving/README.md#scope). The reasoning behind the
positive choices is in [DESIGN.md](DESIGN.md).

---

## What the hours went into

The spend concentrates on the places where evaluation lies to you:

| Area | Why it earned the hours |
|---|---|
| **Partitioning** | A locked holdout, scored once; time-aware and group-aware strategies so no entity or future row straddles a split. Most published "AutoML beats X" numbers die here. |
| **Leakage** | Two detectors with different blind spots, plus a post-modeling guardrail for the combinatorial leaks neither can see. The [module docstring](../tabular/automl/leakage.py) records the measured failure that motivated the second detector. |
| **Selection bias** | Optuna's best-of-N is optimistically biased and sample-based scores aren't comparable to full-data scores, so every candidate is re-scored in one fresh-fold pass before anything is crowned. |
| **Interval coverage** | Measured in backtests and printed next to the configured level; models without native intervals get residual bands *labeled as such*, never dressed up as parametric confidence — and since a band built from `n` residuals cannot exceed `n/(n+1)` coverage whatever level you ask for, the engine computes that ceiling, warns when the configured level sits above it, and prints it on the report badge rather than advertising a number it cannot reach. |
| **Benchmark discipline** | Identical partitions, matched optimization metrics, matched time budgets, challenger metrics computed by shared harness code — and the imbalances that remain are disclosed in the benchmark READMEs rather than smoothed over. When measuring one of them showed the engine's own pruner caused 82% of a gap previously attributed to the challenger, the [ablation](../benchmarks/tabular/README.md#ablation-the-pruner-picks-the-wrong-twin-on-diamonds) shipped and the headline table kept the worse default number. |
| **The cost of the trade** | "Lighter to deploy" was an assertion until [`benchmarks/deployment/`](../benchmarks/deployment/) put artifact size, `/predict` latency and inference dependency closure next to AutoGluon's — including the operating point where the artifact is the heavier one. |
| **Auditability** | Every automated decision is logged, shipped inside the artifact, and rendered in the report. Automation that can't explain itself is not a deliverable. |

## What it deliberately leaves out

### Platform and MLOps surface

- **Experiment tracking and a model registry** (MLflow / W&B class). The
  artifact folder *is* the run record here: `metadata.json` carries the full
  decision log, scores, and config; `report.html` is the human view. A
  tracking server earns its keep across teams and hundreds of runs. Wiring
  one into single-run engines would demonstrate integration, not judgment.
- **Orchestration and automated retraining** (Airflow / Prefect class). The
  engines end at a deployable artifact plus a drift signal (`drift_check.py`,
  the service's `/drift`); acting on that signal belongs to the consuming
  pipeline. Likewise, CI proves correctness here — delivery (CD, registries,
  releases) is out of scope.
- **Serving hardening** — authentication, multi-model routing, streaming,
  autoscaling. The service exists to demonstrate the artifact contract
  (self-describing, zero engine imports); it expects to run behind your
  gateway, one artifact per process. The
  [serving README's Scope note](../serving/README.md#scope) is authoritative.
- **No feature store, no data-versioning tooling** ([DATA.md](DATA.md)
  documents the URL-stability trade-off instead), **no Kubernetes manifests,
  no rate limiting, no PyPI release, no docs site.** Standard integrations a
  production team bolts on, with little to say about modeling judgment.

### Modeling surface

- **Tabular deep learning** (MLP / TabNet / TabPFN class). The budget is
  CPU-only, and on this data class the benchmark shows tuned GBM ensembles
  are the frontier — the roster spends its complexity budget where the
  returns are measured. Deep forecasters (N-BEATS / TFT / Chronos) are out
  for the same reason.
- **Prediction intervals for tabular regression** (conformal / quantile).
  The asymmetry with the forecasting engine's "intervals must be earned"
  stance is real and worth naming: tabular regression ships point
  predictions, with bootstrap CIs on the *metrics* only. Split-conformal
  intervals are the natural next increment and would follow the same
  measured-coverage discipline.
- **Online / incremental learning.** Everything is batch; retrain-on-drift is
  the intended loop.
- **Intermittent-demand forecasters** (Croston / TSB) and **automated tuning
  for the ML forecasters** beyond the SARIMAX order search.

### Evaluation surface

- **Multi-seed benchmark replication and paired significance testing.** Each
  comparison is a single seeded run per system, and no paired bootstrap over
  per-row holdout differences is computed — so the benchmark tables are point
  estimates, not resolved differences. Rather than average that away with
  compute this project does not have, each benchmark README states exactly
  what its numbers do and do not establish.
- **More datasets.** Two tabular datasets and one forecasting panel are a
  demonstration, not a study. The deliberate investment is in the *harness* —
  adding a dataset is five lines in `DATASETS`.
- **Fairness / subgroup auditing.** Permutation importance and SHAP explain
  *what the model uses*, not whether it is equitable across groups. No
  slicing metrics ship here, and explanations are not a fairness audit.
  This is the omission most likely to matter in a regulated setting, and it
  is an omission, not an oversight.

### Scale

Everything is in-memory pandas/scikit-learn, demonstrated to ~50k rows and an
8-series panel. Screening subsamples above 20k rows; SARIMAX caps its history
window. **The judgment on display does not change at 10× the rows — the
infrastructure would.** Out-of-core training, distributed search, and GPU
support are all absent, and adding them would not make the evaluation
discipline any more or less correct.

---

## If there were more hours

Not a roadmap — nothing here is promised. It is the order the next
increments would come back in, which is itself a statement about what I think
matters:

1. **Split-conformal intervals for tabular regression** — closes the stated
   asymmetry with the forecasting engine, and the measured-coverage
   machinery to validate them already exists.
2. **Routing the tabular engine's own predictions through the harness
   scorer** — the benchmark already computes challenger metrics that way; the
   engine's row is its own evaluation through the same metric functions.
   Closing that last gap is disclosed as queued in the
   [benchmark README](../benchmarks/tabular/README.md#methodology).
3. **A paired bootstrap over per-row holdout differences, plus multi-seed
   replication** — the two cheapest things that would turn the benchmark
   tables from point estimates into resolved comparisons. The benchmark
   README is explicit that neither is computed today.
4. **A third tabular dataset with a genuinely messy schema** — the current
   two are clean, which flatters every system in the comparison equally but
   exercises the guardrails less than they deserve.
5. **A multivariate tiebreak for correlation pruning.** The current univariate
   rule keeps the wrong twin on Diamonds and it costs 18.3 RMSE — 82% of that
   dataset's gap to AutoGluon
   ([ablation](../benchmarks/tabular/README.md#ablation-the-pruner-picks-the-wrong-twin-on-diamonds)).
   Scoring the two candidates jointly against the target would have caught it.
   This one moved up the list because measuring it, rather than reasoning
   about it, is what revealed the size.
*(Item 6 was "pin the artifact's requirements against the champion's actual
estimator tree rather than whatever was installed at training time." It was
[measured](../benchmarks/deployment/) at 485 MB of needless closure, so it got
done: pins now come from walking the exported pipeline, and the Diamonds
artifact's inference closure went 694 MB → 209 MB with no change to the model.
Left here because the list is more useful as a record of what measurement
turned into work than as a wish list.)*

## How to read the omissions

The through-line: the hours went where measurable honesty lives —
partitioning, leakage, selection bias, interval coverage, benchmark
discipline — and the omissions are the pieces a production team fills with
standard tooling.

The same triage applied to build-vs-buy is in the tabular README's
"[When a commercial platform still makes sense](../tabular/README.md#when-a-commercial-platform-still-makes-sense)".
