# Deployment benchmark — what the accuracy gap buys

The [tabular benchmark](../tabular/) measures what the engine gives up in
accuracy: 0.8% on Adult, 0.78–4.3% on Diamonds. This folder measures what it
buys back, because "a far lighter deployment story" was an assertion until
somebody put numbers on it.

Reproduce with:

```bash
pip install -r ../../serving/requirements.txt httpx
python measure.py --artifact /path/to/an/exported/artifact
```

Measured on 4 CPU cores / 15 GB RAM, no GPU — the same box as the tabular
benchmark. Committed numbers are in [`results/results.json`](results/results.json),
from the Diamonds demo artifact (champion `Stacked(3)`).

## Artifact size

| Item | Size |
|---|---|
| **Whole artifact folder** | **4.88 MB** |
| `model.joblib` (pipeline + preprocessing + threshold) | 4.53 MB |
| `report.html` (self-contained, figures inlined) | 0.34 MB |
| `metadata.json`, `predict.py`, `drift_check.py`, `Dockerfile`, `requirements.txt` | 0.02 MB |

## Scoring latency

`POST /predict` against [`serving/app.py`](../../serving/) under uvicorn over
real HTTP (not `TestClient`), after a 5-request warmup.

| Request | p50 | p95 | per row |
|---|---|---|---|
| 1 row | 28.4 ms | 39.2 ms | — |
| 1,000 rows | 58.5 ms | 70.8 ms | **0.058 ms** |

Single-row latency is dominated by per-request overhead (JSON parse, DataFrame
construction, recipe replay), not by the model: 1,000× the rows costs 2.2× the
time.

## Inference dependency closure

What a fresh inference environment downloads, resolved with
`pip install --dry-run --ignore-installed` and sized from each wheel's
`Content-Length`. This is the honest way to compare "what you have to ship",
and it is the number that most surprised me.

| Environment | packages | wheels |
|---|---|---|
| Engine artifact — pure-sklearn champion (floor) | **9** | **74 MB** |
| Engine artifact — as pinned for this champion (`Stacked(3)`: LightGBM + HistGB + CatBoost) | 21 | 209 MB |
| `autogluon.tabular` (base) | 36 | 162 MB |
| `autogluon.tabular[lightgbm,xgboost,catboost]` | 43 | 693 MB |
| `autogluon.tabular[+fastai]` — what [`benchmarks/tabular/requirements.txt`](../tabular/requirements.txt) installs | 124 | 4,527 MB |

Against the environment the tabular benchmark actually ran in, the artifact is
**21.7× lighter and 6× fewer packages**; against AutoGluon's GBM extras it is
3.3× lighter. Against a *base* `autogluon.tabular` install it is still 1.3×
heavier (209 vs 162 MB) — because a base install carries no gradient-boosting
library at all, while this champion genuinely needs two. That row stays in the
table rather than being dropped for being inconvenient.

The claim that survives on every axis is the coupling one, not the size one:
the artifact unpickles with zero imports from this repository (verified in
[`serving/tests`](../../serving/tests/)), which is what lets it outlive the
engine version that produced it.

**How the 209 MB row got there.** It used to be 694 MB.
`pinned_requirements()` pinned every model library present in the *training*
environment rather than the ones the champion uses, so this artifact shipped
`xgboost` (131.7 MB, plus `nvidia-nccl-cu12` at 303.4 MB behind it) for a
champion that never calls it, and `pyarrow` (50.1 MB) purely so `predict.py`
could accept parquet input. Pins are now derived by walking the exported
pipeline — Pipeline steps, ColumnTransformer branches, blender members and
`TransformedTargetRegressor` wrappers — and parquet ships commented out as an
opt-in. That removed 485 MB without touching the model. The measured
before/after is why this was worth doing rather than arguing about.

## What is not measured here

- **Container image size.** Needs a Docker daemon, which the measurement
  environment does not have. `serving/Dockerfile` exists and is exercised in
  CI, but no image size is claimed.
- **AutoGluon's scoring latency.** Needs a trained AutoGluon model served
  behind an equivalent endpoint. The latency table above is an absolute
  measurement of this artifact, not a comparison.
- **Cold-start / memory footprint.** Not instrumented.
