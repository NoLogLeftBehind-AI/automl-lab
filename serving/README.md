# Model serving — FastAPI for exported AutoML artifacts

A small scoring service that turns any artifact exported by the
[tabular engine](../tabular/) into an HTTP API. The artifact is
self-describing (schema, preprocessing recipe, classes, drift reference,
tuned threshold all live in `metadata.json` / inside the pipeline), so this
service imports **no engine code** — it's ~200 lines you can read in one
sitting.

## Run

The service serves any exported artifact folder. If you don't have one yet,
run the [classification demo notebook](../tabular/notebooks/automl_classification.ipynb)
once (*Run All*) — it exports `artifacts_classification/` next to itself.
Then:

```bash
pip install -r requirements.txt -r ../tabular/notebooks/artifacts_classification/requirements.txt
MODEL_DIR=../tabular/notebooks/artifacts_classification uvicorn app:app --port 8000
```

or with Docker (the image runs as a non-root user and health-checks itself
against `/health`). `EXTRA_MODELS` installs the champion's model libraries:
lift the **exact pins** from the artifact's own `requirements.txt` — an
unpinned latest can skew against the pickled model, and the artifact is only
mounted at runtime, so the image build can't read those pins itself:

```bash
docker build -t automl-service \
  --build-arg EXTRA_MODELS="$(grep -E 'xgboost|lightgbm|catboost' \
      ../tabular/notebooks/artifacts_classification/requirements.txt | tr '\n' ' ')" .
docker run -p 8000:8000 -v $(pwd)/../tabular/notebooks/artifacts_classification:/app/artifact automl-service
```

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | readiness + model identity: champion, primary metric, holdout scores, training timestamp; 503 with the reason while no artifact is loadable |
| `POST /predict` | `{"rows": [{...}, ...]}` → predictions; classification adds per-class probabilities, and binary decisions already use the tuned threshold baked into the pipeline |
| `POST /drift` | `{"rows": [...]}` → per-feature PSI report against the training distribution, sorted worst-first |

Example against the Adult census demo artifact (send raw rows with the
original column names — extra columns are ignored, missing ones come back as
a 422 listing exactly what was expected):

```bash
curl -s localhost:8000/predict -X POST -H 'content-type: application/json' -d '{
  "rows": [{
    "age": 41, "workclass": "Private", "fnlwgt": 121772, "education": "Masters",
    "education_num": 14, "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty", "relationship": "Husband", "race": "White",
    "sex": "Male", "capital_gain": 0, "capital_loss": 0, "hours_per_week": 45,
    "native_country": "United-States"
  }]
}'
```

Unseen category levels are handled by the pipeline's encoders (that behavior
is tested).

## Tests

The suite trains its own tiny artifact with the tabular engine, so it needs
the engine's training environment (not just the service runtime) plus the
HTTP test client — all test-only, kept out of the image:

```bash
pip install -r requirements.txt -r ../tabular/requirements.txt httpx
python -m pytest tests -q
```

It then exercises health (including the 503 not-ready path), prediction
(probability sanity, unseen categories, schema errors, scoring failures
mapped to 400), drift detection and the 409 no-reference path, and cold-start
concurrency through FastAPI's TestClient — no pre-built artifact needed.

## Scope

Deliberately a demo-scale serving tier: the interesting part is the
self-describing artifact, not service plumbing. There is **no
authentication** (run it behind your gateway), one artifact per process
(scale horizontally / run one container per model), synchronous scoring only
(the artifact's generated `predict.py` covers batch), and tabular artifacts
only (forecasting artifacts ship their own `forecast.py` CLI). And since
`model.joblib` is a pickle — code that runs on load — only mount artifacts
you trained yourself or otherwise trust.
