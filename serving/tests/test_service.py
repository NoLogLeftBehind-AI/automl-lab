"""Service tests: train a tiny artifact with the tabular engine, then exercise
the API end-to-end through FastAPI's TestClient."""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tabular"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
    from automl import AutoML, AutoMLConfig
    rs = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame({
        "x1": rs.randn(n), "x2": rs.randn(n),
        "cat": rs.choice(["a", "b", "c"], n),
        "flag": rs.rand(n) < 0.5,           # exercises the boolean_as_int recipe step
    })
    logit = 1.5 * df.x1 + (df.cat == "a") * 1.0 + 0.8 * df.flag
    df["y"] = (rs.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)
    art = tmp_path_factory.mktemp("svc") / "artifact"
    cfg = AutoMLConfig(target="y", task="classification", artifact_dir=str(art),
                       cv_folds=3, optuna_trials=2, optuna_timeout=20,
                       n_finalists=2, bootstrap_samples=100, ensemble=False, n_jobs=2)
    AutoML(cfg).run(df)
    return art


@pytest.fixture(scope="module")
def client(artifact_dir):
    os.environ["MODEL_DIR"] = str(artifact_dir)
    import app as app_module
    importlib.reload(app_module)          # pick up MODEL_DIR set by this test
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["task"] == "classification"
    assert "LogLoss" in body["holdout_metrics"]


def test_predict_returns_labels_and_probabilities(client):
    rows = [{"x1": 2.0, "x2": 0.1, "cat": "a", "flag": True},
            {"x1": -2.0, "x2": 0.0, "cat": "c", "flag": False}]
    r = client.post("/predict", json={"rows": rows})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2
    assert len(body["predictions"]) == 2
    assert set(body["probabilities"][0].keys()) == {"0", "1"}
    # strongly positive logit row should be the likelier positive
    assert body["probabilities"][0]["1"] > body["probabilities"][1]["1"]


def test_predict_missing_columns_is_422_with_details(client):
    r = client.post("/predict", json={"rows": [{"x1": 1.0}]})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "missing" in detail and "x2" in detail["missing"]


def test_predict_handles_unseen_category(client):
    r = client.post("/predict", json={"rows": [{"x1": 0.0, "x2": 0.0, "cat": "zzz",
                                                "flag": False}]})
    assert r.status_code == 200          # unknown levels are handled by the pipeline


def test_drift_flags_shifted_feature(client):
    rs = np.random.RandomState(1)
    rows = [{"x1": float(v + 10), "x2": float(w), "cat": "a", "flag": True}
            for v, w in zip(rs.randn(80), rs.randn(80))]
    r = client.post("/drift", json={"rows": rows})
    assert r.status_code == 200
    body = r.json()
    x1 = next(row for row in body["report"] if row["feature"] == "x1")
    assert x1["status"] == "MAJOR DRIFT"
    assert body["n_features_shifted"] >= 1


def test_boolean_feature_roundtrips_through_recipe(client):
    """Raw JSON booleans must flow through the boolean_as_int recipe step —
    the serving mirror of the recipe once silently omitted it."""
    r = client.post("/predict", json={"rows": [
        {"x1": 0.5, "x2": 0.0, "cat": "b", "flag": True},
        {"x1": 0.5, "x2": 0.0, "cat": "b", "flag": False}]})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2


def test_concurrent_first_requests_are_safe(client):
    """Regression: _load once published a partially-built module dict, so a
    second thread arriving mid-load could 500 with KeyError('meta')."""
    from concurrent.futures import ThreadPoolExecutor

    import app as app_module
    app_module._state.clear()               # force a cold start under contention
    with ThreadPoolExecutor(max_workers=8) as ex:
        codes = list(ex.map(lambda _: client.get("/health").status_code, range(16)))
    assert codes == [200] * 16


def test_predict_scoring_failure_maps_to_400(client):
    """Well-formed JSON whose values break the pipeline (a string in a numeric
    column) must come back as a clean 400 with the reason, not a 500."""
    r = client.post("/predict", json={"rows": [{"x1": "not-a-number", "x2": 0.0,
                                                "cat": "a", "flag": False}]})
    assert r.status_code == 400
    assert "scoring failed" in r.json()["detail"]


def test_drift_without_reference_is_409(client):
    """An artifact that carries no drift reference must yield a clean 409."""
    import app as app_module
    meta = app_module._load()["meta"]
    saved = meta.get("drift_reference")
    meta["drift_reference"] = None
    try:
        r = client.post("/drift", json={"rows": [{"x1": 0.0, "x2": 0.0,
                                                  "cat": "a", "flag": False}]})
        assert r.status_code == 409
        assert "no drift" in r.json()["detail"]
    finally:
        meta["drift_reference"] = saved


def test_health_reports_unready_artifact_as_503(client, tmp_path):
    """A missing/unloadable artifact is 'not ready': /health must answer 503
    with the actionable reason, not a bare 500."""
    import app as app_module
    saved_state, saved_dir = dict(app_module._state), app_module.MODEL_DIR
    app_module._state.clear()
    app_module.MODEL_DIR = tmp_path / "no-artifact-here"
    try:
        r = client.get("/health")
        assert r.status_code == 503
        assert "model.joblib" in r.json()["detail"]
    finally:
        app_module.MODEL_DIR = saved_dir
        app_module._state.clear()
        app_module._state.update(saved_state)
