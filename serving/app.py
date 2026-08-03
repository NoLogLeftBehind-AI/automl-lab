"""FastAPI scoring service for artifacts exported by the tabular AutoML engine.

Point MODEL_DIR at any exported artifact folder (the directory containing
model.joblib + metadata.json) and run:

    MODEL_DIR=../tabular/notebooks/artifacts_classification \\
        uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   — readiness + model identity (champion, metric, holdout
                     scores); 503 with the reason while no artifact is loadable
    POST /predict  — JSON rows -> predictions (+ class probabilities)
    POST /drift    — JSON rows -> PSI drift report vs the training distribution

The artifact is self-describing: the service reads the feature schema, the
stateless preprocessing recipe, class labels, and the drift reference from
metadata.json, and the tuned decision threshold is already inside the
pipeline — no engine code is imported here.

Trust boundary: model.joblib is a pickle, and unpickling executes code. Only
mount artifacts you trained yourself or otherwise trust.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "artifact"))

app = FastAPI(title="AutoML artifact scoring service",
              description=__doc__, version="0.1.0")

_state: dict = {}
_state_lock = threading.Lock()


def _load() -> dict:
    # Sync endpoints run in FastAPI's threadpool, so concurrent first requests
    # race here: check under the lock and publish the fully-built dict in one
    # update, never key-by-key (a partially populated module dict would leak).
    if "meta" in _state:
        return _state
    with _state_lock:
        if "meta" in _state:
            return _state
        model_path = MODEL_DIR / "model.joblib"
        meta_path = MODEL_DIR / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise RuntimeError(
                f"MODEL_DIR={MODEL_DIR} does not contain model.joblib + metadata.json — "
                "point it at a folder exported by the tabular AutoML engine")
        try:
            bundle = joblib.load(model_path)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"the artifact's champion needs a library that is not installed ({e}). "
                "Install the artifact's own requirements.txt alongside the service "
                '(Docker: build with --build-arg EXTRA_MODELS="xgboost lightgbm ...")'
            ) from e
        except Exception as e:  # corrupt pickle etc. — still a readiness problem
            raise RuntimeError(
                f"model.joblib in MODEL_DIR={MODEL_DIR} failed to load "
                f"({type(e).__name__}: {e})") from e
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            raise RuntimeError(
                f"metadata.json in MODEL_DIR={MODEL_DIR} is unreadable "
                f"({type(e).__name__}: {e})") from e
        if not isinstance(bundle, dict) or "pipeline" not in bundle:
            raise RuntimeError(
                f"model.joblib in MODEL_DIR={MODEL_DIR} has no 'pipeline' entry — "
                "was it exported by the tabular AutoML engine?")
        _state.update({"pipeline": bundle["pipeline"],
                       "label_encoder": bundle.get("label_encoder"),
                       "meta": meta})
    return _state


def _apply_recipe(df: pd.DataFrame, recipe: dict) -> pd.DataFrame:
    """Replay the stateless preparation recorded at training time (mirrors the
    generated predict.py — kept dependency-free here)."""
    df = df.copy()
    if recipe.get("renamed_columns"):
        df = df.rename(columns=recipe["renamed_columns"])
    for col in recipe.get("boolean_as_int", []):
        if col in df.columns and pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("int8")
    for col, derived in recipe.get("date_decompositions", {}).items():
        if col not in df.columns:
            continue
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(df[col], errors="coerce")
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
        parts = {f"{col}__year": parsed.dt.year, f"{col}__month": parsed.dt.month,
                 f"{col}__day": parsed.dt.day, f"{col}__dayofweek": parsed.dt.dayofweek,
                 f"{col}__hour": parsed.dt.hour,
                 f"{col}__is_weekend": (parsed.dt.dayofweek >= 5).astype("float64"),
                 f"{col}__epoch_days": (parsed - pd.Timestamp("1970-01-01")).dt.days}
        for name in derived:
            df[name] = (pd.to_numeric(parts[name], errors="coerce").astype("float64")
                        if name in parts else np.nan)
        df = df.drop(columns=[col])
    for col in recipe.get("text_fillna", []):
        if col in df.columns:
            df[col] = df[col].astype(str).where(df[col].notna(), "")
    for col in recipe.get("categorical_as_string", []):
        if col in df.columns:
            na = df[col].isna()
            df[col] = df[col].astype(str)
            df.loc[na, col] = np.nan
    return df


class Rows(BaseModel):
    rows: list[dict] = Field(..., min_length=1,
                             description="Raw rows with the original column names")


@app.get("/health")
def health() -> dict:
    try:
        s = _load()
    except RuntimeError as e:
        # readiness semantics: a missing/unloadable artifact is "not ready",
        # with the actionable reason in the response, not a bare 500
        raise HTTPException(status_code=503, detail=str(e)) from e
    m = s["meta"]
    return {"status": "ok",
            "task": m["task"],
            "champion": m["champion"],
            "primary_metric": m["primary_metric"],
            "holdout_metrics": m["holdout_metrics"],
            "n_features": len(m["features"]),
            "classes": m["classes"],
            "trained_utc": m["created_utc"],
            "suspected_leakage": m.get("suspected_leakage", False)}


@app.post("/predict")
def predict(payload: Rows) -> dict:
    s = _load()
    m = s["meta"]
    try:
        df = _apply_recipe(pd.DataFrame(payload.rows), m.get("preprocessing_recipe", {}))
    except Exception as e:  # e.g. unparseable dates in a decomposed column
        raise HTTPException(status_code=400,
                            detail=f"invalid rows: {type(e).__name__}: {e}") from e
    missing = [c for c in m["features"] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422,
                            detail={"error": "missing required columns",
                                    "missing": missing,
                                    "expected": m["features"]})
    X = df[m["features"]]
    try:
        pred = s["pipeline"].predict(X)
        proba = s["pipeline"].predict_proba(X) if m["task"] == "classification" else None
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"scoring failed: {type(e).__name__}: {e}") from e

    if m["task"] == "classification":
        assert proba is not None   # always computed for classification above
        le = s["label_encoder"]
        labels = (le.inverse_transform(pred.astype(int)).tolist()
                  if le is not None else pred.tolist())
        return {"n": len(labels),
                "predictions": labels,
                "probabilities": [
                    {str(cls): round(float(p), 6) for cls, p in zip(m["classes"], row)}
                    for row in proba],
                "decision_threshold": m.get("optimal_threshold"),
                "note": ("binary decisions already use the tuned threshold "
                         "baked into the pipeline" if m.get("optimal_threshold")
                         else None)}
    return {"n": len(pred), "predictions": [float(v) for v in pred]}


@app.post("/drift")
def drift(payload: Rows) -> dict:
    s = _load()
    m = s["meta"]
    try:
        df = _apply_recipe(pd.DataFrame(payload.rows), m.get("preprocessing_recipe", {}))
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"invalid rows: {type(e).__name__}: {e}") from e
    reference = m.get("drift_reference")
    if not reference:
        raise HTTPException(status_code=409, detail="this artifact carries no drift "
                                                    "reference — re-export it with one "
                                                    "to enable /drift")

    def psi(p_ref, p_new, eps: float = 1e-4) -> float:
        p_ref = np.clip(np.asarray(p_ref, dtype=float), eps, None)
        p_new = np.clip(np.asarray(p_new, dtype=float), eps, None)
        p_ref, p_new = p_ref / p_ref.sum(), p_new / p_new.sum()
        return float(np.sum((p_new - p_ref) * np.log(p_new / p_ref)))

    out = []
    for col, ref in reference.items():
        if col not in df.columns:
            out.append({"feature": col, "psi": None, "status": "MISSING COLUMN"})
            continue
        if ref["type"] == "skipped":
            out.append({"feature": col, "psi": None, "status": "SKIPPED"})
            continue
        if ref["type"] == "numeric":
            edges = np.asarray(ref["edges"], dtype=float)
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not len(vals):
                out.append({"feature": col, "psi": None, "status": "ALL MISSING"})
                continue
            counts, _ = np.histogram(vals.clip(edges[0], edges[-1]), bins=edges)
            # normalize BEFORE the epsilon clip (same convention as the
            # engine's drift_check.py) so empty bins floor at eps, not eps/N
            score = psi(ref["props"], counts / counts.sum())
        else:
            vc = df[col].fillna("missing").astype(str).value_counts(normalize=True)
            p_ref = list(ref["categories"].values()) + [ref["other"]]
            p_new = [float(vc.get(c, 0.0)) for c in ref["categories"]]
            p_new.append(max(0.0, 1.0 - sum(p_new)))
            score = psi(p_ref, p_new)
        status = ("OK" if score < 0.10 else
                  "MODERATE DRIFT" if score < 0.25 else "MAJOR DRIFT")
        out.append({"feature": col, "psi": round(score, 4), "status": status})
    out.sort(key=lambda r: (r["psi"] is None, -(r["psi"] or 0)))
    n_shifted = sum(1 for r in out if r["status"] in ("MODERATE DRIFT", "MAJOR DRIFT"))
    return {"n_rows": len(df), "n_features_shifted": n_shifted, "report": out}
