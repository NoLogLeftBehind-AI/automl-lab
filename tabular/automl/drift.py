"""Population Stability Index drift reference + standalone checker source.

The training distribution of every feature is snapshotted (decile bins for
numerics, level frequencies for categoricals) into the artifact metadata, and a
dependency-light ``drift_check.py`` ships alongside the model so production
batches can be monitored without this package installed.

PSI reading: < 0.10 stable | 0.10–0.25 moderate shift | > 0.25 major shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_drift_reference(X_train: pd.DataFrame) -> dict:
    ref: dict[str, dict] = {}
    for col in X_train.columns:
        s = X_train[col]
        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            edges = (np.unique(np.percentile(clean, np.linspace(0, 100, 11)))
                     if len(clean) else np.array([]))
            if len(edges) < 3:
                ref[col] = {"type": "skipped", "reason": "low variance"}
                continue
            counts, _ = np.histogram(clean, bins=edges)
            ref[col] = {"type": "numeric", "edges": edges.tolist(),
                        "props": (counts / max(counts.sum(), 1)).tolist()}
        else:
            vc = s.fillna("missing").astype(str).value_counts(normalize=True)
            top20 = vc.head(20)
            ref[col] = {"type": "categorical",
                        "categories": {str(k): float(v) for k, v in top20.items()},
                        "other": float(max(0.0, 1.0 - top20.sum()))}
    return ref


DRIFT_CHECK_SRC = '''# Standalone PSI drift checker — ships with the model artifact.
# Usage:
#     from drift_check import check_drift
#     report = check_drift(new_df, "metadata.json")
#     print(report[report["status"] != "OK"])
# PSI: < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 major shift
import json

import numpy as np
import pandas as pd


def _psi(p_ref, p_new, eps=1e-4):
    p_ref = np.clip(np.asarray(p_ref, dtype=float), eps, None)
    p_new = np.clip(np.asarray(p_new, dtype=float), eps, None)
    p_ref, p_new = p_ref / p_ref.sum(), p_new / p_new.sum()
    return float(np.sum((p_new - p_ref) * np.log(p_new / p_ref)))


def check_drift(new_df, metadata="metadata.json"):
    """metadata: path to metadata.json, or an already-loaded dict."""
    meta = metadata if isinstance(metadata, dict) else json.load(open(metadata))
    rows = []
    for col, ref in meta["drift_reference"].items():
        if col not in new_df.columns:
            rows.append({"feature": col, "psi": np.nan, "status": "MISSING COLUMN"})
            continue
        if ref["type"] == "skipped":
            rows.append({"feature": col, "psi": np.nan, "status": "SKIPPED (low variance)"})
            continue
        if ref["type"] == "numeric":
            edges = np.asarray(ref["edges"], dtype=float)
            vals = pd.to_numeric(new_df[col], errors="coerce").dropna()
            if not len(vals):
                rows.append({"feature": col, "psi": np.nan, "status": "ALL MISSING"})
                continue
            counts, _ = np.histogram(vals.clip(edges[0], edges[-1]), bins=edges)
            # normalize BEFORE the epsilon clip so an empty bin floors at eps
            # (matching the categorical path), not at eps/N after renormalizing
            psi = _psi(ref["props"], counts / counts.sum())
        else:
            vc = new_df[col].fillna("missing").astype(str).value_counts(normalize=True)
            p_ref = list(ref["categories"].values()) + [ref["other"]]
            p_new = [float(vc.get(c, 0.0)) for c in ref["categories"]]
            p_new.append(max(0.0, 1.0 - sum(p_new)))
            psi = _psi(p_ref, p_new)
        status = "OK" if psi < 0.10 else ("MODERATE DRIFT" if psi < 0.25 else "MAJOR DRIFT")
        rows.append({"feature": col, "psi": round(psi, 4), "status": status})
    return (pd.DataFrame(rows)
              .sort_values("psi", ascending=False, na_position="first")
              .reset_index(drop=True))
'''


def check_drift(new_df: pd.DataFrame, metadata) -> pd.DataFrame:
    """In-process convenience wrapper around the shipped checker."""
    ns: dict = {}
    exec(DRIFT_CHECK_SRC, ns)  # single source of truth: run the exact shipped code
    return ns["check_drift"](new_df, metadata)
