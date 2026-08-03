"""Hierarchical forecast reconciliation.

Independently-forecast series in a hierarchy are incoherent: the regional
forecasts won't sum to the total. Reconciliation projects the base forecasts
onto the coherent subspace, ŷ~ = S P ŷ, where S maps bottom-level series to
the full hierarchy (Hyndman et al.). Implemented P matrices:

- ``bottom_up``   — keep the leaves, rebuild parents by summation
- ``ols``         — least-squares projection, identity weights
- ``wls_struct``  — weights = number of leaves under each node
- ``mint_shrink`` — MinT with a shrunk covariance of base-forecast errors,
                    estimated from the champion's backtest folds; when a fold
                    is being evaluated its own errors are excluded. The
                    remaining folds can still lie after the scored one (with
                    K=3 folds there is no purely-past alternative) — a
                    covariance-timing caveat, not target leakage.

Reconciliation moves point forecasts; prediction intervals are translated by
the same adjustment (widths unchanged) and labeled as such — honest, if not
fully probabilistic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import ForecastError

METHODS = ("bottom_up", "ols", "wls_struct", "mint_shrink")


def resolve_hierarchy(hierarchy: dict, series_ids: list) -> tuple:
    """Returns (ordered_series, leaves, S) — S is (n_series, n_leaves)."""
    parents = set(hierarchy)
    children = {c for kids in hierarchy.values() for c in kids}
    unknown = (children | parents) - set(series_ids)
    if unknown:
        raise ForecastError(f"Hierarchy references series not in the panel: "
                            f"{sorted(unknown)}")
    leaves = [s for s in series_ids if s not in parents]
    if not leaves:
        raise ForecastError("Hierarchy has no leaf series (every series is a parent).")

    def descend(node) -> list:
        if node not in hierarchy:
            return [node]
        out = []
        for child in hierarchy[node]:
            out.extend(descend(child))
        return out

    ordered = list(series_ids)
    leaf_pos = {s: i for i, s in enumerate(leaves)}
    S = np.zeros((len(ordered), len(leaves)))
    for r, sid in enumerate(ordered):
        for leaf in descend(sid):
            S[r, leaf_pos[leaf]] = 1.0
    return ordered, leaves, S


def _projection(method: str, S: np.ndarray, W: np.ndarray | None = None,
                leaf_rows: list | None = None) -> np.ndarray:
    n, k = S.shape
    if method == "bottom_up":
        # P selects each leaf's own row — identified by name via leaf_rows,
        # never by unit row-sums: a parent with a single child also has a
        # unit row, and matching on row-sums double-counts that branch.
        if leaf_rows is None or len(leaf_rows) != k:
            raise ValueError("bottom_up needs leaf_rows (one row index per leaf column)")
        P = np.zeros((k, n))
        for j, r in enumerate(leaf_rows):
            P[j, int(r)] = 1.0
        return P
    if method == "ols":
        W = np.eye(n)
    elif method == "wls_struct":
        W = np.diag(S.sum(axis=1))
    elif method == "mint_shrink":
        if W is None:
            raise ValueError("mint_shrink needs an error covariance")
    else:
        # an unrecognized name would otherwise reach pinv(None) as a cryptic
        # numpy TypeError — guardrails fail loudly and specifically here
        raise ValueError(f"unknown reconciliation method {method!r} "
                         f"(expected one of {METHODS})")
    Winv = np.linalg.pinv(W)
    return np.linalg.pinv(S.T @ Winv @ S) @ S.T @ Winv


def shrunk_covariance(errors: np.ndarray) -> np.ndarray:
    """Schäfer–Strimmer-style shrinkage of the error covariance toward its
    diagonal. errors: (n_obs, n_series)."""
    n_obs, n = errors.shape
    if n_obs < 3:
        return np.diag(np.var(errors, axis=0) + 1e-8)
    emp = np.cov(errors, rowvar=False)
    target = np.diag(np.diag(emp))
    # shrinkage intensity from the variance of off-diagonal correlations
    x = (errors - errors.mean(axis=0)) / (errors.std(axis=0) + 1e-12)
    r = (x.T @ x) / n_obs
    var_r = np.zeros_like(r)
    for i in range(n):
        for j in range(n):
            if i != j:
                var_r[i, j] = np.var(x[:, i] * x[:, j]) / n_obs
    off = ~np.eye(n, dtype=bool)
    denom = float(np.sum(r[off] ** 2))
    lam = 1.0 if denom == 0 else float(np.clip(np.sum(var_r[off]) / denom, 0.0, 1.0))
    return lam * target + (1 - lam) * emp + 1e-8 * np.eye(n)


def _stack(preds: dict, ordered: list, index) -> np.ndarray:
    return np.vstack([preds[s].reindex(index)["yhat"].to_numpy() for s in ordered])


def reconcile_predictions(preds: dict, ordered: list, S: np.ndarray, method: str,
                          W: np.ndarray | None = None, leaves: list | None = None) -> dict:
    """Apply ŷ~ = S P ŷ; intervals are translated by the point adjustment.

    ``leaves`` (the leaf series ids, as returned by resolve_hierarchy) is
    required for bottom_up so leaf rows are identified by name.
    """
    index = preds[ordered[0]].index
    base = _stack(preds, ordered, index)            # (n_series, h)
    leaf_rows = [ordered.index(s) for s in leaves] if leaves is not None else None
    P = _projection(method, S, W, leaf_rows)
    tilde = S @ P @ base
    out = {}
    for i, sid in enumerate(ordered):
        p = preds[sid].copy()
        delta = tilde[i] - base[i]
        p["yhat"] = tilde[i]
        if "lo" in p and not p["lo"].isna().all():
            p["lo"] = p["lo"].to_numpy() + delta
            p["hi"] = p["hi"].to_numpy() + delta
        out[sid] = p
    return out


def fold_error_matrix(fold_preds: list, panel, ordered: list, horizon: int,
                      exclude_fold: int | None = None) -> np.ndarray:
    """Base-forecast errors stacked over folds and steps: (n_obs, n_series).

    Interpolated actuals (y_was_filled) never enter the error covariance —
    the engine's rule that synthetic values stay out of evaluation applies
    here too. Their slots are imputed with the series' mean error for the
    fold, keeping the matrix rectangular across series.
    """
    rows = []
    for i, (cutoff, preds) in enumerate(fold_preds):
        if exclude_fold is not None and i == exclude_fold:
            continue
        step_errs = []
        for sid in ordered:
            frame = panel.frames[sid]
            window = frame.loc[frame.index > cutoff].head(horizon)
            p = preds[sid].reindex(window.index)["yhat"]
            err = (window["y"] - p).to_numpy(dtype=float)
            if "y_was_filled" in frame.columns:
                filled = window["y_was_filled"].to_numpy() >= 0.5
                if filled.any():
                    clean = err[~filled & ~np.isnan(err)]
                    err[filled] = float(clean.mean()) if len(clean) else 0.0
            step_errs.append(err)
        rows.append(np.vstack(step_errs).T)         # (h, n_series)
    if not rows:
        return np.zeros((0, len(ordered)))
    return np.vstack(rows)


def coherency_gap(preds: dict, hierarchy: dict) -> float:
    """Mean relative |sum(children) - parent| across parents — how incoherent
    the base forecasts are before reconciliation."""
    gaps = []
    for parent, children in hierarchy.items():
        p = preds[parent]["yhat"].to_numpy()
        s = np.sum([preds[c]["yhat"].to_numpy() for c in children], axis=0)
        denom = np.mean(np.abs(p)) + 1e-12
        gaps.append(float(np.mean(np.abs(s - p)) / denom))
    return float(np.mean(gaps))


def evaluate_reconciliation(fc, methods=METHODS) -> pd.DataFrame:
    """Score each method on the champion's stored backtest folds. MinT's error
    covariance for fold i comes from the other folds only."""
    from .backtest import aggregate_scores, score_predictions

    cfg = fc.config
    ordered, leaves, S = resolve_hierarchy(cfg.hierarchy, fc.panel.series_ids)
    fold_preds = fc.fold_predictions[fc.champion_name]
    n_folds = len(fold_preds)

    rows = []
    for method in ("none",) + tuple(methods):
        frames = []
        for i, (cutoff, preds) in enumerate(fold_preds):
            missing = [s for s in ordered if s not in preds]
            if missing:
                continue
            if method == "none":
                rec = preds
            else:
                W = None
                if method == "mint_shrink":
                    errs = fold_error_matrix(fold_preds, fc.panel, ordered,
                                             cfg.horizon,
                                             exclude_fold=i if n_folds > 1 else None)
                    if len(errs) < 3:
                        continue
                    W = shrunk_covariance(errs)
                rec = reconcile_predictions(preds, ordered, S, method, W, leaves=leaves)
            frames.append(score_predictions(rec, fc.panel, cutoff, cfg.horizon,
                                            fc.season_info["primary"],
                                            cfg.interval_level))
        agg = aggregate_scores(frames)
        if agg:
            rows.append({"method": method, **{k: round(v, 4) for k, v in agg.items()
                                              if k in ("MASE", "WAPE", "RMSE")}})
    return pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)
