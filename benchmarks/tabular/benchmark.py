#!/usr/bin/env python3
"""Benchmark the tabular AutoML engine against AutoGluon.

Every system sees the IDENTICAL train/holdout partition: the engine's own
deterministic load/split (seed 42, duplicates removed, stateless prep) defines
the rows, and challengers are fit on exactly those rows. Challenger metrics
are computed here, from raw holdout predictions; the engine's row reports its
own holdout evaluation — the same sklearn metric functions on the same
partition (see the README's methodology note).

Systems:
    baseline          base-rate (training prior) / mean predictor
    xgb-default       XGBoost, default parameters, minimal ordinal-encode prep
    engine            the tabular AutoML engine (same budget as the demo runs)
    autogluon-medium  AutoGluon presets='medium_quality' (its fast default)
    autogluon-best    AutoGluon presets='best_quality', time-budget-matched to
                      the engine's own Adult fit time (--time-limit, default 1300s)

Usage:
    python benchmark.py --dataset adult --system autogluon-best
    python benchmark.py --all                 # every system x every dataset
    python benchmark.py --table               # print README table from results

Results accumulate in results/results.json (one entry per dataset x system).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tabular"))

from automl import AutoML, AutoMLConfig  # noqa: E402

RESULTS = HERE / "results" / "results.json"
SEED = 42

DATASETS = {
    "adult": {
        "task": "classification",
        "target": "income",
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv",
        "read_kwargs": {
            "header": None,
            "names": ["age", "workclass", "fnlwgt", "education", "education_num",
                      "marital_status", "occupation", "relationship", "race", "sex",
                      "capital_gain", "capital_loss", "hours_per_week",
                      "native_country", "income"],
            "na_values": ["?"], "skipinitialspace": True,
        },
    },
    "diamonds": {
        "task": "regression",
        "target": "price",
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
        "read_kwargs": {},
    },
}


def load_raw(name: str) -> pd.DataFrame:
    spec = DATASETS[name]
    cache = HERE / "results" / f"_{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_csv(spec["url"], **spec["read_kwargs"])
    df.to_parquet(cache)
    return df


def get_partition(name: str):
    """The engine's own deterministic partition — the ground truth for all systems."""
    spec = DATASETS[name]
    aml = AutoML(AutoMLConfig(target=spec["target"], task=spec["task"],
                              random_state=SEED))
    aml.load(load_raw(name))
    p = aml.split()
    # the AutoGluon eval_metric mapping below assumes the engine's
    # auto-selected primary metric; fail fast if a new dataset breaks that
    # (e.g. a heavily skewed target flips the engine to MAE-primary)
    expected = "LogLoss" if aml.ta.task == "classification" else "RMSE"
    if aml.ta.primary_metric != expected:
        raise SystemExit(
            f"engine selected primary metric {aml.ta.primary_metric!r} for '{name}' "
            f"but the challenger metric mapping assumes {expected!r} — update "
            "run_autogluon's eval_metric mapping together with this guard")
    return p, aml.ta


def classification_metrics(y_true, proba_pos) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score
    proba_pos = np.clip(np.asarray(proba_pos, dtype=float), 1e-15, 1 - 1e-15)
    return {"LogLoss": float(log_loss(y_true, np.c_[1 - proba_pos, proba_pos],
                                      labels=[0, 1])),
            "ROC_AUC": float(roc_auc_score(y_true, proba_pos))}


def regression_metrics(y_true, pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
            "MAE": float(mean_absolute_error(y_true, pred)),
            "R2": float(r2_score(y_true, pred))}


# ---------------------------------------------------------------- systems ---

def run_baseline(name: str, args) -> dict:
    p, ta = get_partition(name)
    if ta.task == "classification":
        prev = float(p.y_train.mean())
        return {**classification_metrics(p.y_holdout, np.full(len(p.y_holdout), prev)),
                "ROC_AUC": 0.5, "fit_seconds": 0.0}
    mean = float(p.y_train.mean())
    return {**regression_metrics(p.y_holdout, np.full(len(p.y_holdout), mean)),
            "fit_seconds": 0.0}


def run_xgb_default(name: str, args) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    from xgboost import XGBClassifier, XGBRegressor

    p, ta = get_partition(name)
    num = p.X_train.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in p.X_train.columns if c not in num]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                 unknown_value=-1))]), cat)])
    t0 = time.time()
    if ta.task == "classification":
        pipe = Pipeline([("prep", prep),
                         ("model", XGBClassifier(tree_method="hist", eval_metric="logloss",
                                                 random_state=SEED, n_jobs=-1, verbosity=0))])
        pipe.fit(p.X_train, p.y_train)
        fit_s = time.time() - t0
        return {**classification_metrics(p.y_holdout, pipe.predict_proba(p.X_holdout)[:, 1]),
                "fit_seconds": round(fit_s, 1)}
    pipe = Pipeline([("prep", prep),
                     ("model", XGBRegressor(tree_method="hist", random_state=SEED,
                                            n_jobs=-1, verbosity=0))])
    pipe.fit(p.X_train, p.y_train)
    fit_s = time.time() - t0
    return {**regression_metrics(p.y_holdout, pipe.predict(p.X_holdout)),
            "fit_seconds": round(fit_s, 1)}


def run_engine(name: str, args) -> dict:
    """Full engine run with the exact settings the demo notebooks use."""
    spec = DATASETS[name]
    workdir = tempfile.mkdtemp(prefix=f"engine_{name}_")
    cfg = AutoMLConfig(target=spec["target"], task=spec["task"], random_state=SEED,
                       optuna_trials=12, optuna_timeout=240,
                       artifact_dir=f"{workdir}/artifacts")
    t0 = time.time()
    aml = AutoML(cfg).run(load_raw(name))
    fit_s = time.time() - t0
    metrics = aml.holdout_metrics
    out = ({"LogLoss": metrics["LogLoss"], "ROC_AUC": metrics["ROC_AUC"]}
           if spec["task"] == "classification"
           else {"RMSE": metrics["RMSE"], "MAE": metrics["MAE"], "R2": metrics["R2"]})
    out["fit_seconds"] = round(fit_s, 1)
    out["champion"] = aml.champion_name
    shutil.rmtree(workdir, ignore_errors=True)
    return out


def run_autogluon(name: str, args, presets: str, time_limit=None) -> dict:
    from autogluon.tabular import TabularPredictor

    spec = DATASETS[name]
    p, ta = get_partition(name)
    target = spec["target"]
    train = p.X_train.copy()
    train[target] = np.asarray(p.y_train)

    workdir = tempfile.mkdtemp(prefix=f"ag_{name}_")
    # matched to the engine's auto-selected primary metric (get_partition
    # guards this assumption for every system)
    eval_metric = "log_loss" if ta.task == "classification" else "root_mean_squared_error"
    problem_type = ("binary" if (ta.task == "classification" and ta.is_binary)
                    else ("multiclass" if ta.task == "classification" else "regression"))
    t0 = time.time()
    predictor = TabularPredictor(label=target, eval_metric=eval_metric,
                                 problem_type=problem_type, path=workdir,
                                 verbosity=1)
    # AutoGluon 1.5 exposes no fit-level seed; it seeds its own components.
    predictor.fit(train, presets=presets, time_limit=time_limit)
    fit_s = time.time() - t0

    if ta.task == "classification":
        proba = predictor.predict_proba(p.X_holdout)
        out = classification_metrics(p.y_holdout, proba[1].to_numpy()
                                     if 1 in proba.columns else proba.iloc[:, -1].to_numpy())
    else:
        out = regression_metrics(p.y_holdout, predictor.predict(p.X_holdout).to_numpy())
    out["fit_seconds"] = round(fit_s, 1)
    try:
        out["champion"] = str(predictor.model_best)
        out["n_models"] = int(len(predictor.model_names()))
    except Exception:
        pass
    shutil.rmtree(workdir, ignore_errors=True)
    return out


SYSTEMS = {
    "baseline": run_baseline,
    "xgb-default": run_xgb_default,
    "engine": run_engine,
    "autogluon-medium": lambda n, a: run_autogluon(n, a, presets="medium_quality"),
    "autogluon-best": lambda n, a: run_autogluon(n, a, presets="best_quality",
                                                 time_limit=a.time_limit),
}


def save_result(dataset: str, system: str, payload: dict):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    import sklearn
    try:
        import autogluon.tabular as agt
        ag_version = agt.__version__
    except ImportError:
        ag_version = None
    payload["_meta"] = {"seed": SEED, "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "sklearn": sklearn.__version__, "autogluon": ag_version}
    data.setdefault(dataset, {})[system] = payload
    RESULTS.write_text(json.dumps(data, indent=2))


def print_table():
    data = json.loads(RESULTS.read_text())
    for ds, systems in data.items():
        task = DATASETS[ds]["task"]
        cols = (["LogLoss", "ROC_AUC"] if task == "classification"
                else ["RMSE", "MAE", "R2"])
        print(f"\n### {ds} ({task})")
        header = "| System | " + " | ".join(cols) + " | fit time |"
        print(header)
        print("|" + "---|" * (len(cols) + 2))
        for system in ("baseline", "xgb-default", "engine",
                       "autogluon-medium", "autogluon-best"):
            if system not in systems:
                continue
            r = systems[system]
            vals = " | ".join(f"{r[c]:,.4f}" if c != "RMSE" and c != "MAE"
                              else f"{r[c]:,.1f}" for c in cols)
            mins = r["fit_seconds"] / 60
            print(f"| {system} | {vals} | {mins:.1f} min |")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), help="one dataset")
    ap.add_argument("--system", choices=list(SYSTEMS), help="one system")
    ap.add_argument("--all", action="store_true", help="every system x dataset")
    ap.add_argument("--table", action="store_true", help="print markdown table")
    ap.add_argument("--time-limit", type=int, default=1300,
                    help="seconds for autogluon-best (default 1300 ≈ the engine's "
                         "Adult fit time in the committed run)")
    args = ap.parse_args()

    if args.table:
        print_table()
        return

    jobs = []
    datasets = [args.dataset] if args.dataset else list(DATASETS)
    systems = [args.system] if args.system else list(SYSTEMS)
    if not args.all and not (args.dataset or args.system):
        ap.error("pass --all, --table, or --dataset/--system")
    for ds in datasets:
        for s in systems:
            jobs.append((ds, s))

    for ds, system in jobs:
        print(f"\n{'='*70}\nRUN {system} on {ds}\n{'='*70}", flush=True)
        t0 = time.time()
        result = SYSTEMS[system](ds, args)
        save_result(ds, system, result)
        print(f"-> {json.dumps({k: v for k, v in result.items() if k != '_meta'})}"
              f"  ({(time.time()-t0)/60:.1f} min total)", flush=True)


if __name__ == "__main__":
    main()
