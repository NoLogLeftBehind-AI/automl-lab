"""Ablation: how much of the Diamonds gap to AutoGluon is the engine's own pruning?

The engine's correlation pruner drops `carat` on Diamonds — it keeps `y` (width
in mm) over `carat` (weight) because `y` scored the stronger single-feature
target association. Both are near-duplicates of each other (|rho| = 0.996), so
exactly one survives; the question is whether the pruner picked the right one.

Two runs of the same engine, same seed, same settings, differing only in
`force_keep_columns`. This is the comparison the benchmark README used to list
as "not computed here".

    python ablation_carat.py

Writes results/ablation_carat.json. ~45 min on 4 cores.
"""
import json
import time
from pathlib import Path

from automl import AutoML, AutoMLConfig

from benchmark import DATASETS, SEED, load_raw

HERE = Path(__file__).parent
OUT = HERE / "results" / "ablation_carat.json"

# AutoGluon best_quality on the same partition, from results.json — the
# reference the gap is measured against.
AUTOGLUON_BEST_RMSE = 516.0116130447523


def run(tag: str, force_keep: list) -> dict:
    spec = DATASETS["diamonds"]
    cfg = AutoMLConfig(target=spec["target"], task=spec["task"], random_state=SEED,
                       optuna_trials=12, optuna_timeout=240,   # benchmark.py settings
                       force_keep_columns=force_keep,
                       artifact_dir=str(HERE / "results" / f"_ablation_{tag}"))
    t0 = time.time()
    aml = AutoML(cfg).run(load_raw("diamonds"))
    return {
        "force_keep_columns": force_keep,
        "RMSE": aml.holdout_metrics["RMSE"],
        "MAE": aml.holdout_metrics["MAE"],
        "R2": aml.holdout_metrics["R2"],
        "champion": aml.champion_name,
        "deployment_pick": aml.deployment_pick,
        "features": list(aml.partition.X_train.columns),
        "correlation_drops": list(aml.corr_drops),
        "fit_seconds": round(time.time() - t0, 1),
    }


def main():
    results = {"default_pruning": run("default", []),
               "carat_protected": run("carat", ["carat"])}
    a, b = results["default_pruning"], results["carat_protected"]
    gap = a["RMSE"] - AUTOGLUON_BEST_RMSE
    results["_summary"] = {
        "autogluon_best_rmse": AUTOGLUON_BEST_RMSE,
        "rmse_cost_of_pruning_carat": a["RMSE"] - b["RMSE"],
        "gap_to_autogluon_default_pct": (a["RMSE"] - AUTOGLUON_BEST_RMSE) / AUTOGLUON_BEST_RMSE * 100,
        "gap_to_autogluon_carat_kept_pct": (b["RMSE"] - AUTOGLUON_BEST_RMSE) / AUTOGLUON_BEST_RMSE * 100,
        "share_of_gap_explained_by_pruning_pct": (a["RMSE"] - b["RMSE"]) / gap * 100,
    }
    OUT.write_text(json.dumps(results, indent=2))
    s = results["_summary"]
    print(f"\n  default pruning (keeps 'y')   RMSE {a['RMSE']:.2f}  "
          f"{s['gap_to_autogluon_default_pct']:+.2f}% vs AutoGluon")
    print(f"  'carat' protected             RMSE {b['RMSE']:.2f}  "
          f"{s['gap_to_autogluon_carat_kept_pct']:+.2f}% vs AutoGluon")
    print(f"  cost of the pruner's choice:  {s['rmse_cost_of_pruning_carat']:.2f} RMSE "
          f"= {s['share_of_gap_explained_by_pruning_pct']:.0f}% of the gap")


if __name__ == "__main__":
    main()
