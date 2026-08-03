"""The AutoML orchestrator.

Notebook-friendly staged API (each stage prints/returns something inspectable):

    aml = AutoML(AutoMLConfig(target="income", task="classification"))
    aml.load(df)          # profile + quality guardrails + stateless prep
    aml.split()           # locked holdout + CV strategy + metric selection
    aml.scan_leakage()    # two-detector leakage scan
    aml.prune_correlation()
    aml.screen()          # stage 1: every family, identical folds
    aml.tune()            # stage 2: Optuna with default-baseline guarantee
    aml.rescore()         # blenders join; de-biased final leaderboard on fresh folds
    aml.evaluate()        # calibration, operating point, locked-holdout metrics
    aml.explain()         # permutation importance, PDP, SHAP
    aml.export()          # model.joblib + metadata + predict.py + report.html

or simply ``aml.run(df)`` for the full sequence.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate, train_test_split

from .artifacts import build_metadata, export_artifacts
from .config import AutoMLConfig
from .drift import build_drift_reference, check_drift
from .ensemble import build_blenders
from .evaluate import (bootstrap_ci, evaluate_classification, evaluate_regression,
                       fig_calibration, fig_confusion_roc, fig_lift,
                       fig_regression_diagnostics, finalize_classifier)
from .explain import (compute_permutation_importance, fig_importance,
                      fig_partial_dependence, shap_explanations)
from .leakage import prune_correlated, scan_leakage
from .models import build_roster, fit_with_early_stopping, supports_early_stopping
from .partition import fresh_cv, make_partition
from .preprocess import FeatureSpec
from .profiling import load_dataframe, profile_and_prepare
from .report import build_report
from .search import _disp, tune_finalist
from .target import analyze_target_distribution, encode_target
from .utils import (AutoMLError, DecisionLog, capture_convergence_warnings,
                    quantile_bins_for_stratification, set_model_threads)


def one_se_band(best: dict, cand: dict) -> float:
    """Standard error of the (best - candidate) score difference.

    This is the actual one-standard-error rule of CART pruning and glmnet's
    ``lambda.1se``: the band is a standard *error*, sd/sqrt(k), not the
    marginal fold-to-fold standard *deviation* — which is ~sqrt(k) times
    wider and would wave through margins the folds can distinguish.

    Every candidate is scored on the same fixed splitter, so the fold scores
    are paired and the right scale is the spread of their per-fold
    *differences*: fold difficulty is common to both candidates and cancels,
    typically making this band much tighter than either model's own spread.
    Falls back to the best model's sd/sqrt(k) when per-fold arrays are
    unavailable (hand-built score dicts), and to the marginal std only when
    k is unknown too.
    """
    bf, cf = best.get("folds"), cand.get("folds")
    if bf and cf and len(bf) == len(cf) > 1:
        d = np.asarray(bf, dtype=float) - np.asarray(cf, dtype=float)
        d = d[~np.isnan(d)]
        if len(d) > 1:
            return float(np.std(d, ddof=1) / np.sqrt(len(d)))
    k = len(bf) if bf else 0
    return float(best["std"] / np.sqrt(k)) if k > 1 else float(best["std"])


def parsimony_pick(final_scores: dict) -> str:
    """The deployment pick: the simplest candidate whose final score sits
    within one standard error of the best score.

    The classic one-standard-error rule (CART pruning, glmnet's
    ``lambda.1se``): a margin the folds cannot resolve does not buy the extra
    deployment surface a blend carries. Complexity order, simplest first:
    single model, soft-voting blend, stacking blend; ties break toward the
    better score. ``final_scores`` maps candidate name -> {'mean', 'std',
    'folds'} in sklearn's higher-is-better convention, where 'folds' is the
    per-fold score array that makes the paired comparison possible.
    """
    best_name = max(final_scores, key=lambda n: final_scores[n]["mean"])
    best = final_scores[best_name]

    def complexity(name: str) -> int:
        if name.startswith("Stacked"):
            return 2
        if name.startswith("Voting"):
            return 1
        return 0

    within = [n for n, s in final_scores.items()
              if best["mean"] - s["mean"] <= one_se_band(best, s)]
    return min(within, key=lambda n: (complexity(n), -final_scores[n]["mean"]))


def _screen_sample_indices(y: pd.Series, task: str, cap: int, seed: int) -> np.ndarray:
    """Order-preserving stratified subsample for stage-1 screening.

    The returned positions are sorted so row order survives the subsample —
    essential for order-dependent splitters (TimeSeriesSplit), harmless for
    shuffled ones.
    """
    strat = y if task == "classification" else quantile_bins_for_stratification(y)
    idx = np.arange(len(y))
    keep_idx, _ = train_test_split(idx, train_size=cap, random_state=seed, stratify=strat)
    keep_idx.sort()
    return keep_idx


class AutoML:
    def __init__(self, config: AutoMLConfig):
        # private copy: column sanitization may rewrite config.target, and the
        # caller's config object must stay reusable across runs
        self.config = replace(config)
        self.log = DecisionLog()
        self._t0 = time.time()
        self.figures: dict = {}
        self.tuned: dict = {}
        self.candidates: dict = {}
        self.suspected_leakage = False
        self.threshold = None
        self.calibration_info = {"applied": False}
        self.best_trees: dict[str, int] = {}

    # ------------------------------------------------------------------ load
    def load(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        if df is None:
            df = load_dataframe(self.config)
        # a raw slice for the export smoke test, so the generated scorer is
        # exercised through the full preparation recipe, not on prepared rows
        self.raw_sample = df.head(50).copy()
        self.X_all, y_raw, prof = profile_and_prepare(df.copy(), self.config, self.log)
        self.profile_result = prof
        self.y_all, self.ta, keep_mask = encode_target(y_raw, self.config, self.log)
        if not keep_mask.all():
            self.X_all = self.X_all.loc[keep_mask]
        print(f"Rows: {len(self.X_all):,}   Features: {self.X_all.shape[1]}   "
              f"Task: {self.ta.task}")
        return prof.profile

    # ----------------------------------------------------------------- split
    def split(self):
        self.partition = make_partition(self.X_all, self.y_all, self.ta,
                                        self.config, self.log)
        p = self.partition
        # distribution-driven choices use the training partition only
        self.ta = analyze_target_distribution(p.y_train, self.ta, self.config, self.log)
        return p

    # --------------------------------------------------------------- leakage
    def scan_leakage(self) -> pd.DataFrame:
        p = self.partition
        report, leaky, assoc = scan_leakage(p.X_train, p.y_train, self.ta, self.config,
                                            self.log, self.profile_result.text_columns)
        self.leak_report, self.leaky, self.assoc = report, leaky, assoc
        if leaky:
            p.X_train = p.X_train.drop(columns=leaky)
            p.X_holdout = p.X_holdout.drop(columns=leaky)
        return report

    def prune_correlation(self) -> list:
        p = self.partition
        self.corr_drops = prune_correlated(p.X_train, self.assoc, self.config, self.log)
        if self.corr_drops:
            p.X_train = p.X_train.drop(columns=self.corr_drops)
            p.X_holdout = p.X_holdout.drop(columns=self.corr_drops)
        if p.X_train.shape[1] == 0:
            raise AutoMLError("Leakage scan + correlation pruning removed every feature. "
                              "Inspect aml.leak_report — if a flagged feature is genuinely "
                              "available at prediction time, add it to force_keep_columns.")
        self.spec = FeatureSpec(p.X_train, self.profile_result.text_columns,
                                self.config.high_cardinality_thresh)
        self.log.log("features", f"Final feature set: {p.X_train.shape[1]} features "
                                 f"({self.spec.describe()})")
        return self.corr_drops

    # ------------------------------------------------------------ leaderboard
    def screen(self) -> pd.DataFrame:
        """Stage 1: cross-validate every roster model under identical folds."""
        cfg, p, ta = self.config, self.partition, self.ta
        self.roster = build_roster(self.spec, ta, cfg, self.log, len(p.X_train))

        X_s1, y_s1, groups_s1 = p.X_train, p.y_train, p.groups_train
        if len(p.X_train) > cfg.stage1_sample_cap:
            keep_idx = _screen_sample_indices(p.y_train, ta.task,
                                              cfg.stage1_sample_cap, cfg.random_state)
            X_s1, y_s1 = p.X_train.iloc[keep_idx], p.y_train.iloc[keep_idx]
            groups_s1 = p.groups_train[keep_idx] if p.groups_train is not None else None
            self.log.log("leaderboard", f"Stage-1 screening on a {cfg.stage1_sample_cap:,}-row "
                                        f"sample of {len(p.X_train):,} train rows "
                                        "(finalists are re-scored on the full partition later, "
                                        "so mixed-basis scores never decide the champion)")
        scoring = {"primary": ta.primary_scorer, **ta.secondary_scoring}
        rows, failures = [], {}
        for name, pipe in self.roster.items():
            t0 = time.time()
            with capture_convergence_warnings(self.log, f"screen:{name}"):
                try:
                    res = cross_validate(pipe, X_s1, y_s1, cv=p.cv, scoring=scoring,
                                         n_jobs=cfg.n_jobs, groups=groups_s1,
                                         error_score=np.nan)
                except Exception as e:
                    failures[name] = f"{type(e).__name__}: {e}"
                    continue
            vals = res["test_primary"]
            n_bad = int(np.isnan(vals).sum())
            if n_bad == len(vals):
                failures[name] = "all CV folds failed (see warnings above)"
                continue
            if n_bad:
                self.log.warn("leaderboard", f"{name}: {n_bad}/{len(vals)} folds failed; "
                                             "score uses the remaining folds")
            row = {"model": name,
                   ta.primary_metric: _disp(float(np.nanmean(vals)), ta.primary_scorer),
                   f"{ta.primary_metric}_std": float(np.nanstd(vals)),
                   "fit_time_s": round(float(np.nanmean(res["fit_time"])), 1),
                   "_rank": float(np.nanmean(vals))}
            for sec, scorer in ta.secondary_scoring.items():
                row[sec] = _disp(float(np.nanmean(res[f"test_{sec}"])), scorer)
            rows.append(row)
            print(f"  {name:<28} {ta.primary_metric}={row[ta.primary_metric]:.4f}"
                  f"  ({time.time()-t0:.1f}s)")

        for name, err in failures.items():
            self.log.warn("leaderboard", f"{name} failed and is off the leaderboard: {err}")
        if not rows:
            raise AutoMLError("Every roster model failed stage-1 screening — see the "
                              "decision log for per-model errors.")
        self.leaderboard_s1 = (pd.DataFrame(rows).sort_values("_rank", ascending=False)
                               .reset_index(drop=True))
        self.leaderboard_s1.index += 1
        return self.leaderboard_s1.drop(columns=["_rank"])

    def tune(self) -> dict:
        """Stage 2: Optuna on the finalists (full training partition), with the
        default configuration always kept as the baseline candidate."""
        cfg, p, ta = self.config, self.partition, self.ta
        finalists = self.leaderboard_s1["model"].head(cfg.n_finalists).tolist()
        print(f"Finalists advancing to tuning: {finalists}")
        self.tuned = {}
        for name in finalists:
            self.tuned[name] = tune_finalist(name, self.roster[name], p.X_train, p.y_train,
                                             p.cv, ta, cfg, self.log, p.groups_train)
        return self.tuned

    def _resolve_candidate(self, name: str):
        """Unfitted pipeline with tuned params and (for GBMs) a frozen tree count."""
        pipe = clone(self.roster[name]).set_params(**self.tuned.get(name, {}).get("params", {}))
        if self.config.early_stopping and supports_early_stopping(pipe):
            p = self.partition
            fitted, best_n = fit_with_early_stopping(pipe, p.X_train, p.y_train,
                                                     self.ta.task, self.config.random_state,
                                                     chronological=(p.strategy == "time"),
                                                     log=self.log)
            if best_n:
                est = fitted.steps[-1][1]
                prefix = "model__"
                if est.__class__.__name__ == "TransformedTargetRegressor":
                    est, prefix = est.regressor, "model__regressor__"
                param = ("iterations" if est.__class__.__name__.startswith("CatBoost")
                         else "n_estimators")
                frozen = est.get_params()[param]
                pipe.set_params(**{f"{prefix}{param}": frozen})
                if "early_stopping_rounds" in est.get_params():
                    pipe.set_params(**{f"{prefix}early_stopping_rounds": None})
                self.best_trees[name] = int(frozen)
                self.log.log("tuning", f"{name}: early stopping froze the tree count at "
                                       f"{frozen} (validation-chosen {best_n}, +10% headroom "
                                       "for the full-data fit)")
        return pipe

    def rescore(self) -> pd.DataFrame:
        """De-biased final leaderboard: finalists (tuned) and blenders re-scored with a
        single plain CV on *fresh folds* — no winner's-curse best-of-N, no mixed
        sample-vs-full bases. The champion is picked from this column only.
        (Time-aware runs have no fresh-fold variant; their re-score reuses the
        chronological folds and is labeled accordingly.)"""
        cfg, p, ta = self.config, self.partition, self.ta
        self.candidates = {name: self._resolve_candidate(name) for name in self.tuned}
        if cfg.ensemble:
            blenders = build_blenders(self.candidates, ta, cfg)
            self.candidates.update(blenders)
            if blenders:
                self.log.log("ensemble", f"Blender candidates join the final re-score: "
                                         f"{list(blenders)}")

        cv2 = fresh_cv(ta, cfg, cfg.final_cv_seed_offset)
        fold_basis = "reused chronological folds" if cfg.time_column else "fresh-fold CV"
        if cfg.time_column:
            self.log.log("leaderboard", "Time-aware runs have no reshuffled-fold variant "
                                        "(TimeSeriesSplit is chronological), so this re-score "
                                        "reuses the screening/tuning folds. It removes the "
                                        "mixed sample-vs-full basis, but NOT the tuning "
                                        "search's selection bias — the winners are re-scored "
                                        "on the folds that picked them. Treat near-ties on "
                                        "this leaderboard with care.")
        # dict-of-dict with mixed value types (float mean/std, list of folds) —
        # annotated so the heterogeneous inner dict doesn't widen to `object`
        final_scores: dict[str, dict] = {}
        for name, pipe in self.candidates.items():
            t0 = time.time()
            try:
                res = cross_validate(pipe, p.X_train, p.y_train, cv=cv2,
                                     scoring=ta.primary_scorer, n_jobs=cfg.n_jobs,
                                     groups=p.groups_train, error_score=np.nan)
                vals = res["test_score"]
                if np.all(np.isnan(vals)):
                    raise RuntimeError("all folds failed")
                # per-fold scores are retained so parsimony_pick can compare
                # candidates pairwise on the folds they share
                final_scores[name] = {"mean": float(np.nanmean(vals)),
                                      "std": float(np.nanstd(vals)),
                                      "folds": [float(v) for v in vals]}
                print(f"  {name:<28} final {ta.primary_metric}="
                      f"{_disp(final_scores[name]['mean'], ta.primary_scorer):.4f}"
                      f"  ({time.time()-t0:.1f}s)")
            except Exception as e:
                self.log.warn("leaderboard", f"{name} failed the final re-score and is "
                                             f"out of contention: {type(e).__name__}: {e}")
        if not final_scores:
            raise AutoMLError("No candidate survived the final re-score.")

        rows = []
        for _, r in self.leaderboard_s1.iterrows():
            name = r["model"]
            row = {"model": name,
                   f"screen_{ta.primary_metric}": r[ta.primary_metric],
                   f"final_{ta.primary_metric}": np.nan, "final_std": np.nan,
                   "source": "stage-1 screen"}
            if name in final_scores:
                row[f"final_{ta.primary_metric}"] = _disp(final_scores[name]["mean"],
                                                          ta.primary_scorer)
                row["final_std"] = round(final_scores[name]["std"], 4)
                row["source"] = ("tuned" if self.tuned.get(name, {}).get("improved")
                                 else "defaults") + f" · {fold_basis}"
            rows.append(row)
        for name in self.candidates:
            if name in self.leaderboard_s1["model"].values or name not in final_scores:
                continue
            rows.append({"model": name,
                         f"screen_{ta.primary_metric}": np.nan,
                         f"final_{ta.primary_metric}": _disp(final_scores[name]["mean"],
                                                             ta.primary_scorer),
                         "final_std": round(final_scores[name]["std"], 4),
                         "source": f"blender · {fold_basis}"})

        best_name = max(final_scores, key=lambda k: final_scores[k]["mean"])
        pick = parsimony_pick(final_scores)
        self.deployment_pick = pick
        if pick != best_name:
            self.log.log("leaderboard",
                         f"Deployment pick: {pick} "
                         f"({_disp(final_scores[pick]['mean'], ta.primary_scorer):.4f}) — "
                         "within one standard error of the paired fold differences "
                         f"({one_se_band(final_scores[best_name], final_scores[pick]):.4f}) of "
                         f"{best_name} "
                         f"({_disp(final_scores[best_name]['mean'], ta.primary_scorer):.4f}), "
                         "and a simpler model is cheaper to deploy, retrain, and explain. "
                         + ("champion_policy='parsimonious' -> the pick ships as champion"
                            if cfg.champion_policy == "parsimonious" else
                            "The leaderboard stays accuracy-ranked; set "
                            "champion_policy='parsimonious' to ship the pick instead"))
        else:
            self.log.log("leaderboard",
                         "Deployment pick: the leaderboard winner itself — no simpler "
                         "candidate sits within one standard error of its score")
        champion_name = pick if cfg.champion_policy == "parsimonious" else best_name
        self.champion_name = champion_name
        self.champion_cv_score = _disp(final_scores[champion_name]["mean"], ta.primary_scorer)
        self.champion_pipe = clone(self.candidates[champion_name])
        self.final_scores = final_scores

        lb = pd.DataFrame(rows)
        lb["_rank"] = lb[f"final_{ta.primary_metric}"].where(
            lb[f"final_{ta.primary_metric}"].notna(), lb[f"screen_{ta.primary_metric}"])
        ascending = ta.primary_scorer.startswith("neg_")  # display metric: lower is better
        lb = (lb.sort_values(["_rank"], ascending=ascending, na_position="last")
              .drop(columns=["_rank"]).reset_index(drop=True))
        lb.index += 1
        self.leaderboard = lb
        self.log.log("leaderboard", f"Champion: {champion_name} "
                                    f"({fold_basis} {ta.primary_metric}="
                                    f"{self.champion_cv_score:.4f})")
        return lb

    # ------------------------------------------------------------- evaluation
    def evaluate(self) -> dict:
        cfg, p, ta = self.config, self.partition, self.ta

        if ta.task == "classification":
            final_est, self.threshold, self.calibration_info, _ = finalize_classifier(
                self.champion_pipe, p.X_train, p.y_train, p.cv, ta, cfg, self.log,
                p.groups_train)
        else:
            final_est = clone(self.champion_pipe)

        set_model_threads(final_est, cfg.n_jobs)
        final_est.fit(p.X_train, p.y_train)
        self.final_model = final_est

        if ta.task == "classification":
            metrics, proba, pred, cls_report = evaluate_classification(
                final_est, p.X_holdout, p.y_holdout, ta, cfg, self.log)
            self.classification_report_text = cls_report
            print(f"Champion: {self.champion_name}")
            for k, v in metrics.items():
                print(f"  holdout {k:<18}: {v:.4f}")
            print()
            print(cls_report)
            self.figures["Confusion & ROC"] = fig_confusion_roc(
                np.asarray(p.y_holdout), pred, proba, ta, metrics)
            if ta.is_binary:
                self.figures["Lift chart"] = fig_lift(p.y_holdout, proba[:, 1], ta.task)
                self.figures["Calibration"] = fig_calibration(p.y_holdout, proba[:, 1],
                                                              self.threshold, cfg)
            self.holdout_proba, self.holdout_pred = proba, pred
            near_perfect = metrics.get("ROC_AUC", 0) >= 0.995
        else:
            metrics, pred = evaluate_regression(final_est, p.X_holdout, p.y_holdout)
            self.classification_report_text = None
            print(f"Champion: {self.champion_name}")
            for k, v in metrics.items():
                print(f"  holdout {k:<10}: {v:.4f}")
            self.figures["Holdout diagnostics"] = fig_regression_diagnostics(p.y_holdout, pred)
            self.figures["Lift chart"] = fig_lift(p.y_holdout, pred, ta.task)
            self.holdout_proba, self.holdout_pred = None, pred
            near_perfect = metrics.get("R2", 0) >= 0.99

        self.holdout_metrics = metrics
        if near_perfect:
            self.suspected_leakage = True
            self.log.warn("leakage", "Holdout performance is near-perfect. Real problems "
                                     "rarely score this high — verify the top-importance "
                                     "features are genuinely available at prediction time. "
                                     "The artifact metadata carries suspected_leakage=true.")
        self.ci = bootstrap_ci(p.y_holdout, self.holdout_pred, self.holdout_proba,
                               ta, cfg, metrics, self.log)
        return metrics

    def explain(self) -> pd.DataFrame:
        cfg, p, ta = self.config, self.partition, self.ta
        self.importance = compute_permutation_importance(self.final_model, p.X_holdout,
                                                         p.y_holdout, ta, cfg)
        self.figures["Permutation importance"] = fig_importance(self.importance,
                                                                self.champion_name)
        pdp = fig_partial_dependence(self.final_model, p.X_holdout, self.importance,
                                     self.spec.num_cols, ta, cfg, self.champion_name,
                                     log=self.log)
        if pdp is not None:
            self.figures["Partial dependence"] = pdp
        shap_fig, self.shap_example = shap_explanations(self.final_model, p.X_holdout,
                                                        ta, cfg, self.log)
        if shap_fig is not None:
            self.figures["SHAP — mean |contribution|"] = shap_fig
        if self.suspected_leakage:
            suspects = self.importance.head(5)["feature"].tolist()
            self.log.warn("leakage", f"Top-importance features to verify: {suspects}")
        return self.importance

    # ----------------------------------------------------------------- export
    def export(self):
        cfg, p, ta = self.config, self.partition, self.ta
        drift_reference = build_drift_reference(p.X_train)

        export_model = self.final_model
        if cfg.refit_on_full_data:
            export_model = clone(self.final_model)
            X_full = pd.concat([p.X_train, p.X_holdout])
            y_full = pd.concat([p.y_train, p.y_holdout])
            export_model.fit(X_full, y_full)
            self.log.log("export", f"Exported model refit on 100% of rows ({len(X_full):,}); "
                                   "holdout metrics in the metadata describe the pre-refit "
                                   "model, which never saw the holdout")

        metadata = build_metadata(
            task=ta.task, champion_name=self.champion_name, target=cfg.target,
            features=list(p.X_train.columns), spec=self.spec, ta=ta, config=cfg,
            final_score_basis=(
                ("final scores: single CV pass on the full training partition, reusing "
                 "the chronological folds (time-aware runs have no fresh-fold variant) — "
                 "comparable across all candidates and free of the mixed sample-vs-full "
                 "basis, but fold reuse means the tuning search's selection bias is not "
                 "fully removed")
                if cfg.time_column else
                ("final scores: single fresh-fold CV on the full training "
                 "partition — comparable across all candidates and free of "
                 "the tuning search's selection bias")),
            cv_score=float(self.champion_cv_score), holdout_metrics=self.holdout_metrics,
            threshold=self.threshold, calibration_info=self.calibration_info,
            drift_reference=drift_reference,
            tuned_params=self.tuned.get(self.champion_name, {}).get("params", {}),
            dropped={"quality": self.profile_result.dropped, "leakage": self.leaky,
                     "correlation": self.corr_drops},
            deployment_recommendation={
                "model": self.deployment_pick,
                "policy": cfg.champion_policy,
                "applied": cfg.champion_policy == "parsimonious",
                "rule": ("simplest candidate within one standard error of the best "
                         "final score (single model < voting < stacking)")},
            recipe=self.profile_result.recipe, decision_log=self.log.records,
            suspected_leakage=self.suspected_leakage,
            feature_dtypes={c: str(p.X_train[c].dtype) for c in p.X_train.columns},
            best_trees=self.best_trees)

        art_dir = export_artifacts(export_model, ta.label_encoder, metadata, cfg,
                                   self.log, p.X_holdout,
                                   raw_check=getattr(self, "raw_sample", None))
        runtime = time.time() - self._t0
        report_path = build_report(
            art_dir / "report.html",
            title=f"AutoML model report — {cfg.target} ({ta.task})",
            meta=metadata, leaderboard=self.leaderboard,
            holdout_metrics=self.holdout_metrics, ci=self.ci,
            importance=getattr(self, "importance", None), figures=self.figures,
            decision_log=self.log.to_frame(), profile=self.profile_result.profile,
            classification_report_text=self.classification_report_text,
            runtime_s=runtime)
        self.log.log("export", f"Model report -> {report_path}")
        self.log.log("export", f"Total runtime: {runtime/60:.1f} min")
        self.metadata = metadata
        import matplotlib.pyplot as plt
        for fig in self.figures.values():   # everything now lives in report.html
            if fig is not None:
                plt.close(fig)
        return art_dir

    def drift_check(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """PSI report for a new batch against the training distribution."""
        return check_drift(new_df, {"drift_reference": build_drift_reference(
            self.partition.X_train)})

    # -------------------------------------------------------------------- run
    def run(self, df: pd.DataFrame | None = None):
        """The whole workflow in one call. Returns self for inspection."""
        self.load(df)
        self.split()
        self.scan_leakage()
        self.prune_correlation()
        self.screen()
        self.tune()
        self.rescore()
        self.evaluate()
        self.explain()
        self.export()
        return self
