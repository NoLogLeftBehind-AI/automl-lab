"""Shared helpers: errors, decision logging, model-thread control."""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd


class AutoMLError(Exception):
    """A guardrail stopped the run — the message says why and what to do."""


# Targeted filters only — the engine deliberately does NOT silence warnings
# wholesale (convergence and deprecation warnings are diagnostics). Each entry
# below is a known-cosmetic message with a reason:
#
# - LightGBM 4.x's sklearn wrapper tracks feature names internally even when the
#   surrounding Pipeline feeds it plain numpy arrays, so sklearn's name check
#   fires on every predict despite fit/predict receiving identical input.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names, but LGBMClassifier")
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names, but LGBMRegressor")


class DecisionLog:
    """Every automated decision lands here, printable and serializable."""

    def __init__(self, verbose: bool = True):
        self.records: list[dict] = []
        self.verbose = verbose

    def log(self, stage: str, message: str):
        self.records.append({"stage": stage, "decision": message})
        if self.verbose:
            print(f"[{stage}] {message}")

    def warn(self, stage: str, message: str):
        self.log(stage, f"WARNING: {message}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=["stage", "decision"])


_FORBIDDEN_NAME_CHARS = re.compile(r'[\[\]<>{}"\':,]')


def sanitize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """XGBoost/LightGBM reject JSON-special characters in feature names.

    Returns the (possibly renamed) frame and a {original: sanitized} mapping that
    is recorded in the artifact metadata so scoring code can reproduce it.
    """
    mapping = {}
    seen = set()
    for col in df.columns:
        new = _FORBIDDEN_NAME_CHARS.sub("_", str(col)).strip()
        if new != col:
            base = new
            i = 1
            while new in seen or new in df.columns:
                new = f"{base}_{i}"
                i += 1
            mapping[col] = new
        seen.add(new)
    if mapping:
        df = df.rename(columns=mapping)
    return df, mapping


def check_duplicate_columns(df: pd.DataFrame):
    dupes = df.columns[df.columns.duplicated()].unique().tolist()
    if dupes:
        raise AutoMLError(
            f"Duplicate column names found: {dupes}. Deduplicate upstream "
            "(e.g. df = df.loc[:, ~df.columns.duplicated()]) — ambiguous columns "
            "make every downstream step unreliable."
        )


def set_model_threads(pipeline, n_threads: int):
    """Pin the estimator's own thread count.

    Model-level parallelism nested inside joblib-parallel cross-validation causes
    catastrophic oversubscription (LightGBM in particular can slow down ~1000x),
    so models run single-threaded during CV and multi-threaded for final fits.
    """
    est = pipeline
    skip = ("LogisticRegression",)  # n_jobs is a deprecated no-op since sklearn 1.8
    # unwrap Pipeline / TransformedTargetRegressor / threshold & calibration
    # wrappers to reach the estimator
    while True:
        if hasattr(est, "steps"):
            est = est.steps[-1][1]
        elif hasattr(est, "regressor"):
            est = est.regressor
        elif hasattr(est, "estimator"):
            est = est.estimator
        else:
            break
    if est.__class__.__name__ in skip:
        return pipeline
    for param in ("n_jobs", "thread_count"):
        if hasattr(est, param):
            try:
                est.set_params(**{param: n_threads})
            except ValueError:
                pass
            return pipeline
    return pipeline


def quantile_bins_for_stratification(y: pd.Series, n_bins: int = 10) -> pd.Series | None:
    """Bin a continuous target so regression splits can stratify on it."""
    try:
        bins = pd.qcut(y, q=min(n_bins, max(2, y.nunique())), duplicates="drop", labels=False)
        if pd.Series(bins).nunique() < 2:
            return None
        # every stratum needs >= 2 members for train_test_split
        counts = pd.Series(bins).value_counts()
        if counts.min() < 2:
            return None
        return bins
    except (ValueError, IndexError):
        return None


def capture_convergence_warnings(decision_log: DecisionLog, stage: str):
    """Context manager: count ConvergenceWarnings instead of silencing everything."""
    from sklearn.exceptions import ConvergenceWarning

    class _Ctx:
        def __enter__(self):
            self._cw = warnings.catch_warnings(record=True)
            self._records = self._cw.__enter__()
            warnings.simplefilter("always", ConvergenceWarning)
            return self

        def __exit__(self, *exc):
            n = sum(1 for w in self._records if issubclass(w.category, ConvergenceWarning))
            self._cw.__exit__(*exc)
            if n:
                decision_log.warn(stage, f"{n} convergence warning(s) during fitting — "
                                         "linear-model scores may understate the family; "
                                         "consider more iterations or stronger regularization")
            return False

    return _Ctx()


def infer_task(y: pd.Series) -> str:
    """Infer classification vs regression from the target when task='auto'."""
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= 2:
        return "classification"
    # small-cardinality integers are usually labels, not quantities
    if nunique <= 20 and np.allclose(y.dropna() % 1, 0):
        return "classification"
    return "regression"
