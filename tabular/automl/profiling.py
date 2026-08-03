"""Data loading, quality guardrails, and stateless feature preparation.

Everything here is either a *drop* (with the reason logged) or a *stateless,
row-wise transform* (date decomposition, text NaN-fill, dtype normalization)
recorded as a recipe in the artifact metadata, so the generated scoring script
can reproduce it exactly at inference time. Nothing in this module learns from
data — learned preprocessing lives inside the serialized sklearn pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import AutoMLConfig
from .utils import AutoMLError, DecisionLog, check_duplicate_columns, sanitize_column_names


def load_dataframe(config: AutoMLConfig) -> pd.DataFrame:
    if config.data_path is None:
        raise AutoMLError("No data: set config.data_path or pass a DataFrame to AutoML.load().")
    path = Path(config.data_path)
    if not path.exists():
        raise AutoMLError(f"Data file not found: {path}. Set config.data_path to a .csv or .parquet file.")
    if path.suffix.lower() in (".parquet", ".pq"):
        try:
            return pd.read_parquet(path)
        except ImportError as e:
            raise AutoMLError(
                "Reading parquet requires a parquet engine: pip install pyarrow"
            ) from e
    return pd.read_csv(path)


def _sniff_datetime(sample: pd.Series) -> bool:
    """Datetimes read from CSV arrive as strings — sniff them."""
    if sample.str.contains(r"[-/:]", regex=True).mean() <= 0.9:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
        except (TypeError, ValueError):
            return False
    return bool(parsed.notna().mean() > 0.95)


def decompose_datetime(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, list]:
    """Expand a datetime column into model-ready numeric parts, in place."""
    parsed = df[col]
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(df[col], errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        # tz-aware columns: normalize to naive UTC so epoch arithmetic works
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    derived = []
    parts = {
        f"{col}__year": parsed.dt.year,
        f"{col}__month": parsed.dt.month,
        f"{col}__day": parsed.dt.day,
        f"{col}__dayofweek": parsed.dt.dayofweek,
        f"{col}__hour": parsed.dt.hour,
        f"{col}__is_weekend": (parsed.dt.dayofweek >= 5).astype("float64"),
        f"{col}__epoch_days": (parsed - pd.Timestamp("1970-01-01")).dt.days.astype("float64"),
    }
    for name, values in parts.items():
        vals = pd.to_numeric(values, errors="coerce")
        if vals.nunique(dropna=True) > 1:  # skip parts that carry no signal (e.g. hour in daily data)
            df[name] = vals.astype("float64")
            derived.append(name)
    return df.drop(columns=[col]), derived


class ProfileResult:
    def __init__(self):
        self.profile: pd.DataFrame = None
        self.dropped: dict[str, str] = {}          # column -> reason
        self.text_columns: list[str] = []
        self.recipe: dict = {"renamed_columns": {}, "date_decompositions": {},
                             "text_fillna": [], "categorical_as_string": [],
                             "boolean_as_int": []}


def profile_and_prepare(df: pd.DataFrame, config: AutoMLConfig,
                        log: DecisionLog) -> tuple[pd.DataFrame, pd.Series, ProfileResult]:
    """Apply quality guardrails and stateless preparation. Returns (X, y_raw, result)."""
    result = ProfileResult()
    check_duplicate_columns(df)

    if config.target not in df.columns:
        raise AutoMLError(f"Target column '{config.target}' not found. "
                          f"Available: {list(df.columns)}")

    user_drops = [c for c in config.drop_columns if c in df.columns]
    if user_drops:
        df = df.drop(columns=user_drops)
        log.log("ingest", f"Dropped user-specified columns: {user_drops}")

    n_missing_target = int(df[config.target].isna().sum())
    if n_missing_target:
        df = df.dropna(subset=[config.target])
        log.log("ingest", f"Dropped {n_missing_target} rows with missing target")
    if not len(df):
        raise AutoMLError("No rows left after dropping missing targets.")

    if config.drop_duplicate_rows:
        n_dupes = int(df.duplicated().sum())
        if n_dupes:
            df = df.drop_duplicates()
            log.log("ingest", f"Dropped {n_dupes} duplicate rows — duplicates straddling the "
                              "train/holdout boundary would inflate every score")
    else:
        n_dupes = int(df.duplicated().sum())
        if n_dupes:
            log.warn("ingest", f"{n_dupes} duplicate rows present (drop_duplicate_rows=False) — "
                               "CV and holdout scores may be optimistic")

    df, renamed = sanitize_column_names(df)
    if renamed:
        result.recipe["renamed_columns"] = renamed
        log.log("ingest", f"Sanitized {len(renamed)} column name(s) containing characters "
                          f"XGBoost/LightGBM reject: {renamed}")
        if config.target in renamed:
            config.target = renamed[config.target]

    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    bool_cols = [c for c in bool_cols if c != config.target]
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype("int8")
        result.recipe["boolean_as_int"] = bool_cols
        log.log("ingest", f"Cast boolean columns to int8: {bool_cols}")

    X = df.drop(columns=[config.target])
    y_raw = df[config.target]
    keep = set(config.force_keep_columns)
    n = len(df)

    profile_rows = []
    date_cols, text_cols = [], []
    for col in list(X.columns):
        s = X[col]
        n_missing = int(s.isna().sum())
        nunique = int(s.nunique(dropna=True))
        flags = []

        miss_frac = n_missing / n
        top_frac = (s.value_counts(dropna=True).iloc[0] / max(n - n_missing, 1)) if nunique else 1.0
        is_datetime = pd.api.types.is_datetime64_any_dtype(s)
        is_textlike = (not pd.api.types.is_numeric_dtype(s)) and not is_datetime

        avg_len, looks_like_datetime = 0.0, False
        if is_textlike:
            sample = s.dropna().astype(str).head(500)
            if len(sample):
                avg_len = float(sample.str.len().mean())
                looks_like_datetime = _sniff_datetime(sample)

        is_date = is_datetime or looks_like_datetime
        if miss_frac >= config.max_missing_frac:
            flags.append(f"missing {miss_frac:.0%}")
        if nunique <= 1 or top_frac >= config.near_constant_thresh:
            flags.append("constant/near-constant")
        if (nunique / max(n, 1) >= config.id_unique_frac
                and not is_date
                and (is_textlike or pd.api.types.is_integer_dtype(s))):
            # timestamps are naturally near-unique; they take the datetime
            # decomposition path instead of the ID drop
            flags.append("ID-like")
        is_text = (not is_date) and avg_len >= config.text_min_avg_len
        if is_date:
            flags.append("datetime")
        if is_text:
            flags.append("free text")
        if is_textlike and not is_date and not is_text and nunique > config.high_cardinality_thresh:
            flags.append(f"high cardinality ({nunique} levels)")

        profile_rows.append({"feature": col, "dtype": str(s.dtype),
                             "missing_%": round(100 * miss_frac, 2),
                             "n_unique": nunique, "flags": ", ".join(flags) or "—"})

        hard_drop = [f for f in flags if f.startswith(("missing", "constant", "ID-like"))]
        if hard_drop and col not in keep:
            result.dropped[col] = ", ".join(hard_drop)
            log.log("quality", f"Dropping '{col}': {', '.join(hard_drop)}")
            X = X.drop(columns=[col])
            continue
        if is_date and col not in keep:
            date_cols.append(col)
        elif is_text and col not in keep:
            text_cols.append(col)

    result.profile = pd.DataFrame(profile_rows).set_index("feature")

    # --- datetime columns: decompose into model-ready parts or drop --------
    for col in date_cols:
        if config.date_features:
            X, derived = decompose_datetime(X, col)
            if derived:
                result.recipe["date_decompositions"][col] = derived
                log.log("dates", f"Decomposed '{col}' -> {derived}")
            else:
                result.dropped[col] = "datetime with no usable parts"
                log.log("dates", f"Dropping '{col}': datetime parses but no part carries signal")
        else:
            X = X.drop(columns=[col])
            result.dropped[col] = "raw datetime (date_features=False)"
            log.log("dates", f"Dropping '{col}': raw datetime (date_features disabled)")

    # --- free-text columns: TF-IDF or drop ----------------------------------
    if text_cols and config.text_features:
        kept_text = text_cols[: config.max_text_columns]
        for col in text_cols[config.max_text_columns:]:
            X = X.drop(columns=[col])
            result.dropped[col] = f"free text beyond max_text_columns={config.max_text_columns}"
            log.log("text", f"Dropping '{col}': more than {config.max_text_columns} text columns")
        for col in kept_text:
            X[col] = X[col].astype(str).where(X[col].notna(), "")
        result.text_columns = kept_text
        result.recipe["text_fillna"] = kept_text
        if kept_text:
            log.log("text", f"Text columns -> TF-IDF features: {kept_text}")
    elif text_cols:
        for col in text_cols:
            X = X.drop(columns=[col])
            result.dropped[col] = "free text (text_features=False)"
            log.log("text", f"Dropping '{col}': free text (text_features disabled)")

    # --- normalize remaining categoricals to plain strings (NaN preserved) --
    cat_cols = [c for c in X.columns
                if not pd.api.types.is_numeric_dtype(X[c]) and c not in result.text_columns]
    for c in cat_cols:
        na_mask = X[c].isna()
        X[c] = X[c].astype(str)
        X.loc[na_mask, c] = np.nan
    if cat_cols:
        result.recipe["categorical_as_string"] = cat_cols
        log.log("quality", f"Normalized {len(cat_cols)} categorical column(s) to string dtype")

    if X.shape[1] == 0:
        raise AutoMLError(
            "Every feature was dropped by the quality guardrails: "
            + "; ".join(f"'{c}' ({r})" for c, r in result.dropped.items())
            + ". Use force_keep_columns to override a guardrail, or fix the data upstream."
        )

    log.log("quality", f"{len(result.dropped)} feature(s) dropped, {X.shape[1]} remain "
                       f"({len(result.text_columns)} text)")
    return X, y_raw, result
