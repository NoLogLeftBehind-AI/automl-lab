"""Preprocessing blueprints, matched to model family.

Three blueprints — one preprocessing recipe per kind of algorithm:

- ``linear``     — median impute + scale numerics; one-hot low-cardinality
                   categoricals (adaptive rare-level pooling); cross-fitted
                   TargetEncoder for high-cardinality; TF-IDF for text.
- ``tree``       — median impute numerics (RF/ET cannot take NaN); ordinal-encode
                   low-cardinality; TargetEncoder high-cardinality; hashed text.
- ``native_cat`` — for HistGradientBoosting / LightGBM: numerics pass through
                   with NaN intact (both handle missing values natively — median
                   imputation would erase a predictive signal), low-cardinality
                   categoricals become integer codes, high-cardinality get
                   TargetEncoder, text is hashed.

                   Native categorical *splits* are wired for
                   HistGradientBoosting only: ``native_categorical_indices()``
                   feeds its ``categorical_features`` parameter. LightGBM
                   currently receives no ``categorical_feature`` argument and
                   so treats those integer codes as ordered numerics — it takes
                   this blueprint for the native NaN handling, not for
                   categorical splits. Wiring it is a behavior change that
                   would invalidate the committed benchmark and demo numbers,
                   so it is deliberately deferred rather than done silently.

Every component is standard scikit-learn, so the serialized artifact needs no
custom code at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

TEXT_HASH_FEATURES = 256


class FeatureSpec:
    """Which column gets which treatment, resolved once per run."""

    def __init__(self, X: pd.DataFrame, text_columns: list, high_cardinality_thresh: int):
        self.text_cols = [c for c in text_columns if c in X.columns]
        rest = [c for c in X.columns if c not in self.text_cols]
        self.num_cols = [c for c in rest if pd.api.types.is_numeric_dtype(X[c])]
        cats = [c for c in rest if c not in self.num_cols]
        card = {c: X[c].nunique(dropna=True) for c in cats}
        self.low_card_cols = [c for c in cats if card[c] <= high_cardinality_thresh]
        self.high_card_cols = [c for c in cats if card[c] > high_cardinality_thresh]

    @property
    def cat_cols(self):
        return self.low_card_cols + self.high_card_cols

    @property
    def all_features(self):
        return self.num_cols + self.cat_cols + self.text_cols

    def describe(self) -> str:
        return (f"{len(self.num_cols)} numeric, {len(self.low_card_cols)} categorical, "
                f"{len(self.high_card_cols)} high-cardinality categorical, "
                f"{len(self.text_cols)} text")


def _adaptive_min_frequency(n_rows: int) -> int:
    """Rare-level pooling threshold that doesn't wipe out categoricals on small data."""
    return int(np.clip(n_rows // 200, 2, 10))


def _target_encoder(task: str, rng: int) -> TargetEncoder:
    # cross-fitted internally during fit_transform, so it does not leak the target
    return TargetEncoder(target_type="continuous" if task == "regression" else "auto",
                         random_state=rng)


def _text_branch(kind: str):
    if kind == "linear":
        return TfidfVectorizer(max_features=2000, ngram_range=(1, 2), sublinear_tf=True)
    # fixed output width regardless of vocabulary size -> safe on any corpus,
    # dense-friendly width for tree models
    return HashingVectorizer(n_features=TEXT_HASH_FEATURES, alternate_sign=False, norm="l2")


def make_preprocessor(kind: str, spec: FeatureSpec, task: str, rng: int,
                      n_rows: int) -> ColumnTransformer:
    transformers = []

    if kind == "linear":
        if spec.num_cols:
            transformers.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), spec.num_cols))
        if spec.low_card_cols:
            transformers.append(("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=_adaptive_min_frequency(n_rows),
                                      sparse_output=False)),
            ]), spec.low_card_cols))
        if spec.high_card_cols:
            transformers.append(("hicat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("te", _target_encoder(task, rng)),
                ("scale", StandardScaler()),
            ]), spec.high_card_cols))

    elif kind == "tree":
        if spec.num_cols:
            transformers.append(("num", SimpleImputer(strategy="median"), spec.num_cols))
        if spec.low_card_cols:
            transformers.append(("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), spec.low_card_cols))
        if spec.high_card_cols:
            transformers.append(("hicat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("te", _target_encoder(task, rng)),
            ]), spec.high_card_cols))

    elif kind == "native_cat":
        if spec.num_cols:
            transformers.append(("num", "passthrough", spec.num_cols))  # NaN stays: GBMs route it
        if spec.low_card_cols:
            transformers.append(("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan,
                encoded_missing_value=np.nan), spec.low_card_cols))
        if spec.high_card_cols:
            transformers.append(("hicat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("te", _target_encoder(task, rng)),
            ]), spec.high_card_cols))
    else:
        raise ValueError(f"unknown blueprint kind: {kind}")

    for col in spec.text_cols:
        transformers.append((f"txt_{col}", _text_branch(kind), col))

    # tree/native blueprints must stay dense (HistGB rejects sparse input);
    # the linear blueprint may stay sparse so wide TF-IDF blocks don't blow up memory
    return ColumnTransformer(transformers, remainder="drop",
                             sparse_threshold=0.3 if kind == "linear" else 0,
                             verbose_feature_names_out=bool(spec.text_cols))


def native_categorical_indices(spec: FeatureSpec) -> list:
    """Column indices of the ordinal-coded categoricals in the ``native_cat`` output.

    Output column order is the transformer order: numerics, low-card categoricals,
    target-encoded high-card, then text blocks.
    """
    start = len(spec.num_cols)
    return list(range(start, start + len(spec.low_card_cols)))
