"""Shared helpers for the forecasting engine (same design language as the
tabular engine: loud guardrails, an auditable decision log,
narrowly-targeted warning filters only)."""
from __future__ import annotations

import warnings
from contextlib import contextmanager

import pandas as pd


class ForecastError(Exception):
    """A guardrail stopped the run — the message says why and what to do."""


def base_freq_alias(freq) -> str:
    """Lowercased base alias of a pandas frequency string, with multiplier and
    anchor stripped: '30min' -> 'min', 'W-SUN' -> 'w', '2h' -> 'h', legacy 'T'
    -> 'min'. Frequency-family decisions must exact-match this — prefix
    matching is how 'min' ends up classified as monthly."""
    if not freq:
        return ""
    base = str(freq).split("-")[0]
    i = 0
    while i < len(base) and base[i].isdigit():
        i += 1
    alias = base[i:].lower()
    return {"t": "min"}.get(alias, alias)


# Targeted filter only (the engine never silences wholesale): LightGBM 4.x's
# sklearn wrapper tracks feature names internally even when fed plain numpy
# arrays, so sklearn's name check fires on every predict despite fit/predict
# receiving identical input.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names, but LGBMRegressor")


class DecisionLog:
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



@contextmanager
def silence_fit_warnings():
    """Context manager for statsmodels/Prophet fits.

    Statistical model fitting emits a steady stream of ConvergenceWarning /
    ValueWarning / RuntimeWarning noise for orders that the AIC search will
    reject anyway; callers count what matters (fit failures) explicitly.
    This is scoped to individual fits — the engine never silences globally.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def quiet_prophet_logs():
    import logging
    for name in ("prophet", "cmdstanpy"):
        logging.getLogger(name).setLevel(logging.WARNING)
