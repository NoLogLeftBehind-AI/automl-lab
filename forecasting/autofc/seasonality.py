"""Seasonal-period detection.

Candidates come from the calendar (daily data -> weekly/yearly, hourly ->
daily/weekly, ...) and are confirmed by **seasonal strength** from a classical
decomposition: detrend with a centered moving average of the candidate period,
average the detrended values by phase, and measure how much variance the
seasonal means explain (Hyndman's F_s measure, classical flavor). A candidate
only counts if the data genuinely repeats at that period.

Why not ACF on the differenced series (the obvious alternative): first
differencing all but erases long-period cycles — on daily electricity load it
fails to confirm even the yearly cycle, which is how this implementation
earned its regression test.

The strongest confirmed *short* candidate becomes the primary period used by
SARIMAX/ETS/Theta and seasonal-naive; long confirmed periods (e.g. yearly)
enter SARIMAX as Fourier regressors and the ML models as calendar features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .data import Panel
from .utils import DecisionLog, base_freq_alias

# calendar candidates per pandas frequency family (periods in steps at n=1)
_CANDIDATES = {
    "h": [24, 168], "D": [7, 365], "W": [52], "M": [12], "ME": [12],
    "MS": [12], "Q": [4], "QE": [4], "QS": [4], "B": [5], "min": [60, 1440],
}
STRENGTH_CONFIRM = 0.10


def _candidates_for_freq(freq) -> list:
    """Calendar candidates for a pandas frequency string.

    The base alias is matched exactly, case-insensitively — prefix matching
    would send 'min' to the monthly entry — and multiplied frequencies scale
    the candidate periods ('30min' data has its daily cycle at 48 steps, not
    1440). Anchored suffixes ('W-SUN', 'Q-DEC') and the legacy minute alias
    'T' are normalized first.
    """
    if not freq:
        return []
    base = str(freq).split("-")[0]
    i = 0
    while i < len(base) and base[i].isdigit():
        i += 1
    mult = int(base[:i]) if i else 1
    alias = base_freq_alias(freq)
    for key, cands in _CANDIDATES.items():
        if alias == key.lower():
            if mult <= 1:
                return list(cands)
            return sorted({c_m for c in cands if (c_m := int(round(c / mult))) >= 2})
    return []


def _seasonal_strength(y: np.ndarray, m: int) -> float:
    """Bias-corrected variance share explained by phase means of the detrended
    series. Raw phase means spuriously explain ~1/k of the variance when only
    k cycles are observed (a random walk scores ~0.27 at 3 cycles), so the raw
    share is shrunk by that baseline; genuinely seasonal series survive it.

    Scope of the correction: ``m/n`` is the expected explained share under an
    *i.i.d.* null. An integrated series leaves serially-correlated detrended
    residuals whose null strength exceeds ``m/n``, so this shrinks — but does
    not eliminate — spurious confirmations at long candidate periods on
    random-walk-like data. A block-bootstrap or phase-shuffle null would
    calibrate the threshold properly; that is not implemented here.
    """
    n = len(y)
    if n < 3 * m or np.std(y) == 0:
        return 0.0
    s = pd.Series(y, dtype=float)
    trend = s.rolling(m, center=True, min_periods=max(3, m // 2)).mean()
    detrended = (s - trend).to_numpy()
    phase = np.arange(n) % m
    ok = ~np.isnan(detrended)
    if ok.sum() < 3 * m // 2:
        return 0.0
    var_d = float(np.var(detrended[ok]))
    if var_d == 0:
        return 0.0
    seasonal_means = pd.Series(detrended[ok]).groupby(phase[ok]).transform("mean").to_numpy()
    var_r = float(np.var(detrended[ok] - seasonal_means))
    raw = max(0.0, 1.0 - var_r / var_d)
    bias = m / ok.sum()                      # ≈ 1/k for k observed cycles
    return max(0.0, (raw - bias) / (1.0 - bias)) if bias < 1 else 0.0


def detect_seasonality(panel: Panel, config: ForecastConfig, log: DecisionLog) -> dict:
    """Returns {'primary': int, 'confirmed': [int], 'strength': {period: score}}."""
    if config.seasonal_period is not None:
        m = int(config.seasonal_period)
        log.log("seasonality", f"Seasonal period set by config: {m}")
        return {"primary": m, "confirmed": [m] if m > 1 else [], "strength": {}}

    candidates = _candidates_for_freq(panel.freq)

    scores = {}
    for m in candidates:
        vals = []
        for f in panel.frames.values():
            if len(f) >= 3 * m:  # need a few full cycles to confirm
                vals.append(_seasonal_strength(f["y"].to_numpy(), m))
        if vals:
            scores[m] = float(np.median(vals))

    confirmed = [m for m, s in scores.items() if s >= STRENGTH_CONFIRM]
    primary = max(confirmed, key=lambda m: scores[m]) if confirmed else 1
    # prefer a short period as primary (weekly over yearly for daily data),
    # picking the strongest among the short confirmed ones: statistical models
    # handle short seasonality directly, long cycles are better served by
    # Fourier terms / calendar features
    short_confirmed = [m for m in confirmed if m <= 60]
    if short_confirmed:
        primary = max(short_confirmed, key=lambda m: scores[m])

    if confirmed:
        log.log("seasonality", f"Detected seasonal period(s) {confirmed} "
                               f"(median seasonal strength: "
                               f"{ {m: round(scores[m], 2) for m in confirmed} }); "
                               f"primary = {primary}")
    else:
        log.log("seasonality", f"No seasonality confirmed (candidates {candidates}, "
                               f"strengths { {m: round(s, 2) for m, s in scores.items()} }) "
                               "-> non-seasonal models")
    return {"primary": primary, "confirmed": confirmed, "strength": scores}
