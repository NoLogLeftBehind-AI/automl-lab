"""Holiday calendar features.

When ``config.country_holidays`` is set (e.g. ``"US"``), the engine derives
deterministic holiday indicators — day-of, day-before, day-after — that are
computable for any future date, so they are safe forecast features:

- the global ML models get them as calendar features,
- auto-SARIMAX gets all three indicators as deterministic exogenous input
  (alongside the Fourier terms),
- Prophet uses its own built-in country holidays.

Only meaningful for sub-weekly data (daily/hourly/business-daily/minute);
coarser frequencies skip them with a logged decision.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .utils import base_freq_alias

try:
    import holidays as _holidays_pkg
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False


def holidays_supported_for(freq: str) -> bool:
    # exact family match ('2h' and '30min' count; 'min' must not match monthly)
    return base_freq_alias(freq) in ("d", "h", "b", "min")


@lru_cache(maxsize=8)
def _country_calendar(country: str, years: tuple):
    return _holidays_pkg.country_holidays(country, years=list(years))


def holiday_flags(idx: pd.DatetimeIndex, country: str) -> pd.DataFrame:
    """DataFrame(is_holiday, is_day_before_holiday, is_day_after_holiday)."""
    if not HAS_HOLIDAYS:
        raise ImportError("holiday features need the 'holidays' package: "
                          "pip install holidays")
    years = tuple(range(idx.min().year - 1, idx.max().year + 2))
    cal = _country_calendar(country, years)
    dates = idx.normalize()
    is_hol = np.array([d in cal for d in dates], dtype=float)
    before = np.array([(d + pd.Timedelta(days=1)) in cal for d in dates], dtype=float)
    after = np.array([(d - pd.Timedelta(days=1)) in cal for d in dates], dtype=float)
    return pd.DataFrame({"is_holiday": is_hol, "is_day_before_holiday": before,
                         "is_day_after_holiday": after}, index=idx)
