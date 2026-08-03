"""Panel loading, frequency inference, and time-series guardrails.

The engine's data contract is a LONG frame: one row per (series, timestamp)
with a date column, a numeric target, an optional series-ID column, and any
number of exogenous columns (which must be knowable for future dates — that's
what makes them usable in a forecast). ``from_wide`` converts the common
wide layout (one column per series) into it.

Guardrails (every action logged):

- unparseable dates, non-numeric targets  -> clear error
- duplicate timestamps within a series    -> mean-aggregated (or error, by config)
- irregular timestamps                    -> reindexed to the inferred frequency grid
- gaps up to ``max_gap_frac``             -> time-interpolated (logged per series)
- gappier series, too-short series,
  constant series                          -> dropped loudly (constant series also
                                             break MASE's scale denominator)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .utils import DecisionLog, ForecastError


class Panel:
    """Regularized per-series frames, all on a shared frequency."""

    def __init__(self):
        self.frames: dict[str, pd.DataFrame] = {}   # series -> frame indexed by date,
                                                     # columns: ["y", *exog_cols]
        self.freq: str | None = None
        self.exog_cols: list[str] = []
        self.exog_codings: dict[str, dict] = {}      # categorical exog -> value->code map
        self.dropped: dict[str, str] = {}

    @property
    def series_ids(self) -> list[str]:
        return list(self.frames)

    def __len__(self):
        return sum(len(f) for f in self.frames.values())

    def truncate(self, cutoff) -> "Panel":
        """Panel restricted to rows at or before ``cutoff`` (for backtesting)."""
        p = Panel()
        p.freq, p.exog_cols = self.freq, self.exog_cols
        p.frames = {sid: f.loc[:cutoff] for sid, f in self.frames.items()}
        p.frames = {sid: f for sid, f in p.frames.items() if len(f) >= 3}
        return p

    def future_index(self, horizon: int) -> pd.DatetimeIndex:
        last = max(f.index[-1] for f in self.frames.values())
        return pd.date_range(last, periods=horizon + 1, freq=self.freq)[1:]

    def summary(self) -> pd.DataFrame:
        rows = []
        for sid, f in self.frames.items():
            rows.append({"series": sid, "rows": len(f),
                         "start": f.index[0], "end": f.index[-1],
                         "mean": round(float(f["y"].mean()), 2),
                         "missing_filled_%": round(100 * f["y_was_filled"].mean(), 2)
                         if "y_was_filled" in f else 0.0})
        return pd.DataFrame(rows).set_index("series")


def from_wide(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    """Wide (one column per series) -> long (ds, series, y)."""
    return df.melt(id_vars=[date_col], var_name="series", value_name="y")


def load_dataframe(config: ForecastConfig) -> pd.DataFrame:
    if config.data_path is None:
        raise ForecastError("No data: set config.data_path or pass a DataFrame to load().")
    path = Path(config.data_path)
    if not path.exists():
        raise ForecastError(f"Data file not found: {path}")
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


_NS_PER = {"D": 86_400_000_000_000, "h": 3_600_000_000_000,
           "min": 60_000_000_000, "s": 1_000_000_000}


def _canonical_fixed_alias(delta: pd.Timedelta):
    """Largest calendar-style alias for a fixed delta: 1 day -> 'D', 2h -> '2h'.

    pandas versions disagree on Timedelta->offset naming (pandas 3 renders one
    day as '24h', pandas 2 as 'D'), so the fallback path names offsets itself.
    """
    ns = delta.value
    if ns <= 0:
        return None
    for alias, unit_ns in _NS_PER.items():
        if ns % unit_ns == 0:
            n = ns // unit_ns
            return alias if n == 1 else f"{n}{alias}"
    return None


def _infer_freq(idx: pd.DatetimeIndex):
    freq = pd.infer_freq(idx)
    if freq:
        return freq
    # fall back to the modal timestamp delta (robust to a few gaps)
    deltas = pd.Series(np.diff(idx.values))
    if not len(deltas):
        return None
    mode = pd.Timedelta(deltas.mode().iloc[0])
    alias = _canonical_fixed_alias(mode)
    if alias:
        return alias
    try:
        return pd.tseries.frequencies.to_offset(mode).freqstr
    except ValueError:
        return None


def _offset_nanos(freq: str):
    """Fixed duration of an offset label in nanoseconds, or None if anchored.

    pandas 3 made 'D' a calendar offset (no longer a Tick), so Timedelta
    conversion raises there — treat Day as 24h explicitly, which is exact for
    the timezone-naive indexes the engine works with.
    """
    try:
        off = pd.tseries.frequencies.to_offset(freq)
    except ValueError:
        return None
    try:
        return off.nanos                       # Tick offsets: h, min, s, pandas-2 'D'
    except (AttributeError, ValueError):
        pass
    if type(off).__name__ == "Day":
        return off.n * _NS_PER["D"]
    return None                                # anchored offsets: W-SUN, M, Q, ...


def _unify_freqs(freqs: dict):
    """One frequency for the panel, or None if series genuinely disagree.

    Labels that denote the same fixed duration ('D' vs '24h') are unified to
    the canonical alias; anchored offsets (e.g. 'W-SUN') must match exactly.
    """
    unique = set(freqs.values())
    if len(unique) == 1:
        return next(iter(unique))
    nanos = {_offset_nanos(f) for f in unique}
    if None in nanos or len(nanos) != 1:
        return None
    return _canonical_fixed_alias(pd.Timedelta(nanos.pop(), unit="ns")) \
        or sorted(unique)[0]


def build_panel(df: pd.DataFrame, config: ForecastConfig, log: DecisionLog) -> Panel:
    for col in [config.date_col, config.target] + ([config.series_col] if config.series_col else []):
        if col not in df.columns:
            raise ForecastError(f"Column '{col}' not found. Available: {list(df.columns)}")

    df = df.copy()
    try:
        df[config.date_col] = pd.to_datetime(df[config.date_col], format="mixed")
    except (TypeError, ValueError):
        try:
            df[config.date_col] = pd.to_datetime(df[config.date_col])
        except (TypeError, ValueError) as e:
            raise ForecastError(f"Could not parse '{config.date_col}' as datetimes: {e}") from e
    try:
        df[config.target] = pd.to_numeric(df[config.target], errors="raise")
    except (TypeError, ValueError) as e:
        raise ForecastError(f"Target '{config.target}' is not numeric: {e}") from e

    reserved = {config.date_col, config.target, config.series_col}
    exog = (list(config.exog_cols) if config.exog_cols is not None
            else [c for c in df.columns if c not in reserved])
    exog_codings = {}
    for c in exog:
        if not pd.api.types.is_numeric_dtype(df[c]):
            codes, uniques = pd.factorize(df[c])
            exog_codings[c] = {str(v): int(i) for i, v in enumerate(uniques)}
            df[c] = codes.astype(float)
            log.log("data", f"Exogenous column '{c}' is categorical -> integer-coded "
                            "(the coding ships in the artifact metadata)")
    if exog:
        log.log("data", f"Exogenous regressors: {exog} — these must be provided for "
                        "future dates at forecast time")

    n_missing_target = int(df[config.target].isna().sum())
    if n_missing_target:
        df = df.dropna(subset=[config.target])
        log.log("data", f"Dropped {n_missing_target} rows with missing target")

    groups = (df.groupby(config.series_col, observed=True) if config.series_col
              else [("series_0", df)])

    panel = Panel()
    panel.exog_cols = exog
    panel.exog_codings = exog_codings
    freqs = {}
    for sid, g in groups:
        sid = str(sid)
        g = g.sort_values(config.date_col)
        dup = g[config.date_col].duplicated().sum()
        if dup:
            if config.duplicate_policy == "error":
                raise ForecastError(f"Series '{sid}' has {dup} duplicate timestamps "
                                    "(set duplicate_policy='mean' to aggregate them).")
            # NOTE: integer-coded categorical exog gets averaged too — duplicate
            # rows that disagree on a category produce fractional codes; the
            # guardrail favors continuity over strictness here
            g = g.groupby(config.date_col, as_index=False).mean(numeric_only=True)
            log.warn("data", f"Series '{sid}': {dup} duplicate timestamps mean-aggregated")
        g = g.set_index(config.date_col)

        if len(g) < 3:
            panel.dropped[sid] = f"only {len(g)} timestamp(s)"
            log.warn("data", f"Series '{sid}' dropped: only {len(g)} timestamp(s) — "
                             "too short to infer a frequency, let alone forecast")
            continue

        freq = config.freq or _infer_freq(g.index)
        if freq is None:
            panel.dropped[sid] = "frequency could not be inferred"
            log.warn("data", f"Series '{sid}' dropped: cannot infer a regular frequency "
                             "from its timestamps (set config.freq to override)")
            continue
        freqs[sid] = freq

        full_idx = pd.date_range(g.index[0], g.index[-1], freq=freq)
        gap_frac = 1 - len(g) / len(full_idx)
        if gap_frac > config.max_gap_frac:
            panel.dropped[sid] = f"{gap_frac:.0%} of timestamps missing"
            log.warn("data", f"Series '{sid}' dropped: {gap_frac:.0%} of the regular "
                             f"grid is missing (> max_gap_frac={config.max_gap_frac:.0%})")
            continue
        frame = g.reindex(full_idx)[[config.target] + exog].rename(columns={config.target: "y"})
        was_missing = frame["y"].isna()
        if was_missing.any():
            # Interpolation runs once over the full series, before backtest
            # truncation — a gap spanning a fold cutoff is filled using both
            # anchors. Filled points are marked (y_was_filled) and excluded
            # from all backtest scoring; interpolation is bounded by
            # max_gap_frac. Leading exog gaps back-fill from the earliest
            # observed value.
            frame["y"] = frame["y"].interpolate(method="time", limit_direction="both")
            frame[exog] = frame[exog].ffill().bfill()
            log.log("data", f"Series '{sid}': filled {int(was_missing.sum())} missing "
                            f"timestamp(s) by time interpolation ({gap_frac:.1%} of grid)")
        frame["y_was_filled"] = was_missing.astype(float)
        panel.frames[sid] = frame

    if freqs:
        unified = config.freq or _unify_freqs(freqs)
        if unified is None:
            raise ForecastError(f"Series disagree on frequency: {freqs}. Resample "
                                "upstream or set config.freq explicitly.")
        panel.freq = unified
    else:
        panel.freq = None
    if not panel.frames:
        raise ForecastError("No usable series left after data guardrails: "
                            + "; ".join(f"'{s}' ({r})" for s, r in panel.dropped.items()))

    # length + variance guardrails (the seasonal-aware minimum is re-checked
    # in AutoForecast.load once the seasonal period is known)
    min_len = (config.min_series_length if config.min_series_length
               else max(2 * config.horizon + 10, 30))
    for sid in list(panel.frames):
        f = panel.frames[sid]
        if len(f) < min_len:
            panel.dropped[sid] = f"only {len(f)} points (< {min_len})"
            log.warn("data", f"Series '{sid}' dropped: {len(f)} points is too short "
                             f"for horizon={config.horizon} with {config.n_backtests} "
                             f"backtests (needs >= {min_len})")
            del panel.frames[sid]
        elif float(f["y"].std()) == 0.0:
            panel.dropped[sid] = "constant target"
            log.warn("data", f"Series '{sid}' dropped: constant target (nothing to "
                             "model, and it breaks MASE scaling)")
            del panel.frames[sid]
    if not panel.frames:
        raise ForecastError("No usable series left after length/variance guardrails: "
                            + "; ".join(f"'{s}' ({r})" for s, r in panel.dropped.items()))

    log.log("data", f"Panel ready: {len(panel.frames)} series, {len(panel):,} rows, "
                    f"freq={panel.freq}" + (f", {len(panel.dropped)} series dropped"
                                            if panel.dropped else ""))
    return panel
