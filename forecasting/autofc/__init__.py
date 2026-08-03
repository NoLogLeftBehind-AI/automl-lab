"""autofc — automated time-series forecasting for single and multi-series
(panel) data, built on statsmodels, Prophet, and gradient boosting.

    from autofc import AutoForecast, ForecastConfig

    fc = AutoForecast(ForecastConfig(horizon=28, series_col="series"))
    fc.run(df)               # or: load() -> backtest() -> forecast() -> export()
    fc.leaderboard           # rolling-origin backtest leaderboard (MASE primary)
    fc.forecast_frame        # final H-step forecast with prediction intervals
"""
from .config import ForecastConfig
from .core import AutoForecast
from .data import from_wide
from .utils import ForecastError

__all__ = ["AutoForecast", "ForecastConfig", "ForecastError", "from_wide"]
__version__ = "0.1.0"
