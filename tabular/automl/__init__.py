"""automl — an automated modeling engine for tabular classification and
regression, built on scikit-learn.

    from automl import AutoML, AutoMLConfig

    aml = AutoML(AutoMLConfig(target="income", task="classification"))
    aml.run(df)                      # or call the stages one by one
    aml.leaderboard                  # de-biased final leaderboard
    aml.holdout_metrics              # locked-holdout evaluation
    # artifacts: model.joblib, metadata.json, predict.py, drift_check.py, report.html
"""
from .config import AutoMLConfig
from .core import AutoML
from .utils import AutoMLError

__all__ = ["AutoML", "AutoMLConfig", "AutoMLError"]
__version__ = "0.1.0"
