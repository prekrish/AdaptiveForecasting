"""
AdaptiveForecaster - A simplified, user-friendly interface for time series forecasting using sktime.

This package provides an easy-to-use API for time series forecasting with support for:
- Multiple forecasting algorithms (ARIMA, ETS, Naive, etc.)
- Algorithm-specific transformations
- Automatic hyperparameter tuning via grid search
- Cross-validation
- Visualization of forecasts
"""

# Suppress common warnings
import warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", message="The user-specified parameters provided alongside auto=True in AutoETS may not be respected")

__version__ = "0.1.0"

# Import main class for easy access
from adaptiveforecast.forecaster import AdaptiveForecaster

# Make key modules available at package level
from adaptiveforecast import models
from adaptiveforecast import transformations
from adaptiveforecast import visualization

__all__ = ['AdaptiveForecaster']