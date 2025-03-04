"""
Adaptive Forecaster - A user-friendly wrapper for time series forecasting using sktime
===================================================================================

Provides a simple interface for users to perform time series forecasting by
abstracting away the complexities of underlying algorithms and processes.

Classes
-------
AdaptiveForecaster
    Main forecasting interface class that simplifies working with sktime.
"""

__version__ = '0.1.0'

from adaptiveforecast.core import AdaptiveForecaster

__all__ = ['AdaptiveForecaster']