"""
Transformation utilities for the AdaptiveForecaster.
"""
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sktime.forecasting.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter
import numpy as np


def get_cv_splitter(cv_params=None, cv_initial_window=None, forecast_horizon=10, y_train=None):
    """
    Get cross-validation splitter based on parameters.
    
    Parameters
    ----------
    cv_params : dict, optional
        Cross-validation parameters.
    cv_initial_window : int, optional
        Initial window size.
    forecast_horizon : int, optional
        Forecast horizon.
    y_train : pandas.Series, optional
        Training data, used to calculate default initial window size.
    
    Returns
    -------
    object
        CV splitter instance.
    """
    cv_params = cv_params or {}
    method = cv_params.get('method', 'expanding')
    step_length = cv_params.get('step', 1)
    
    initial_window = _calculate_initial_window(cv_initial_window, y_train)
    fh = np.arange(1, forecast_horizon + 1)
    
    if method.lower() == 'sliding':
        window_length = cv_params.get('window_length', initial_window)
        return SlidingWindowSplitter(
            window_length=window_length,
            step_length=step_length,
            fh=fh
        )
    
    return ExpandingWindowSplitter(
        initial_window=initial_window,
        step_length=step_length,
        fh=fh
    )

def _calculate_initial_window(cv_initial_window, y_train):
    """Calculate initial window size for CV."""
    if cv_initial_window is not None:
        return max(cv_initial_window, 2)
    
    if y_train is not None:
        return max(2, int(len(y_train) * 0.7))
    
    return 10