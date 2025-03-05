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
    print(f"CV params: {cv_params}")
    if cv_params is None:
        cv_params = {}
    
    method = cv_params.get('method', 'expanding')
    step_length = cv_params.get('step', 1)
    
    # Determine initial window size if not provided
    if cv_initial_window is None:
        if y_train is not None:
            # Default to 70% of training data
            initial_window = max(2, int(len(y_train) * 0.7))
        else:
            # Use a reasonable default
            initial_window = 10
    else:
        initial_window = cv_initial_window
    
    # Ensure initial window is at least 2 observations
    initial_window = max(initial_window, 2)
    
    # Create forecasting horizon for CV
    fh = np.arange(1, forecast_horizon + 1)
    
    if method.lower() == 'expanding':
        return ExpandingWindowSplitter(
            initial_window=initial_window,
            step_length=step_length,
            fh=fh
        )
    elif method.lower() == 'sliding':
        window_length = cv_params.get('window_length', initial_window)
        return SlidingWindowSplitter(
            window_length=window_length,
            step_length=step_length,
            fh=fh
        )
    else:
        print(f"Warning: Unknown CV method '{method}'. Using ExpandingWindowSplitter.")
        return ExpandingWindowSplitter(
            initial_window=initial_window,
            step_length=step_length,
            fh=fh
        )