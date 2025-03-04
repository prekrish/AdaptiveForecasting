"""
Data processing utilities for the AdaptiveForecaster.
"""

import pandas as pd
import numpy as np
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split


def process_input_data(df, target=None, date_column=None):
    """
    Process input data and handle Series/DataFrame conversion.
    
    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        Input data.
    target : str, optional
        Target column name if df is a DataFrame.
    date_column : str, optional
        Date column name if df is a DataFrame.
        
    Returns
    -------
    pandas.Series or pandas.DataFrame
        Processed data.
    
    Raises
    ------
    ValueError
        If input validation fails.
    """
    if isinstance(df, pd.Series):
        # For Series input, just use it directly
        # Store the name for reference but keep as Series
        target_name = df.name if df.name is not None else 'y'
        return df, target_name, None
    else:
        # For DataFrame input, require explicit target and date_column
        if target is None:
            raise ValueError("When providing a DataFrame, you must specify a target column name.")
                
        # Verify target column exists
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe. Available columns: {list(df.columns)}")
                
        # Handle date_column if provided
        if date_column is not None:
            if date_column in df.columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                    try:
                        df[date_column] = pd.to_datetime(df[date_column])
                    except Exception as e:
                        raise ValueError(f"Could not convert {date_column} to datetime: {str(e)}")
                
                # Set the date column as index
                df = df.set_index(date_column)
                
                # Try to infer frequency
                if df.index.freq is None:
                    try:
                        inferred_freq = pd.infer_freq(df.index)
                        if inferred_freq is not None:
                            df = df.asfreq(inferred_freq)
                    except Exception as e:
                        print(f"Warning: Could not infer frequency: {str(e)}")
            else:
                raise ValueError(f"Date column '{date_column}' not found in dataframe. Available columns: {list(df.columns)}")
        else:
            # If no date_column provided for DataFrame, raise error
            raise ValueError("When providing a DataFrame, you must specify a date_column name.")
                
        # Extract the target series from the DataFrame
        return df[target], target, date_column


def prepare_data(df, test_size, forecast_horizon, exog_variables=None, transformations=None, seasonal_period=None):
    """
    Prepare data for forecasting.
    
    Parameters
    ----------
    df : pandas.Series
        Time series data to forecast.
    test_size : float or int
        Size of the test set.
    forecast_horizon : int
        Number of steps to forecast.
    exog_variables : list, optional
        List of exogenous variables.
    transformations : list, optional
        List of transformations to apply.
    seasonal_period : int, optional
        Seasonal period for seasonal transformations.
        
    Returns
    -------
    tuple
        (y_train, y_test, X_train, X_test, fh) tuple.
    """
    # Extract target
    y = df
    
    # Ensure the index has a frequency if using seasonal transformations
    if (transformations is not None and 
        ('deseasonalize' in transformations or 'detrend' in transformations)):
        if y.index.freq is None:
            try:
                # Try to infer and set frequency
                inferred_freq = pd.infer_freq(y.index)
                if inferred_freq is not None:
                    y = y.copy()
                    y.index = pd.DatetimeIndex(y.index, freq=inferred_freq)
                else:
                    print("Warning: Could not infer frequency from time series index. "
                         "This may cause issues with seasonal decomposition.")
            except Exception as e:
                print(f"Warning: Error setting frequency: {str(e)}")
    
    X = None
    if exog_variables is not None:
        # Note: This needs to be implemented if exogenous variables are supported
        # X = df[exog_variables]
        raise ValueError("Exogenous variables are not supported when using Series input. Please provide a DataFrame.")
    
    # Create train-test split
    y_train, y_test = temporal_train_test_split(y, test_size=test_size)
    X_train, X_test = None, None
    
    # Create forecasting horizon
    fh = ForecastingHorizon(
        np.arange(1, forecast_horizon + 1), is_relative=True
    )
    
    return y_train, y_test, X_train, X_test, fh