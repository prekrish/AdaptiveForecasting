"""
Input validation utilities for the AdaptiveForecaster.
"""

import pandas as pd


def validate_inputs(df, target, algorithms, selected_forecasters, transformations, 
                    seasonal_period, exog_variables, impute_method, impute_value, 
                    impute_forecaster, valid_transforms, valid_impute_methods):
    """
    Validate user inputs.
    
    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        Input data.
    target : str
        Target column name.
    algorithms : list
        List of algorithms to use.
    selected_forecasters : list
        List of selected forecasters to use.
    transformations : list
        List of transformations to apply.
    seasonal_period : int
        Seasonal period for seasonal transformations.
    exog_variables : list
        List of exogenous variables.
    impute_method : str
        Method for imputing missing values.
    impute_value : float
        Value to use for constant imputation.
    impute_forecaster : object
        Forecaster to use for imputation.
    valid_transforms : list
        List of valid transformations.
    valid_impute_methods : list
        List of valid imputation methods.
        
    Raises
    ------
    ValueError
        If input validation fails.
    """
    # Check if target exists in dataframe (only for DataFrame inputs)
    if isinstance(df, pd.DataFrame) and target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")
    
    # Validate selected_forecasters is a subset of algorithms
    invalid_forecasters = [f for f in selected_forecasters if f not in algorithms]
    if invalid_forecasters:
        raise ValueError(f"Selected forecasters {invalid_forecasters} are not in the algorithms list: {algorithms}")
    
    # Validate transformations
    if transformations is not None:
        for transform in transformations:
            if transform.lower() not in valid_transforms:
                raise ValueError(f"Transformation '{transform}' not supported. Choose from {valid_transforms}")
                
        # Check if seasonal_period is provided for seasonal transformations
        if 'deseasonalize' in transformations and seasonal_period is None:
            raise ValueError("seasonal_period must be provided when using 'deseasonalize' transformation.")
        
    # Validate impute method if impute transformation is used
    if transformations is not None and 'impute' in transformations:
        if impute_method not in valid_impute_methods:
            raise ValueError(f"Impute method '{impute_method}' not supported. Choose from {valid_impute_methods}")
            
        # Check if required parameters are provided for specific impute methods
        if impute_method == 'constant' and impute_value is None:
            raise ValueError("impute_value must be provided when impute_method='constant'")
            
        if impute_method == 'forecaster' and impute_forecaster is None:
            raise ValueError("impute_forecaster must be provided when impute_method='forecaster'")
        
    # Validate exogenous variables
    if exog_variables is not None:
        if isinstance(df, pd.Series):
            raise ValueError("Exogenous variables are not supported when using Series input. Please provide a DataFrame.")
        
        for var in exog_variables:
            if var not in df.columns:
                raise ValueError(f"Exogenous variable '{var}' not found in dataframe.")


def process_algorithms(algorithms):
    """
    Process and validate algorithms parameter.
    
    Parameters
    ----------
    algorithms : str or list
        Algorithms to use.
        
    Returns
    -------
    list
        Processed algorithms list.
    """
    if algorithms is None:
        return ['naive', 'arima', 'ets']
    elif isinstance(algorithms, str):
        return [algorithms]
    else:
        return algorithms


def process_transformations(transformations):
    """
    Process transformations parameter.
    
    Parameters
    ----------
    transformations : str or list
        Transformations to apply.
        
    Returns
    -------
    list
        Processed transformations list.
    """
    if isinstance(transformations, str):
        return [transformations]
    return transformations


def process_selected_forecasters(selected_forecasters, algorithms):
    """
    Process selected_forecasters parameter.
    
    Parameters
    ----------
    selected_forecasters : str or list
        Selected forecasters to use.
    algorithms : list
        List of all available algorithms.
        
    Returns
    -------
    list
        Processed selected forecasters list.
    """
    if selected_forecasters is None:
        # If not specified, use all algorithms
        return algorithms.copy()
    elif isinstance(selected_forecasters, str):
        return [selected_forecasters]
    else:
        return selected_forecasters