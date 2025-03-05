"""
Module for creating and configuring forecasting models.
"""
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.lag import Lag
from sktime.transformations.series.impute import Imputer

def get_algorithm(name, transformations=None, seasonal_period=None):
    """
    Get a forecasting algorithm by name.
    
    Parameters
    ----------
    name : str
        Algorithm name.
    transformations : list, optional
        List of transformations to apply.
    seasonal_period : int, optional
        Seasonal period for seasonal models.
        
    Returns
    -------
    object
        A sktime forecaster instance.
    """
    # Create base forecaster
    if name.lower() == 'naive':
        algorithm = NaiveForecaster(strategy="mean")
    elif name.lower() == 'theta':
        algorithm = ThetaForecaster()
    elif name.lower() == 'arima':
        algorithm = AutoARIMA(
            start_p=1, start_q=1, max_p=3, max_q=3, 
            seasonal=seasonal_period is not None,
            sp=seasonal_period,
            d=1, max_d=2,
            suppress_warnings=True
        )
    elif name.lower() == 'ets':
        algorithm = AutoETS(
            auto=True, 
            seasonal=seasonal_period is not None,
            sp=seasonal_period
        )
    elif name.lower() == 'exp_smoothing':
        algorithm = ExponentialSmoothing(
            seasonal=seasonal_period is not None,
            sp=seasonal_period
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")
    
    # Apply transformations if specified
    if transformations:
        transformers = []
        
        # Process each transformation
        for transform in transformations:
            if transform is None:
                continue
                
            # Handle string transformations
            if isinstance(transform, str):
                transform_name = transform.lower()
                
                if transform_name == 'deseasonalize':
                    transformers.append(
                        ("deseasonalize", Deseasonalizer(sp=seasonal_period))
                    )
                elif transform_name == 'detrend':
                    transformers.append(
                        ("detrend", Detrender())
                    )
                elif transform_name == 'boxcox':
                    transformers.append(
                        ("boxcox", BoxCoxTransformer())
                    )
                elif transform_name == 'lag':
                    lag_values = [1, 2, 3]  # Example lag values
                    transformers.append(
                        ("lag", Lag(lags=lag_values))
                    )
                elif transform_name == 'impute':
                    transformers.append(
                        ("impute", Imputer(method="drift", missing_values=None))
                    )
            
            # Handle dictionary transformations
            elif isinstance(transform, dict) and 'name' in transform:
                transform_name = transform['name'].lower()
                
                if transform_name == 'impute':
                    # Extract parameters
                    method = transform.get('method', 'drift')
                    missing_values = transform.get('missing_values', None)
                    value = transform.get('value', None)
                    
                    transformers.append(
                        ("impute", Imputer(
                            method=method, 
                            missing_values=missing_values,
                            value=value
                        ))
                    )
                elif transform_name == 'deseasonalize':
                    model = transform.get('model', 'additive')
                    transformers.append(
                        ("deseasonalize", Deseasonalizer(
                            sp=seasonal_period,
                            model=model
                        ))
                    )
                elif transform_name == 'detrend':
                    model = transform.get('model', 'linear')
                    transformers.append(
                        ("detrend", Detrender(
                            model=model
                        ))
                    )
        
        if transformers:
            # Add the forecaster as the final step
            transformers.append(("forecaster", algorithm))
            
            # Create the transformed target forecaster
            algorithm = TransformedTargetForecaster(steps=transformers)
    
    return algorithm

def get_param_grid(algorithm_name, grid_search):
    """
    Get default parameter grid for an algorithm.
    
    Parameters
    ----------
    algorithm_name : str
        Algorithm name.
    grid_search : dict or bool
        Grid search configuration.
    
    Returns
    -------
    dict
        Parameter grid.
    """
    # Check if grid_search is a dictionary with algorithm-specific params
    if isinstance(grid_search, dict) and algorithm_name in grid_search:
        return grid_search[algorithm_name]
    
    # If grid_search is False, return empty dict
    if grid_search is False:
        return {}
        
    # Default parameter grids
    if algorithm_name == 'naive':
        return {
            "forecaster__strategy": ["mean", "last", "drift"],
            "forecaster__window_length": [3, 6, 12],
            "forecaster__sp": [12]  # Default seasonal period
        }
    elif algorithm_name == 'theta':
        return {"forecaster__theta": [0, 0.5, 1, 1.5, 2]}
    elif algorithm_name == 'arima':
        return {
            "forecaster__d": [0, 1], 
            "forecaster__max_p": [2, 3], 
            "forecaster__max_q": [2, 3],
            "forecaster__sp": [12]  # Default seasonal period
        }
    elif algorithm_name == 'ets':
        return {
            "forecaster__error": ["add", "mul"], 
            "forecaster__trend": ["add", "mul", None],
            "forecaster__damped_trend": [True, False]
        }
    elif algorithm_name == 'exp_smoothing':
        return {
            "forecaster__trend": ["add", "mul", None],
            "forecaster__seasonal": ["add", "mul", None],
            "forecaster__damped_trend": [True, False]
        }
    elif algorithm_name == 'impute':
        return {
            "impute__method": ["mean", "median", "ffill", "bfill", "drift"]
        }
    elif algorithm_name == 'deseasonalize':
        return {
            "deseasonalize__model": ["additive", "multiplicative"]
        }
    elif algorithm_name == 'detrend':
        return {
            "detrend__model": ["linear", "polynomial"]
        }
    else:
        return {}