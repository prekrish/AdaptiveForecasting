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

def _create_transformer_from_string(transform_name, seasonal_period):
    """Create transformer from string name."""
    transform_name = transform_name.lower()
    transformers = {
        'deseasonalize': lambda: ("deseasonalize", Deseasonalizer(sp=seasonal_period)),
        'detrend': lambda: ("detrend", Detrender()),
        'boxcox': lambda: ("boxcox", BoxCoxTransformer()),
        'lag': lambda: ("lag", Lag(lags=[1, 2, 3])),
        'impute': lambda: ("impute", Imputer(method="drift", missing_values=None))
    }
    
    return transformers.get(transform_name, lambda: None)()

def _create_transformer_from_dict(transform_dict, seasonal_period):
    """Create transformer from dictionary configuration."""
    transform_name = transform_dict['name'].lower()
    
    if transform_name == 'impute':
        return ("impute", Imputer(
            method=transform_dict.get('method', 'drift'),
            missing_values=transform_dict.get('missing_values', None),
            value=transform_dict.get('value', None)
        ))
    
    elif transform_name == 'deseasonalize':
        return ("deseasonalize", Deseasonalizer(
            sp=seasonal_period,
            model=transform_dict.get('model', 'additive')
        ))
    
    elif transform_name == 'detrend':
        return ("detrend", Detrender(
            model=transform_dict.get('model', 'linear')
        ))
    
    return None

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
    algorithm = _create_base_algorithm(name.lower(), seasonal_period)
    
    if not transformations:
        return algorithm
        
    transformers = _process_transformations(transformations, seasonal_period)
    if not transformers:
        return algorithm
        
    transformers.append(("forecaster", algorithm))
    return TransformedTargetForecaster(steps=transformers)

def _create_base_algorithm(name, seasonal_period):
    """Create base forecasting algorithm."""
    algorithms = {
        'naive': lambda: NaiveForecaster(strategy="mean"),
        'theta': lambda: ThetaForecaster(),
        'arima': lambda: AutoARIMA(
            start_p=1, start_q=1, max_p=3, max_q=3, 
            seasonal=seasonal_period is not None,
            sp=seasonal_period, d=1, max_d=2,
            suppress_warnings=True
        ),
        'ets': lambda: AutoETS(
            auto=True, 
            seasonal=seasonal_period is not None,
            sp=seasonal_period
        ),
        'exp_smoothing': lambda: ExponentialSmoothing(
            seasonal=seasonal_period is not None,
            sp=seasonal_period
        )
    }
    
    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}")
    
    return algorithms[name]()

def _process_transformations(transformations, seasonal_period):
    """Process and create transformation steps."""
    transformers = []
    
    for transform in transformations:
        if transform is None:
            continue
            
        if isinstance(transform, str):
            transformer = _create_transformer_from_string(transform, seasonal_period)
        elif isinstance(transform, dict) and 'name' in transform:
            transformer = _create_transformer_from_dict(transform, seasonal_period)
        else:
            continue
            
        if transformer:
            transformers.append(transformer)
            
    return transformers

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