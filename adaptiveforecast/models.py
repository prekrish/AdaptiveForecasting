"""
Module for creating and configuring forecasting models.
"""

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.compose import (
    EnsembleForecaster,
    TransformedTargetForecaster,
    MultiplexForecaster
)


def get_algorithm(name, transformations=None, seasonal_period=None, forecasters=None, 
                 impute_method=None, impute_value=None, impute_forecaster=None, 
                 missing_values=None):
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
    forecasters : dict, optional
        Dictionary of already fitted forecasters, needed for ensemble.
    impute_method : str, optional
        Method for imputing missing values.
    impute_value : float, optional
        Value to use for constant imputation.
    impute_forecaster : object, optional
        Forecaster to use for imputation.
    missing_values : list or float, optional
        Values to consider as missing.
    
    Returns
    -------
    object
        Forecasting algorithm instance.
    """
    algorithm = None
    
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
    elif name.lower() == 'ensemble':
        # Only create ensemble if we have other algorithms
        if forecasters is None or len(forecasters) < 2:
            raise ValueError("Ensemble requires at least 2 other algorithms to be trained first.")
        
        forecaster_list = list(forecasters.values())
        algorithm = EnsembleForecaster(forecasters=forecaster_list)
        
    # Apply transformations if specified
    if transformations is not None and algorithm is not None:
        algorithm = apply_transformations(
            algorithm, 
            transformations, 
            seasonal_period, 
            impute_method, 
            impute_value, 
            impute_forecaster, 
            missing_values
        )
        
    return algorithm


def apply_transformations(forecaster, transformations, seasonal_period=None, impute_method=None, 
                         impute_value=None, impute_forecaster=None, missing_values=None):
    """
    Apply specified transformations to a forecaster.
    
    Parameters
    ----------
    forecaster : object
        Base forecaster to transform.
    transformations : list
        List of transformations to apply.
    seasonal_period : int, optional
        Seasonal period for seasonal transformations.
    impute_method : str, optional
        Method for imputing missing values.
    impute_value : float, optional
        Value to use for constant imputation.
    impute_forecaster : object, optional
        Forecaster to use for imputation.
    missing_values : list or float, optional
        Values to consider as missing.
    
    Returns
    -------
    object
        Transformed forecaster.
    """
    from sktime.transformations.series.detrend import Deseasonalizer, Detrender
    from sktime.transformations.series.boxcox import BoxCoxTransformer
    from sktime.transformations.series.lag import Lag
    from sktime.transformations.series.impute import Imputer
    
    transformers = []
    
    if transformations is None:
        return forecaster
    
    for transform in transformations:
        if transform.lower() == 'deseasonalize':
            transformers.append(
                ("deseasonalize", Deseasonalizer(sp=seasonal_period))
            )
        elif transform.lower() == 'detrend':
            transformers.append(
                ("detrend", Detrender())
            )
        elif transform.lower() == 'boxcox':
            transformers.append(
                ("boxcox", BoxCoxTransformer())
            )
        elif transform.lower() == 'lag':
            lag_values = [1, 2, 3]  # Example lag values
            transformers.append(
                ("lag", Lag(lags=lag_values))
            )
        elif transform.lower() == 'impute':
            # Configure imputer based on parameters
            if impute_method == 'constant':
                imputer = Imputer(
                    method=impute_method,
                    value=impute_value,
                    missing_values=missing_values
                )
            elif impute_method == 'forecaster':
                imputer = Imputer(
                    method=impute_method,
                    forecaster=impute_forecaster,
                    missing_values=missing_values
                )
            elif impute_method == 'random':
                imputer = Imputer(
                    method=impute_method,
                    random_state=42,  # Using a fixed seed for reproducibility
                    missing_values=missing_values
                )
            else:
                imputer = Imputer(
                    method=impute_method,
                    missing_values=missing_values
                )
            
            transformers.append(
                ("impute", imputer)
            )
            
    if not transformers:
        return forecaster
        
    transformed_forecaster = TransformedTargetForecaster(
        steps=[*transformers, ("forecaster", forecaster)]
    )
    
    return transformed_forecaster


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
    if isinstance(grid_search, dict) and algorithm_name in grid_search:
        return grid_search[algorithm_name]
        
    # Default parameter grids
    if algorithm_name == 'naive':
        return {"strategy": ["mean", "last", "drift"]}
    elif algorithm_name == 'theta':
        return {"theta": [0, 0.5, 1, 1.5, 2]}
    elif algorithm_name == 'arima':
        return {"d": [0, 1], "max_p": [2, 3], "max_q": [2, 3]}
    elif algorithm_name == 'ets':
        return {"error": ["add", "mul"], "trend": ["add", "mul", None]}
    elif algorithm_name == 'exp_smoothing':
        return {
            "trend": ["add", "mul", None],
            "seasonal": ["add", "mul", None],
            "damped_trend": [True, False]
        }
    else:
        return {}


def create_multiplex_forecaster(algorithms, selected_forecasters, transformations, seasonal_period, 
                              impute_method, impute_value, impute_forecaster, missing_values):
    """
    Create a MultiplexForecaster with the specified algorithms.
    
    Parameters
    ----------
    algorithms : list
        List of algorithms to include.
    selected_forecasters : list
        List of selected forecasters to tune.
    transformations : list
        List of transformations to apply.
    seasonal_period : int
        Seasonal period for seasonal models.
    impute_method : str
        Method for imputing missing values.
    impute_value : float
        Value to use for constant imputation.
    impute_forecaster : object
        Forecaster to use for imputation.
    missing_values : list or float
        Values to consider as missing.
    
    Returns
    -------
    tuple
        MultiplexForecaster and forecaster list.
    """
    # Create a list of forecasters with transformations applied
    forecaster_list = []
    
    for algorithm_name in algorithms:
        if algorithm_name == 'ensemble':
            continue  # Skip ensemble for MultiplexForecaster
            
        print(f"Preparing {algorithm_name.upper()} model...")
        
        # Get base forecaster with transformations
        forecaster = get_algorithm(
            algorithm_name, 
            transformations, 
            seasonal_period, 
            None,  # No forecasters yet for ensemble
            impute_method, 
            impute_value, 
            impute_forecaster, 
            missing_values
        )
        
        if forecaster is not None:
            forecaster_list.append((algorithm_name, forecaster))
    
    if not forecaster_list:
        raise ValueError("No valid forecasters to fit.")
        
    # Create MultiplexForecaster
    multiplex = MultiplexForecaster(
        forecasters=forecaster_list,
        selected_forecaster=None  # Default to first forecaster initially
    )
    
    return multiplex, forecaster_list


def add_algorithm_params_to_grid(param_grid, selected_forecasters, grid_search):
    """
    Add algorithm-specific parameters to the parameter grid.
    
    Parameters
    ----------
    param_grid : dict
        Parameter grid to modify.
    selected_forecasters : list
        List of forecasters to include in the grid.
    grid_search : dict or bool
        Grid search configuration.
    
    Returns
    -------
    dict
        Modified parameter grid.
    """
    if isinstance(grid_search, dict):
        for algo_name in param_grid["selected_forecaster"]:
            if algo_name in grid_search:
                for param, values in grid_search[algo_name].items():
                    param_grid[f"{algo_name}__{param}"] = values
    else:
        # Add default parameter grids for each algorithm
        for algo_name in param_grid["selected_forecaster"]:
            default_params = get_param_grid(algo_name, {})
            for param, values in default_params.items():
                param_grid[f"{algo_name}__{param}"] = values
    
    return param_grid