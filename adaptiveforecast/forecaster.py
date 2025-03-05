"""
Main module containing the AdaptiveForecaster class.
"""
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import List, Dict, Union, Optional, Tuple

# Suppress specific warnings
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", message="The user-specified parameters provided alongside auto=True in AutoETS may not be respected")

# sktime imports
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter, 
    SlidingWindowSplitter,
    ForecastingGridSearchCV
)
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError, 
    MeanAbsoluteError, 
    MeanSquaredError
)

# Local imports
from adaptiveforecast.models import get_algorithm, get_param_grid
from adaptiveforecast.transformations import get_cv_splitter
from adaptiveforecast.visualization import plot_forecasts, create_forecast_dataframe

class AdaptiveForecaster:
    """
    A user-friendly interface for time series forecasting using sktime.
    
    This class provides a simple interface for users to perform time series forecasting
    by abstracting away the complexities of the underlying algorithms and processes.
    
    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        Input time series data.
    target : str, optional
        Target column name if df is a DataFrame.
    date_column : str, optional
        Date column name if df is a DataFrame.
    forecast_horizon : int, default=10
        Number of steps to forecast.
    test_size : float or int, default=0.2
        Size of the test set.
    algorithms : list or str, default=['naive', 'arima', 'ets']
        List of algorithms to use.
    transformations : list or str or dict, optional
        List of transformations to apply. Can be a dict mapping algorithms to transformations.
    seasonal_period : int, optional
        Seasonal period for seasonal transformations.
    grid_search : bool or dict, default=False
        Grid search configuration.
    cross_validation : bool or dict, default=False
        Cross-validation configuration.
    scoring : str, default="mae"
        Scoring metric to use.
    """
    
    # Mapping of scoring strings to metric objects
    METRIC_MAP = {
        "mae": MeanAbsoluteError(),
        "mse": MeanSquaredError(),
        "mape": MeanAbsolutePercentageError(),
        "rmse": MeanSquaredError(square_root=True)
    }
    
    def __init__(
        self,
        df: Union[pd.DataFrame, pd.Series],
        target: Optional[str] = None,
        date_column: Optional[str] = None,
        forecast_horizon: int = 10,
        test_size: Union[float, int] = 0.2,
        algorithms: Union[List[str], str] = None,
        transformations: Union[List[str], str, Dict[str, List[str]]] = None,
        seasonal_period: Optional[int] = None,
        grid_search: Union[bool, Dict] = False,
        cross_validation: Union[bool, Dict] = False,
        scoring: str = "mae"
    ):
        # Store input parameters
        self.df = df
        self.target = target
        self.date_column = date_column
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size
        self.seasonal_period = seasonal_period
        self.scoring = scoring
        
        # Process algorithms
        if algorithms is None:
            self.algorithms = ['naive', 'arima', 'ets']
        elif isinstance(algorithms, str):
            self.algorithms = [algorithms]
        else:
            self.algorithms = algorithms
            
        # Process transformations
        if transformations is None:
            self.transformations = {}
        elif isinstance(transformations, str):
            self.transformations = {algo: [transformations] for algo in self.algorithms}
        elif isinstance(transformations, list):
            self.transformations = {algo: transformations for algo in self.algorithms}
        elif isinstance(transformations, dict):
            self.transformations = transformations
        else:
            self.transformations = {}
            
        # Process grid search
        if isinstance(grid_search, bool):
            self.grid_search = grid_search
            self.grid_search_params = {}
        else:
            self.grid_search = True
            self.grid_search_params = grid_search or {}
            
        # Process cross-validation
        if isinstance(cross_validation, bool):
            self.cross_validation = cross_validation
            self.cv_params = {}
        else:
            self.cross_validation = True
            self.cv_params = cross_validation or {}
            
        # Initialize other attributes
        self.data = None
        self.y_train = None
        self.y_test = None
        self.fh = None
        self.forecasters = {}
        self.forecasts = {}
        self.metrics = {}
        self.best_model = None
        self.cv_results = {}
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare the data for forecasting."""
        # Handle DataFrame vs Series
        if isinstance(self.df, pd.DataFrame):
            if self.target is None:
                # If no target is specified, use the first column
                self.target = self.df.columns[0]
                
            # Extract target series
            self.data = self.df[self.target].copy()
        else:
            # It's already a Series
            self.data = self.df.copy()
            
        # Split into train and test sets
        self.y_train, self.y_test = temporal_train_test_split(
            self.data, test_size=self.test_size
        )
        
        # Create forecasting horizon
        self.fh = ForecastingHorizon(
            np.arange(1, self.forecast_horizon + 1),
            is_relative=True
        )
        
        print(f"Data prepared: {len(self.y_train)} training samples, {len(self.y_test)} test samples")
        
    def fit(self):
        """Fit all forecasting models."""
        self.forecasters = {}
        self.cv_results = {}
        
        # Create train-test split
        self.y_train, self.y_test = temporal_train_test_split(
            self.df, test_size=self.test_size
        )
        
        # Create forecast horizon
        self.fh = ForecastingHorizon(
            np.arange(1, self.forecast_horizon + 1),
            is_relative=True
        )
        
        # Fit each algorithm
        for name in self.algorithms:
            print(f"Fitting {name}...")
            try:
                # Get transformations for this algorithm
                if isinstance(self.transformations, dict) and self.transformations is not None:
                    algo_transforms = self.transformations.get(name, None)
                else:
                    algo_transforms = self.transformations
                
                # Create algorithm
                algorithm = get_algorithm(name, algo_transforms, self.seasonal_period)
                
                # Check if grid search is enabled
                if self.grid_search:
                    # Get base parameter grid for the algorithm
                    param_grid = get_param_grid(name, self.grid_search_params if isinstance(self.grid_search_params, dict) else self.grid_search)
                    
                    # Add transformer parameters if applicable
                    if isinstance(algo_transforms, list) and algo_transforms:
                        for transform in algo_transforms:
                            if isinstance(transform, str):
                                transform_name = transform
                            elif isinstance(transform, dict) and 'name' in transform:
                                transform_name = transform['name']
                            else:
                                continue
                                
                            # Get parameter grid for this transformer
                            transform_params = get_param_grid(transform_name, self.grid_search_params)
                            
                            # Add to main parameter grid
                            if transform_params:
                                param_grid.update(transform_params)
                    
                    if param_grid:
                        # Create CV splitter
                        cv = get_cv_splitter(
                            self.cv_params, 
                            None, 
                            self.forecast_horizon, 
                            self.y_train
                        )
                        
                        # Create grid search
                        try:
                            grid_search = ForecastingGridSearchCV(
                                forecaster=algorithm,
                                param_grid=param_grid,
                                cv=cv,
                                scoring=self.METRIC_MAP.get(self.scoring, None),
                                strategy="refit",
                                backend="loky",
                                backend_params={"n_jobs": -1}
                            )
                            
                            # Fit grid search
                            grid_search.fit(self.y_train)
                            
                            # Store best forecaster
                            self.forecasters[name] = grid_search
                            self.cv_results[name] = grid_search.cv_results_
                            
                            print(f"  Best parameters: {grid_search.best_params_}")
                            
                        except Exception as e:
                            print(f"  Error in grid search for {name}: {str(e)}")
                            print(f"  Param grid was: {param_grid}")
                            print(f"  Algorithm was: {type(algorithm).__name__}")
                            print(f"  Falling back to direct fit...")
                            
                            # Fall back to direct fit
                            algorithm.fit(self.y_train)
                            self.forecasters[name] = algorithm
                    else:
                        # No grid search, just fit
                        algorithm.fit(self.y_train)
                        self.forecasters[name] = algorithm
                else:
                    # No grid search, just fit
                    algorithm.fit(self.y_train)
                    self.forecasters[name] = algorithm
                
                print(f"  {name.capitalize()} fitted successfully.")
            except Exception as e:
                print(f"  Error fitting {name}: {str(e)}")
                print(f"  Algorithm type: {type(algorithm).__name__ if 'algorithm' in locals() else 'Unknown'}")
                if 'param_grid' in locals():
                    print(f"  Parameter grid: {param_grid}")
                if 'algorithm' in locals() and isinstance(algorithm, TransformedTargetForecaster) and hasattr(algorithm, 'steps'):
                    print(f"  Transformer steps: {algorithm.steps}")
        
        return self
        
    def predict(self):
        """Generate forecasts for all fitted models."""
        if not self.forecasters:
            print("No fitted models available. Run fit() first.")
            return self
            
        print("Generating forecasts...")
        
        self.forecasts = {}
        self.metrics = {}
        
        for name, forecaster in self.forecasters.items():
            try:
                # Generate forecast
                forecast = forecaster.predict(self.fh)
                self.forecasts[name] = forecast
                
                # Calculate metrics if test data is available
                if self.y_test is not None and len(self.y_test) > 0:
                    y_test_pred = self.y_test.iloc[:self.forecast_horizon]
                    forecast_values = forecast[:len(y_test_pred)]
                    
                    # Calculate metrics
                    mae = MeanAbsoluteError().evaluate(y_test_pred, forecast_values)
                    mse = MeanSquaredError().evaluate(y_test_pred, forecast_values)
                    rmse = np.sqrt(mse)
                    mape = MeanAbsolutePercentageError().evaluate(y_test_pred, forecast_values)
                    
                    self.metrics[name] = {
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'mape': mape
                    }
                    
                    print(f"{name.upper()} metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}")
                    
            except Exception as e:
                print(f"Error generating forecast for {name}: {str(e)}")
        
        # Determine best model based on RMSE
        if self.metrics:
            self.best_model = min(self.metrics, key=lambda x: self.metrics[x]['rmse'])
            print(f"Best model: {self.best_model.upper()} (RMSE={self.metrics[self.best_model]['rmse']:.4f})")
        
        return self
    
    def plot_forecasts(self, figsize=(12, 6), include_history=True, models=None):
        """Plot forecasts for all or selected models."""
        if not self.forecasts:
            print("No forecasts available. Run predict() first.")
            return
        
        return plot_forecasts(
            self.y_train, 
            self.y_test, 
            self.forecasts, 
            self.forecast_horizon,
            best_model=self.best_model,
            figsize=figsize,
            include_history=include_history,
            models=models,
            target=self.target or ''
        )
    
    def get_forecast_dataframe(self):
        """Get a DataFrame with forecasts and errors."""
        if not self.forecasts:
            print("No forecasts available. Run predict() first.")
            return None
        
        return create_forecast_dataframe(
            self.y_test,
            self.forecasts,
            self.forecast_horizon
        )
    
    def get_best_params(self):
        """Get the best parameters for the best model."""
        if not self.best_model or self.best_model not in self.forecasters:
            return None
        
        forecaster = self.forecasters[self.best_model]
        if hasattr(forecaster, 'best_params_'):
            return forecaster.best_params_
        
        return None
    
    def save_summary(self, filename="forecast_summary.json"):
        """Save a summary of the forecasting results to a JSON file."""
        if not self.forecasts:
            print("No forecasts available. Run predict() first.")
            return
        
        summary = {
            "forecast_horizon": self.forecast_horizon,
            "algorithms": self.algorithms,
            "best_model": self.best_model,
            "metrics": {k: {m: float(v) for m, v in metrics.items()} 
                       for k, metrics in self.metrics.items()}
        }
        
        # Add best parameters if available
        best_params = self.get_best_params()
        if best_params:
            summary["best_params"] = best_params
        
        # Save to file
        import json
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=4)
            
        print(f"Summary saved to {filename}") 