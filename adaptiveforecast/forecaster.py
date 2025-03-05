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
    
    def __init__(self, df, target=None, date_column=None, forecast_horizon=10, 
                 test_size=0.2, algorithms=None, transformations=None, 
                 seasonal_period=None, grid_search=False, cross_validation=False, 
                 scoring="mae"):
        # Initialize with default algorithms if None
        self.algorithms = ['naive', 'arima', 'ets'] if algorithms is None else \
                         [algorithms] if isinstance(algorithms, str) else algorithms
        
        # Process transformations once
        self.transformations = self._process_transformations(transformations)
        
        # Store other parameters
        self.df = df
        self.target = target
        self.date_column = date_column
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size
        self.seasonal_period = seasonal_period
        self.grid_search = grid_search
        self.grid_search_params = grid_search if isinstance(grid_search, dict) else {}
        self.cross_validation = cross_validation
        self.cv_params = cross_validation if isinstance(cross_validation, dict) else {}
        self.scoring = scoring
        
        # Initialize result containers
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
        
    def _process_transformations(self, transformations):
        """Helper method to process transformations configuration."""
        if transformations is None:
            return {}
        elif isinstance(transformations, str):
            return {algo: [transformations] for algo in self.algorithms}
        elif isinstance(transformations, list):
            return {algo: transformations for algo in self.algorithms}
        elif isinstance(transformations, dict):
            return transformations
        return {}

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
        for name in self.algorithms:
            try:
                self._fit_single_model(name)
            except Exception as e:
                print(f"Error fitting {name}: {str(e)}")
        return self
    
    def _fit_single_model(self, name):
        """Helper method to fit a single model."""
        print(f"Fitting {name}...")
        algo_transforms = self.transformations.get(name, None)
        algorithm = get_algorithm(name, algo_transforms, self.seasonal_period)
        
        if self.grid_search:
            self._fit_with_grid_search(name, algorithm)
        else:
            algorithm.fit(self.y_train)
            self.forecasters[name] = algorithm
            
        print(f"  {name.capitalize()} fitted successfully.")

    def _fit_with_grid_search(self, name, algorithm):
        """Helper method to fit model with grid search."""
        param_grid = get_param_grid(name, self.grid_search_params)
        if not param_grid:
            algorithm.fit(self.y_train)
            self.forecasters[name] = algorithm
            return
            
        cv = get_cv_splitter(self.cv_params, None, self.forecast_horizon, self.y_train)
        
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
            grid_search.fit(self.y_train)
            self.forecasters[name] = grid_search
            self.cv_results[name] = grid_search.cv_results_
            print(f"  Best parameters: {grid_search.best_params_}")
        except Exception as e:
            print(f"  Grid search failed, falling back to direct fit: {str(e)}")
            algorithm.fit(self.y_train)
            self.forecasters[name] = algorithm

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