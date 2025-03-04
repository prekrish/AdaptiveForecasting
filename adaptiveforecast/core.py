"""
Core module containing the AdaptiveForecaster class.
"""

import pandas as pd
import numpy as np
import json
import warnings
from typing import List, Dict, Union, Optional, Tuple

# Local imports
from adaptiveforecast.utils.data_processing import process_input_data, prepare_data
from adaptiveforecast.utils.validation import (
    validate_inputs, process_algorithms, process_transformations, process_selected_forecasters
)
from adaptiveforecast.models import (
    get_algorithm, get_param_grid, create_multiplex_forecaster, add_algorithm_params_to_grid
)
from adaptiveforecast.transformations import get_cv_splitter
from adaptiveforecast.metrics import (
    METRIC_MAP, calculate_metrics, format_metrics_dataframe, 
    get_best_model, get_cv_results_dataframe, select_cv_display_columns
)
from adaptiveforecast.visualization import (
    plot_forecasts, create_forecast_dataframe, plot_cv_results
)
from adaptiveforecast.mlflow_utils import (
    setup_mlflow, log_parameters, log_model, log_metrics, start_run, end_run
)

# sktime imports for grid search
from sktime.forecasting.model_selection import ForecastingGridSearchCV


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
    algorithms : list or str, optional
        List of algorithms to use. If None, uses ['naive', 'arima', 'ets'].
    selected_forecasters : list or str, optional
        List of forecasters to tune. If None, uses all algorithms.
    cross_validation : bool or dict, default=False
        Cross-validation configuration.
    grid_search : bool or dict, default=False
        Grid search configuration.
    transformations : list or str, optional
        List of transformations to apply.
    seasonal_period : int, optional
        Seasonal period for seasonal transformations.
    exog_variables : list, optional
        List of exogenous variables.
    impute_method : str, default='drift'
        Method for imputing missing values.
    impute_value : float, optional
        Value to use for constant imputation.
    impute_forecaster : object, optional
        Forecaster to use for imputation.
    missing_values : list or float, optional
        Values to consider as missing.
    use_multiplex : bool, default=True
        Whether to use MultiplexForecaster for model selection.
    cv_initial_window : int, optional
        Initial window size for cross-validation.
    cv_step_length : int, default=1
        Step length for cross-validation.
    scoring : str, default="mae"
        Scoring metric for model evaluation.
    """
    
    # Valid transformation options
    VALID_TRANSFORMS = ['deseasonalize', 'detrend', 'boxcox', 'lag', 'impute']
    
    # Valid impute methods
    VALID_IMPUTE_METHODS = [
        'drift', 'linear', 'nearest', 'constant', 'mean', 'median', 
        'backfill', 'bfill', 'pad', 'ffill', 'random', 'forecaster'
    ]
    
    def __init__(
        self,
        df: Union[pd.DataFrame, pd.Series],
        target: str = None,
        date_column: str = None,
        forecast_horizon: int = 10,
        test_size: Union[float, int] = 0.2,
        algorithms: Union[List[str], str] = None,
        selected_forecasters: Optional[Union[List[str], str]] = None,
        cross_validation: Union[bool, Dict] = False,
        grid_search: Union[bool, Dict] = False,
        transformations: Union[List[str], str] = None,
        seasonal_period: Optional[int] = None,
        exog_variables: Optional[List[str]] = None,
        impute_method: str = 'drift',
        impute_value: Optional[Union[int, float]] = None,
        impute_forecaster: Optional[object] = None,
        missing_values: Optional[Union[str, int, float, list]] = None,
        use_multiplex: bool = True,
        cv_initial_window: Optional[int] = None,
        cv_step_length: int = 1,
        scoring: str = "mae"
    ):
        # Process input data
        self.df, self.target, self.date_column = process_input_data(df, target, date_column)
        
        # Store parameters
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size
        self.transformations = process_transformations(transformations)
        self.seasonal_period = seasonal_period
        self.exog_variables = exog_variables
        self.impute_method = impute_method
        self.impute_value = impute_value
        self.impute_forecaster = impute_forecaster
        self.missing_values = missing_values
        self.use_multiplex = use_multiplex
        self.cv_initial_window = cv_initial_window
        self.cv_step_length = cv_step_length
        self.scoring = scoring.lower() if scoring else "mae"
        
        # Process algorithms
        self.algorithms = process_algorithms(algorithms)
        self.selected_forecasters = process_selected_forecasters(selected_forecasters, self.algorithms)
        
        # Process cross-validation and grid search settings
        self.cross_validation = cross_validation
        self.grid_search = grid_search
        
        # Initialize containers for models and results
        self.forecasters = {}
        self.forecasts = {}
        self.metrics = {}
        self.best_model = None
        self.best_forecast = None
        self.cv_results_ = None
        
        # Set up MLflow tracking
        self.mlflow_enabled = setup_mlflow()
        
        # Validate and prepare data
        validate_inputs(
            self.df, self.target, self.algorithms, self.selected_forecasters,
            self.transformations, self.seasonal_period, self.exog_variables,
            self.impute_method, self.impute_value, self.impute_forecaster,
            self.VALID_TRANSFORMS, self.VALID_IMPUTE_METHODS
        )
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for forecasting."""
        # Split data and create forecast horizon
        self.y_train, self.y_test, self.X_train, self.X_test, self.fh = prepare_data(
            self.df, self.test_size, self.forecast_horizon, 
            self.exog_variables, self.transformations, self.seasonal_period
        )
    
    def fit(self):
        """
        Fit all specified forecasting models.
        
        Returns
        -------
        self
            Fitted forecaster instance.
        """
        # Initialize forecasters dictionary
        self.forecasters = {}
        
        # Start MLflow run
        start_run(self.mlflow_enabled)
        
        # Log parameters
        self._log_parameters()
        
        # Prepare data
        self._prepare_data()
        
        # Case 1: Only one algorithm provided
        if len(self.algorithms) == 1:
            algorithm_name = self.algorithms[0]
            print(f"Only one algorithm specified: {algorithm_name.upper()}")
            
            if self.mlflow_enabled:
                log_parameters(self.mlflow_enabled, {
                    "fit_method": "single_algorithm",
                    "algorithm": algorithm_name
                })
            
            # Get base forecaster with transformations
            forecaster = get_algorithm(
                algorithm_name, self.transformations, self.seasonal_period,
                None, self.impute_method, self.impute_value, 
                self.impute_forecaster, self.missing_values
            )
            
            if forecaster is None:
                raise ValueError(f"Could not create forecaster for {algorithm_name}.")
            
            # If grid search is provided, use GridSearchCV
            if self.grid_search:
                print(f"Using GridSearchCV to find best parameters for {algorithm_name.upper()}...")
                fitted_forecaster = self._fit_with_grid_search(algorithm_name, forecaster)
                if fitted_forecaster is not None:
                    self.forecasters[algorithm_name] = fitted_forecaster
                    print(f"{algorithm_name.upper()} model fitted successfully with GridSearchCV!")
                    
                    # Log model to MLflow
                    log_model(self.mlflow_enabled, algorithm_name, fitted_forecaster)
                else:
                    # Fallback to simple fit if grid search fails
                    print(f"GridSearchCV failed. Falling back to simple fit for {algorithm_name.upper()}...")
                    if self._fit_simple(algorithm_name, forecaster):
                        self.forecasters[algorithm_name] = forecaster
                        print(f"{algorithm_name.upper()} model fitted successfully with simple fit!")
                        
                        # Log model to MLflow
                        log_model(self.mlflow_enabled, algorithm_name, forecaster)
            else:
                # Use simple fit
                print(f"Grid search not provided. Using simple fit for {algorithm_name.upper()}...")
                if self._fit_simple(algorithm_name, forecaster):
                    self.forecasters[algorithm_name] = forecaster
                    print(f"{algorithm_name.upper()} model fitted successfully with simple fit!")
                    
                    # Log model to MLflow
                    log_model(self.mlflow_enabled, algorithm_name, forecaster)
        
        # Case 2: Multiple algorithms with multiplex=True
        elif self.use_multiplex and len(self.algorithms) > 1:
            print("Multiple algorithms with multiplex=True. Using MultiplexForecaster to find best algorithm...")
            
            if self.mlflow_enabled:
                log_parameters(self.mlflow_enabled, {
                    "fit_method": "multiplex",
                    "algorithms": self.algorithms
                })
                
            self._fit_multiplex()
        
        # Case 3: Multiple algorithms with multiplex=False
        else:
            print("Multiple algorithms with multiplex=False. Fitting each algorithm individually...")
            
            if self.mlflow_enabled:
                log_parameters(self.mlflow_enabled, {
                    "fit_method": "multiple_individual",
                    "algorithms": self.algorithms
                })
                
            self._fit_multiple_individual()
        
        # End MLflow run
        end_run(self.mlflow_enabled)
        
        return self
    
    def _log_parameters(self):
        """Log parameters to MLflow."""
        if not self.mlflow_enabled:
            return
        
        # Create a parameters dictionary
        params = {
            "forecast_horizon": self.forecast_horizon,
            "test_size": self.test_size,
            "scoring": self.scoring,
            "use_multiplex": self.use_multiplex
        }
        
        # Add transformations
        if self.transformations:
            params["transformations"] = str(self.transformations)
            
        # Add seasonal period
        if self.seasonal_period:
            params["seasonal_period"] = self.seasonal_period
            
        # Add grid search parameters
        if isinstance(self.grid_search, dict):
            for algo, algo_params in self.grid_search.items():
                for param_name, param_values in algo_params.items():
                    # Convert to string to handle non-serializable objects
                    params[f"grid_{algo}_{param_name}"] = str(param_values)
        else:
            params["grid_search"] = self.grid_search
            
        # Add cross-validation parameters
        if isinstance(self.cross_validation, dict):
            for param_name, param_value in self.cross_validation.items():
                params[f"cv_{param_name}"] = param_value
        else:
            params["cross_validation"] = self.cross_validation
        
        # Log parameters
        log_parameters(self.mlflow_enabled, params)
    
    def _fit_multiple_individual(self):
        """Fit multiple algorithms individually based on grid search settings."""
        # Track if any models were successfully fitted
        any_fitted = False
        
        # Iterate through all specified algorithms
        for algorithm_name in self.algorithms:
            print(f"Fitting {algorithm_name.upper()} model...")
            
            # Get base forecaster with transformations
            forecaster = get_algorithm(
                algorithm_name, self.transformations, self.seasonal_period,
                self.forecasters if algorithm_name == 'ensemble' else None,
                self.impute_method, self.impute_value, 
                self.impute_forecaster, self.missing_values
            )
            
            if forecaster is None:
                print(f"Warning: Could not create forecaster for {algorithm_name}. Skipping.")
                continue
            
            # Check if grid search parameters are provided for this algorithm
            has_grid_params = (isinstance(self.grid_search, dict) and 
                              algorithm_name in self.grid_search and 
                              self.grid_search[algorithm_name])
            
            # If grid search is enabled and parameters are provided for this algorithm, use GridSearchCV
            if self.grid_search and has_grid_params:
                print(f"  Grid search parameters provided for {algorithm_name}. Using GridSearchCV...")
                try:
                    fitted_forecaster = self._fit_with_grid_search(algorithm_name, forecaster)
                    if fitted_forecaster is not None:
                        self.forecasters[algorithm_name] = fitted_forecaster
                        any_fitted = True
                        print(f"  {algorithm_name.upper()} model fitted successfully with GridSearchCV!")
                        
                        # Log model to MLflow
                        log_model(self.mlflow_enabled, algorithm_name, fitted_forecaster)
                    else:
                        # Fallback to simple fit if grid search fails
                        print(f"  GridSearchCV failed. Falling back to simple fit for {algorithm_name}...")
                        if self._fit_simple(algorithm_name, forecaster):
                            self.forecasters[algorithm_name] = forecaster
                            any_fitted = True
                            print(f"  {algorithm_name.upper()} model fitted successfully with simple fit!")
                            
                            # Log model to MLflow
                            log_model(self.mlflow_enabled, algorithm_name, forecaster)
                except Exception as e:
                    print(f"  Error during grid search: {str(e)}")
                    print(f"  Falling back to simple fit for {algorithm_name}...")
                    if self._fit_simple(algorithm_name, forecaster):
                        self.forecasters[algorithm_name] = forecaster
                        any_fitted = True
                        print(f"  {algorithm_name.upper()} model fitted successfully with simple fit!")
                        
                        # Log model to MLflow
                        log_model(self.mlflow_enabled, algorithm_name, forecaster)
            else:
                # Use simple fit
                if self.grid_search and not has_grid_params:
                    print(f"  No grid search parameters provided for {algorithm_name}. Using simple fit...")
                else:
                    print(f"  Grid search disabled. Using simple fit for {algorithm_name}...")
                    
                if self._fit_simple(algorithm_name, forecaster):
                    self.forecasters[algorithm_name] = forecaster
                    any_fitted = True
                    print(f"  {algorithm_name.upper()} model fitted successfully with simple fit!")
                    
                    # Log model to MLflow
                    log_model(self.mlflow_enabled, algorithm_name, forecaster)
        
        if not any_fitted:
            print("Warning: No models were successfully fitted.")
        
        return self
    
    def _fit_simple(self, algorithm_name, forecaster):
        """
        Fit a forecaster without grid search.
        
        Parameters
        ----------
        algorithm_name : str
            Algorithm name.
        forecaster : object
            Forecaster instance.
            
        Returns
        -------
        bool
            Whether the fit was successful.
        """
        try:
            print(f"  Fitting {algorithm_name} with default parameters...")
            if self.X_train is None:
                forecaster.fit(self.y_train)
            else:
                forecaster.fit(self.y_train, X=self.X_train)
            return True
        except Exception as e:
            print(f"  Error fitting {algorithm_name}: {str(e)}")
            return False
    
    def _fit_with_grid_search(self, algorithm_name, forecaster):
        """
        Perform grid search for a specific algorithm.
        
        Parameters
        ----------
        algorithm_name : str
            Algorithm name.
        forecaster : object
            Forecaster instance.
            
        Returns
        -------
        object
            Fitted forecaster with optimal parameters.
        """
        print(f"  Performing grid search for {algorithm_name}...")
        
        # Initialize param_grid to avoid UnboundLocalError
        param_grid = {}
        
        try:
            # Get parameter grid
            param_grid = get_param_grid(algorithm_name, self.grid_search)
            
            # If no param_grid is provided, just fit without grid search
            if not param_grid:
                print(f"  No parameter grid provided for {algorithm_name}. Fitting with default parameters.")
                if self._fit_simple(algorithm_name, forecaster):
                    return forecaster
                return None
            
            # Configure cross-validation if specified
            if self.cross_validation:
                cv_params = self.cross_validation if isinstance(self.cross_validation, dict) else {}
                cv = get_cv_splitter(
                    cv_params, self.cv_initial_window, 
                    self.cv_step_length, self.forecast_horizon, 
                    self.y_train
                )
            else:
                cv = get_cv_splitter(
                    None, self.cv_initial_window, 
                    self.cv_step_length, self.forecast_horizon, 
                    self.y_train
                )
            
            # Get the appropriate metric object
            if self.scoring in METRIC_MAP:
                scoring_metric = METRIC_MAP[self.scoring]
            else:
                print(f"  Warning: Unknown scoring metric '{self.scoring}'. Defaulting to 'mae'.")
                scoring_metric = METRIC_MAP["mae"]
            
            # Create and fit grid search
            grid_search = ForecastingGridSearchCV(
                forecaster=forecaster,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring_metric,
                error_score='raise'
            )
            
            if self.X_train is None:
                grid_search.fit(self.y_train)
            else:
                grid_search.fit(self.y_train, X=self.X_train)
            
            # Store the best forecaster
            forecaster = grid_search.best_forecaster_
            
            # Store CV results
            self._store_cv_results(grid_search)
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Best score: {grid_search.best_score_:.4f}")
            
            return forecaster
            
        except Exception as e:
            print(f"  Error during grid search: {str(e)}")
            print("  Falling back to default parameters.")
            
            # Try with different metrics if the error is related to scoring
            if "scoring" in str(e).lower():
                return self._try_alternative_metrics(algorithm_name, forecaster, param_grid)
            else:
                # For other errors, fit with default parameters
                if self._fit_simple(algorithm_name, forecaster):
                    return forecaster
                return None
    
    def _try_alternative_metrics(self, algorithm_name, forecaster, param_grid):
        """
        Try alternative metrics if the original scoring fails.
        
        Parameters
        ----------
        algorithm_name : str
            Algorithm name.
        forecaster : object
            Forecaster instance.
        param_grid : dict
            Parameter grid.
            
        Returns
        -------
        object
            Fitted forecaster with optimal parameters.
        """
        # If param_grid is empty, just fit without grid search
        if not param_grid:
            print(f"  No parameter grid provided for {algorithm_name}. Fitting with default parameters.")
            if self._fit_simple(algorithm_name, forecaster):
                return forecaster
            return None
        
        # Configure cross-validation
        if self.cross_validation:
            cv_params = self.cross_validation if isinstance(self.cross_validation, dict) else {}
            cv = get_cv_splitter(
                cv_params, self.cv_initial_window, 
                self.cv_step_length, self.forecast_horizon, 
                self.y_train
            )
        else:
            cv = get_cv_splitter(
                None, self.cv_initial_window, 
                self.cv_step_length, self.forecast_horizon, 
                self.y_train
            )
            
        for metric_name, metric_obj in METRIC_MAP.items():
            if metric_name == self.scoring:
                continue  # Skip the one we already tried
                
            try:
                print(f"  Attempting with {metric_name}...")
                grid_search = ForecastingGridSearchCV(
                    forecaster=forecaster,
                    param_grid=param_grid,
                    cv=cv,
                    scoring=metric_obj
                )
                
                if self.X_train is None:
                    grid_search.fit(self.y_train)
                else:
                    grid_search.fit(self.y_train, X=self.X_train)
                    
                # If we get here, the fit was successful
                print(f"  Successfully fitted with {metric_name}")
                
                # Store the best forecaster
                forecaster = grid_search.best_forecaster_
                
                # Store CV results
                self._store_cv_results(grid_search)
                    
                print(f"  Best parameters: {grid_search.best_params_}")
                print(f"  Best score: {grid_search.best_score_:.4f}")
                return forecaster
            except Exception as e2:
                print(f"  Failed with {metric_name}: {str(e2)}")
        
        # If all metrics fail, fit with default parameters
        print("  All metrics failed. Fitting with default parameters.")
        if self._fit_simple(algorithm_name, forecaster):
            return forecaster
        return None
        
    def _fit_multiplex(self):
        """
        Fit models using MultiplexForecaster for efficient model selection.
        
        Returns
        -------
        self
            Fitted forecaster instance.
        """
        try:
            # Create multiplex forecaster
            multiplex, forecaster_list = create_multiplex_forecaster(
                self.algorithms, 
                self.selected_forecasters,
                self.transformations,
                self.seasonal_period,
                self.impute_method,
                self.impute_value,
                self.impute_forecaster,
                self.missing_values
            )
            
            if not forecaster_list:
                print("Warning: No valid forecasters to fit.")
                return self
            
            # Configure cross-validation
            cv_params = self.cross_validation if isinstance(self.cross_validation, dict) else {}
            cv = get_cv_splitter(
                cv_params, self.cv_initial_window, 
                self.cv_step_length, self.forecast_horizon, 
                self.y_train
            )
            
            # Create parameter grid for MultiplexForecaster
            # Only include the selected forecasters in the grid search
            param_grid = {"selected_forecaster": [name for name, _ in forecaster_list 
                                                if name in self.selected_forecasters]}
            
            if not param_grid["selected_forecaster"]:
                print("Warning: No valid selected forecasters for grid search. Using all available forecasters.")
                param_grid["selected_forecaster"] = [name for name, _ in forecaster_list]
            
            print(f"Forecasters to be tuned: {param_grid['selected_forecaster']}")
            
            # Add algorithm-specific parameters
            param_grid = add_algorithm_params_to_grid(param_grid, param_grid["selected_forecaster"], self.grid_search)
            
            print("Performing grid search across selected algorithms...")
            print(f"Parameter grid: {param_grid}")
            
            # Get the appropriate metric object
            if self.scoring in METRIC_MAP:
                scoring_metric = METRIC_MAP[self.scoring]
            else:
                print(f"Warning: Unknown scoring metric '{self.scoring}'. Defaulting to 'mae'.")
                scoring_metric = METRIC_MAP["mae"]
            
            # Create and fit grid search
            try:
                grid_search = self._create_and_fit_grid_search(multiplex, param_grid, cv, scoring_metric)
            except Exception as e:
                print(f"Error during grid search: {str(e)}")
                print("Trying with a different scoring metric...")
                
                # Try with different metrics
                for metric_name, metric_obj in METRIC_MAP.items():
                    if metric_name == self.scoring:
                        continue  # Skip the one we already tried
                        
                    try:
                        print(f"Attempting with {metric_name}...")
                        grid_search = self._create_and_fit_grid_search(multiplex, param_grid, cv, metric_obj)
                        
                        # If we get here, the fit was successful
                        print(f"Successfully fitted with {metric_name}")
                        break
                    except Exception as e2:
                        print(f"Failed with {metric_name}: {str(e2)}")
                else:
                    # If all metrics fail, raise the original error
                    raise e
            
            # Store CV results
            self.cv_results_ = grid_search.cv_results_
            
            # Get the best forecaster
            best_params = grid_search.best_params_
            best_algorithm = best_params.get("selected_forecaster")
            
            print(f"Best algorithm selected: {best_algorithm.upper()}")
            print(f"Best parameters: {best_params}")
            print(f"Best score: {grid_search.best_score_:.4f}")
            
            # Store the fitted forecaster
            self.forecasters["multiplex"] = grid_search
            
            # Log model to MLflow
            log_model(self.mlflow_enabled, "multiplex", grid_search)
            
            # Also store the best individual forecaster for reference
            if best_algorithm:
                self._store_best_individual_forecaster(best_algorithm, best_params)
                
            print("MultiplexForecaster fitted successfully!")
            return self
            
        except Exception as e:
            print(f"Error during multiplex fitting: {str(e)}")
            print("Falling back to fitting algorithms individually...")
            return self._fit_multiple_individual()
    
    def _create_and_fit_grid_search(self, forecaster, param_grid, cv, scoring_metric):
        """
        Create and fit a grid search CV object.
        
        Parameters
        ----------
        forecaster : object
            Forecaster instance.
        param_grid : dict
            Parameter grid.
        cv : object
            CV splitter.
        scoring_metric : object
            Scoring metric.
            
        Returns
        -------
        object
            Fitted grid search object.
        """
        grid_search = ForecastingGridSearchCV(
            forecaster=forecaster,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring_metric,
            n_jobs=-1,
            error_score='raise'
        )
        
        if self.X_train is None:
            grid_search.fit(self.y_train)
        else:
            grid_search.fit(self.y_train, X=self.X_train)
            
        return grid_search
    
    def _store_best_individual_forecaster(self, best_algorithm, best_params):
        """
        Store the best individual forecaster with its optimal parameters.
        
        Parameters
        ----------
        best_algorithm : str
            Algorithm name.
        best_params : dict
            Best parameters.
        """
        # Get the best individual forecaster with its optimal parameters
        best_forecaster = get_algorithm(
            best_algorithm, 
            self.transformations, 
            self.seasonal_period,
            None,  # No forecasters for ensemble
            self.impute_method, 
            self.impute_value, 
            self.impute_forecaster, 
            self.missing_values
        )
        
        if best_forecaster is None:
            print(f"Warning: Could not create forecaster for {best_algorithm}.")
            return
        
        # Apply the best parameters found by grid search
        best_algo_params = {
            k.replace(f"{best_algorithm}__", ""): v 
            for k, v in best_params.items() 
            if k.startswith(f"{best_algorithm}__")
        }
        
        if best_algo_params:
            try:
                best_forecaster.set_params(**best_algo_params)
            except Exception as e:
                print(f"Warning: Could not set parameters for {best_algorithm}: {str(e)}")
            
        # Fit the best individual forecaster
        if self._fit_simple(best_algorithm, best_forecaster):
            # Store the forecaster
            self.forecasters[best_algorithm] = best_forecaster
            
            # Log model to MLflow
            log_model(self.mlflow_enabled, best_algorithm, best_forecaster)
    
    
    def predict(self):
        """
        Generate forecasts for all fitted models.
        
        Returns
        -------
        self
            Forecaster instance with predictions.
        """
        # Check if any forecasters have been fitted
        if not self.forecasters:
            print("No forecasters have been fitted. Run fit() first.")
            return self
        
        # Start MLflow run for predictions
        start_run(self.mlflow_enabled, "AdaptiveForecaster_Predict")
        
        # If using MultiplexForecaster, predict with it first
        if "multiplex" in self.forecasters:
            print("Generating forecast with MultiplexForecaster...")
            self._predict_with_forecaster("multiplex")
            
            # Get the best algorithm name
            best_algo = self.get_best_algorithm()
            if best_algo and best_algo in self.forecasters:
                print(f"Best algorithm selected by MultiplexForecaster: {best_algo.upper()}")
                
                # Log to MLflow
                if self.mlflow_enabled:
                    log_parameters(self.mlflow_enabled, {"best_algorithm": best_algo})
        
        # Generate forecasts for all other fitted models
        for name in self.forecasters:
            if name == "multiplex":
                continue  # Already predicted
            
            print(f"Generating forecast for {name.upper()}...")
            self._predict_with_forecaster(name)
        
        # Determine best model based on RMSE if metrics are available
        if self.metrics:
            best_model_name = get_best_model(self.metrics)
            self.best_model = best_model_name
            self.best_forecast = self.forecasts[best_model_name]
            
            # Log to MLflow
            if self.mlflow_enabled:
                log_parameters(self.mlflow_enabled, {"best_model_by_rmse": best_model_name})
                
                # Log best metrics
                best_metrics = self.metrics[best_model_name]
                log_metrics(self.mlflow_enabled, "best", best_metrics)
        
        # Check if any forecasts were generated
        if not self.forecasts:
            print("No forecasts could be generated. Check for errors during prediction.")
            return self
        
        print("All forecasts generated successfully!")
        
        # Print the forecasts
        self.print_forecasts()
        
        # End MLflow run
        end_run(self.mlflow_enabled)
        
        return self

    def _predict_with_forecaster(self, name):
        """
        Generate forecast and calculate metrics for a specific forecaster.
        
        Parameters
        ----------
        name : str
            Name of the forecaster.
            
        Returns
        -------
        bool
            Whether the prediction was successful.
        """
        try:
            forecaster = self.forecasters[name]
            
            # Generate forecast
            if self.X_test is None:
                forecast = forecaster.predict(fh=self.fh)
            else:
                forecast = forecaster.predict(fh=self.fh, X=self.X_test)
                
            self.forecasts[name] = forecast
            
            # Calculate metrics if test data is available
            if self.y_test is not None and len(self.y_test) >= self.forecast_horizon:
                try:
                    metrics = calculate_metrics(self.y_test, forecast, self.forecast_horizon)
                    if metrics:
                        self.metrics[name] = metrics
                        
                        # Log metrics to MLflow
                        log_metrics(self.mlflow_enabled, name, metrics)
                except Exception as e:
                    print(f"  Warning: Could not calculate metrics for {name}: {str(e)}")
                    # Still keep the forecast even if metrics calculation fails
            
            return True
        except Exception as e:
            print(f"  Error predicting with {name}: {str(e)}")
            return False

    def get_best_algorithm(self):
        """
        Get the best algorithm selected by MultiplexForecaster.
        
        Returns
        -------
        str
            Name of the best algorithm.
        """
        if "multiplex" in self.forecasters:
            multiplex = self.forecasters["multiplex"]
            if hasattr(multiplex, "best_params_"):
                return multiplex.best_params_.get("selected_forecaster", None)
        return self.best_model

    def get_best_params(self):
        """
        Get the best parameters found during grid search.
        
        Returns
        -------
        dict
            Best parameters.
        """
        if "multiplex" in self.forecasters:
            multiplex = self.forecasters["multiplex"]
            if hasattr(multiplex, "best_params_"):
                return multiplex.best_params_
        return None

    def get_cv_results(self):
        """
        Get cross-validation results from grid search.
        
        Returns
        -------
        pandas.DataFrame
            Formatted CV results.
        """
        return get_cv_results_dataframe(self.cv_results_)

    def print_cv_results(self, n_best=5):
        """
        Print the best cross-validation results.
        
        Parameters
        ----------
        n_best : int, default=5
            Number of best results to display.
        """
        if self.cv_results_ is None:
            print("No cross-validation results available. Run fit() with grid_search=True first.")
            return
        
        results_df = self.get_cv_results()
        
        # Check if results_df is a string (error message)
        if isinstance(results_df, str):
            print(results_df)
            return
        
        # Get formatted results
        top_results = select_cv_display_columns(results_df, n_best)
        
        if isinstance(top_results, str):
            print(top_results)
            return
        
        # Print the best n results
        print("\n" + "="*50)
        print(f"CROSS-VALIDATION RESULTS (TOP {n_best})")
        print("="*50)
        print(top_results)
        print("="*50)

    def _store_cv_results(self, grid_search):
        """
        Store cross-validation results from grid search.
        
        Parameters
        ----------
        grid_search : object
            Fitted grid search object.
        """
        # Initialize cv_results_ if it doesn't exist
        if not hasattr(self, 'cv_results_') or self.cv_results_ is None:
            self.cv_results_ = grid_search.cv_results_
        else:
            # Append results from this algorithm
            for key, values in grid_search.cv_results_.items():
                if key in self.cv_results_:
                    if isinstance(values, np.ndarray) and isinstance(self.cv_results_[key], np.ndarray):
                        self.cv_results_[key] = np.append(self.cv_results_[key], values)
                    elif isinstance(values, list) and isinstance(self.cv_results_[key], list):
                        self.cv_results_[key].extend(values)
                    else:
                        # For non-array values, just overwrite
                        self.cv_results_[key] = values
                else:
                    self.cv_results_[key] = values

    def print_forecasts(self):
        """
        Print forecasts for all models.
        """
        if not self.forecasts:
            print("No forecasts available. Run predict() first.")
            return
            
        # Create a DataFrame for better display
        forecast_df = create_forecast_dataframe(self.y_test, self.forecasts, self.forecast_horizon)
        
        if forecast_df is None or forecast_df.empty:
            print("No forecast data to display.")
            return
            
        # Print the forecasts
        print("\n" + "="*50)
        print("FORECASTS")
        print("="*50)
        print(forecast_df)
        print("="*50)
        
        # Print metrics if available
        if self.metrics:
            print("\n" + "="*50)
            print("PERFORMANCE METRICS")
            print("="*50)
            
            # Create a DataFrame for metrics
            metrics_df = format_metrics_dataframe(self.metrics)
            
            if metrics_df is not None:
                print(metrics_df)
            else:
                print("No metrics available.")
                
            print("="*50)
            
            # Print best model
            if self.best_model:
                print(f"\nBest model based on RMSE: {self.best_model.upper()}")

    def plot_forecasts(self, figsize=(12, 6), include_history=True, models=None):
        """
        Plot forecasts for all or selected models.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 6)
            Figure size.
        include_history : bool, default=True
            Whether to include historical data in the plot.
        models : list, default=None
            List of model names to plot. If None, all models are plotted.
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object.
        """
        if not self.forecasts:
            print("No forecasts available. Run predict() first.")
            return None
        
        return plot_forecasts(
            self.y_train, self.y_test, self.forecasts, self.forecast_horizon,
            self.best_model, figsize, include_history, models, self.target
        )

    def get_summary(self):
        """
        Get a summary of the forecasting results.
        
        Returns
        -------
        dict
            A dictionary containing summary information about the forecasting results.
        """
        summary = {
            "algorithms": self.algorithms,
            "selected_forecasters": self.selected_forecasters,
            "forecast_horizon": self.forecast_horizon,
            "transformations": self.transformations,
            "best_model": self.best_model if hasattr(self, 'best_model') else None,
        }
        
        # Add metrics if available
        if hasattr(self, 'metrics') and self.metrics:
            summary["metrics"] = self.metrics
        
        # Add forecasts if available
        if hasattr(self, 'forecasts') and self.forecasts:
            # Convert forecasts to dictionary of lists for easier serialization
            forecast_dict = {}
            for name, forecast in self.forecasts.items():
                if isinstance(forecast, pd.DataFrame):
                    forecast_dict[name] = forecast.iloc[:, 0].tolist()
                else:  # It's a Series
                    forecast_dict[name] = forecast.tolist()
            summary["forecasts"] = forecast_dict
            
            # Add forecast dates if available
            if len(self.forecasts) > 0:
                first_forecast = next(iter(self.forecasts.values()))
                if hasattr(first_forecast, 'index'):
                    dates = [str(d) for d in first_forecast.index]
                    summary["forecast_dates"] = dates
        
        # Add CV results if available
        if hasattr(self, 'cv_results_') and self.cv_results_ is not None:
            # Convert CV results to a simpler format
            cv_summary = {}
            for key, value in self.cv_results_.items():
                if isinstance(value, np.ndarray):
                    cv_summary[key] = value.tolist()
                else:
                    cv_summary[key] = value
            summary["cv_results"] = cv_summary
        
        # Add best parameters if available
        if "multiplex" in getattr(self, 'forecasters', {}):
            multiplex = self.forecasters["multiplex"]
            if hasattr(multiplex, "best_params_"):
                summary["best_params"] = multiplex.best_params_
        
        return summary

    def save_summary(self, filename="forecast_summary.json"):
        """
        Save the forecast summary to a JSON file.
        
        Parameters
        ----------
        filename : str, default="forecast_summary.json"
            The name of the file to save the summary to.
        
        Returns
        -------
        bool
            True if the summary was successfully saved, False otherwise.
        """
        try:
            summary = self.get_summary()
            
            # Convert numpy types to Python native types
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                else:
                    return obj
            
            summary = convert_numpy(summary)
            
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=4)
            
            print(f"Summary saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving summary: {str(e)}")
            return False