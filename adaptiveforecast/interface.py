from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import warnings
import matplotlib.pyplot as plt
import json

# Suppress specific warnings
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", message="The user-specified parameters provided alongside auto=True in AutoETS may not be respected")

# Import performance metrics
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError, 
    MeanAbsoluteError, 
    MeanSquaredError
)

# sktime imports
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter, 
    SlidingWindowSplitter,
    ForecastingGridSearchCV
)
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.lag import Lag
from sktime.transformations.series.impute import Imputer

# Algorithm imports
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


class AdaptiveForecaster:
    """
    A user-friendly interface for time series forecasting using sktime.
    
    This class provides a simple interface for users to perform time series forecasting
    by abstracting away the complexities of the underlying algorithms and processes.
    """
    
    # Mapping of scoring strings to metric objects
    METRIC_MAP = {
        "mae": MeanAbsoluteError(),
        "mse": MeanSquaredError(),
        "mape": MeanAbsolutePercentageError(),
        "rmse": MeanSquaredError(square_root=True)
    }
    
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
        self.df = self._process_input_data(df, target, date_column)
        
        # Store parameters
        self.target = target
        self.date_column = date_column
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size
        self.transformations = self._process_transformations(transformations)
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
        self.algorithms = self._process_algorithms(algorithms)
        self.selected_forecasters = self._process_selected_forecasters(selected_forecasters)
        
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
        
        # Validate and prepare data
        self._validate_inputs()
        self._prepare_data()
    
    def _process_input_data(self, df, target, date_column):
        """Process input data and handle Series/DataFrame conversion."""
        if isinstance(df, pd.Series):
            # For Series input, just use it directly
            # Store the name for reference but keep as Series
            self.target = df.name if df.name is not None else 'y'
            return df
        else:
            # For DataFrame input, require explicit target and date_column
            if target is None:
                raise ValueError("When providing a DataFrame, you must specify a target column name.")
                    
            self.target = target
            
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
            return df[target]
    
    def _process_algorithms(self, algorithms):
        """Process and validate algorithms parameter."""
        if algorithms is None:
            return ['naive', 'arima', 'ets']
        elif isinstance(algorithms, str):
            return [algorithms]
        else:
            return algorithms
    
    def _process_transformations(self, transformations):
        """Process transformations parameter."""
        if isinstance(transformations, str):
            return [transformations]
        return transformations
    
    def _process_selected_forecasters(self, selected_forecasters):
        """Process selected_forecasters parameter."""
        if selected_forecasters is None:
            # If not specified, use all algorithms
            return self.algorithms.copy()
        elif isinstance(selected_forecasters, str):
            return [selected_forecasters]
        else:
            return selected_forecasters
    
    def _validate_inputs(self):
        """Validate user inputs."""
        # Check if target exists in dataframe (only for DataFrame inputs)
        if isinstance(self.df, pd.DataFrame) and self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataframe.")
        
        # Validate selected_forecasters is a subset of algorithms
        invalid_forecasters = [f for f in self.selected_forecasters if f not in self.algorithms]
        if invalid_forecasters:
            raise ValueError(f"Selected forecasters {invalid_forecasters} are not in the algorithms list: {self.algorithms}")
        
        # Validate transformations
        if self.transformations is not None:
            for transform in self.transformations:
                if transform.lower() not in self.VALID_TRANSFORMS:
                    raise ValueError(f"Transformation '{transform}' not supported. Choose from {self.VALID_TRANSFORMS}")
                    
            # Check if seasonal_period is provided for seasonal transformations
            if 'deseasonalize' in self.transformations and self.seasonal_period is None:
                raise ValueError("seasonal_period must be provided when using 'deseasonalize' transformation.")
            
        # Validate impute method if impute transformation is used
        if self.transformations is not None and 'impute' in self.transformations:
            if self.impute_method not in self.VALID_IMPUTE_METHODS:
                raise ValueError(f"Impute method '{self.impute_method}' not supported. Choose from {self.VALID_IMPUTE_METHODS}")
                
            # Check if required parameters are provided for specific impute methods
            if self.impute_method == 'constant' and self.impute_value is None:
                raise ValueError("impute_value must be provided when impute_method='constant'")
                
            if self.impute_method == 'forecaster' and self.impute_forecaster is None:
                raise ValueError("impute_forecaster must be provided when impute_method='forecaster'")
            
        # Validate exogenous variables
        if self.exog_variables is not None:
            if isinstance(self.df, pd.Series):
                raise ValueError("Exogenous variables are not supported when using Series input. Please provide a DataFrame.")
            
            for var in self.exog_variables:
                if var not in self.df.columns:
                    raise ValueError(f"Exogenous variable '{var}' not found in dataframe.")
    
    def _prepare_data(self):
        """Prepare data for forecasting."""
        # Extract target and exogenous variables
        y = self.df
        
        # Ensure the index has a frequency if using seasonal transformations
        if (self.transformations is not None and 
            ('deseasonalize' in self.transformations or 'detrend' in self.transformations)):
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
        if self.exog_variables is not None:
            raise ValueError("Exogenous variables are not supported when using Series input. Please provide a DataFrame.")
        
        # Create train-test split
        self.y_train, self.y_test = temporal_train_test_split(y, test_size=self.test_size)
        self.X_train, self.X_test = None, None
                
        # Create forecasting horizon
        self.fh = ForecastingHorizon(
            np.arange(1, self.forecast_horizon + 1), is_relative=True
        )
    
    def _get_cv_splitter(self, cv_params=None):
        """Get cross-validation splitter based on parameters."""
        if cv_params is None:
            cv_params = {}
        
        method = cv_params.get('method', 'expanding')
        
        # Determine initial window size if not provided
        if self.cv_initial_window is None:
            # Default to 70% of training data
            initial_window = int(len(self.y_train) * 0.7)
        else:
            initial_window = self.cv_initial_window
        
        # Ensure initial window is at least 2 observations
        initial_window = max(initial_window, 2)
        
        # Get step length
        step_length = self.cv_step_length
        
        # Create forecasting horizon for CV
        fh = np.arange(1, self.forecast_horizon + 1)
        
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
    
    def _get_algorithm(self, name, with_transformations=True):
        """Get a forecasting algorithm by name."""
        algorithm = None
        
        if name.lower() == 'naive':
            algorithm = NaiveForecaster(strategy="mean")
        elif name.lower() == 'theta':
            algorithm = ThetaForecaster()
        elif name.lower() == 'arima':
            algorithm = AutoARIMA(
                start_p=1, start_q=1, max_p=3, max_q=3, 
                seasonal=self.seasonal_period is not None,
                sp=self.seasonal_period,
                d=1, max_d=2,
                suppress_warnings=True
            )
        elif name.lower() == 'ets':
            algorithm = AutoETS(
                auto=True, 
                seasonal=self.seasonal_period is not None,
                sp=self.seasonal_period
            )
        elif name.lower() == 'exp_smoothing':
            algorithm = ExponentialSmoothing(
                seasonal=self.seasonal_period is not None,
                sp=self.seasonal_period
            )
        elif name.lower() == 'ensemble':
            # Only create ensemble if we have other algorithms
            if len(self.forecasters) < 2:
                raise ValueError("Ensemble requires at least 2 other algorithms to be trained first.")
            
            forecaster_list = list(self.forecasters.values())
            algorithm = EnsembleForecaster(forecasters=forecaster_list)
            
        # Apply transformations if specified
        if with_transformations and self.transformations is not None and algorithm is not None:
            algorithm = self._apply_transformations(algorithm)
            
        return algorithm
    
    def _apply_transformations(self, forecaster):
        """Apply specified transformations to a forecaster."""
        transformers = []
        
        if self.transformations is None:
            return forecaster
        
        for transform in self.transformations:
            if transform.lower() == 'deseasonalize':
                transformers.append(
                    ("deseasonalize", Deseasonalizer(sp=self.seasonal_period))
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
                if self.impute_method == 'constant':
                    imputer = Imputer(
                        method=self.impute_method,
                        value=self.impute_value,
                        missing_values=self.missing_values
                    )
                elif self.impute_method == 'forecaster':
                    imputer = Imputer(
                        method=self.impute_method,
                        forecaster=self.impute_forecaster,
                        missing_values=self.missing_values
                    )
                elif self.impute_method == 'random':
                    imputer = Imputer(
                        method=self.impute_method,
                        random_state=42,  # Using a fixed seed for reproducibility
                        missing_values=self.missing_values
                    )
                else:
                    imputer = Imputer(
                        method=self.impute_method,
                        missing_values=self.missing_values
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
    
    def _get_param_grid(self, algorithm_name):
        """Get default parameter grid for an algorithm."""
        if isinstance(self.grid_search, dict) and algorithm_name in self.grid_search:
            return self.grid_search[algorithm_name]
            
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
    
    def fit(self):
        """Fit all specified forecasting models."""
        # Initialize forecasters dictionary
        self.forecasters = {}
        
        # Initialize MLflow tracking
        self._setup_mlflow()
        
        # Start MLflow run
        try:
            if self.mlflow_enabled:
                import mlflow
                from datetime import datetime
                mlflow.start_run(run_name=f"AdaptiveForecaster_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                # Log parameters
                self._log_parameters()
        except Exception as e:
            print(f"Warning: Could not start MLflow run: {str(e)}")
        
        # Prepare data
        self._prepare_data()
        
        # Case 1: Only one algorithm provided
        if len(self.algorithms) == 1:
            algorithm_name = self.algorithms[0]
            print(f"Only one algorithm specified: {algorithm_name.upper()}")
            
            if self.mlflow_enabled:
                try:
                    import mlflow
                    mlflow.log_param("fit_method", "single_algorithm")
                    mlflow.log_param("algorithm", algorithm_name)
                except Exception as e:
                    print(f"Warning: Could not log parameters to MLflow: {str(e)}")
            
            # Get base forecaster with transformations
            forecaster = self._get_algorithm(algorithm_name)
            
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
                    if self.mlflow_enabled:
                        self._log_model(algorithm_name, fitted_forecaster)
                else:
                    # Fallback to simple fit if grid search fails
                    print(f"GridSearchCV failed. Falling back to simple fit for {algorithm_name.upper()}...")
                    if self._fit_simple(algorithm_name, forecaster):
                        self.forecasters[algorithm_name] = forecaster
                        print(f"{algorithm_name.upper()} model fitted successfully with simple fit!")
                        
                        # Log model to MLflow
                        if self.mlflow_enabled:
                            self._log_model(algorithm_name, forecaster)
            else:
                # Use simple fit
                print(f"Grid search not provided. Using simple fit for {algorithm_name.upper()}...")
                if self._fit_simple(algorithm_name, forecaster):
                    self.forecasters[algorithm_name] = forecaster
                    print(f"{algorithm_name.upper()} model fitted successfully with simple fit!")
                    
                    # Log model to MLflow
                    if self.mlflow_enabled:
                        self._log_model(algorithm_name, forecaster)
        
        # Case 2: Multiple algorithms with multiplex=True
        elif self.use_multiplex and len(self.algorithms) > 1:
            print("Multiple algorithms with multiplex=True. Using MultiplexForecaster to find best algorithm...")
            
            if self.mlflow_enabled:
                try:
                    import mlflow
                    mlflow.log_param("fit_method", "multiplex")
                    mlflow.log_param("algorithms", self.algorithms)
                except Exception as e:
                    print(f"Warning: Could not log parameters to MLflow: {str(e)}")
                
            return self._fit_multiplex()
        
        # Case 3: Multiple algorithms with multiplex=False
        else:
            print("Multiple algorithms with multiplex=False. Fitting each algorithm individually...")
            
            if self.mlflow_enabled:
                try:
                    import mlflow
                    mlflow.log_param("fit_method", "multiple_individual")
                    mlflow.log_param("algorithms", self.algorithms)
                except Exception as e:
                    print(f"Warning: Could not log parameters to MLflow: {str(e)}")
                
            return self._fit_multiple_individual()
        
        # End MLflow run
        if self.mlflow_enabled:
            try:
                import mlflow
                mlflow.end_run()
            except Exception as e:
                print(f"Warning: Could not end MLflow run: {str(e)}")
        
        return self
    
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        try:
            import mlflow
            
            # Set tracking URI
            mlflow.set_tracking_uri("http://localhost:8080")
            
            # Create experiment if it doesn't exist
            experiment_name = "AdaptiveForecaster"
            
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
                
                mlflow.set_experiment(experiment_name)
                self.mlflow_enabled = True
                print(f"MLflow tracking enabled. Experiment: {experiment_name}, ID: {experiment_id}")
            except Exception as e:
                print(f"Warning: Could not set up MLflow experiment: {str(e)}")
                self.mlflow_enabled = False
        except ImportError:
            print("Warning: MLflow not installed. Model tracking disabled.")
            self.mlflow_enabled = False
    
    def _log_parameters(self):
        """Log parameters to MLflow."""
        if not self.mlflow_enabled:
            return
        
        try:
            import mlflow
            
            # Log basic parameters
            mlflow.log_param("forecast_horizon", self.forecast_horizon)
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("scoring", self.scoring)
            mlflow.log_param("use_multiplex", self.use_multiplex)
            
            # Log transformations
            if self.transformations:
                mlflow.log_param("transformations", str(self.transformations))
                
            # Log seasonal period if applicable
            if self.seasonal_period:
                mlflow.log_param("seasonal_period", self.seasonal_period)
                
            # Log grid search parameters
            if isinstance(self.grid_search, dict):
                for algo, params in self.grid_search.items():
                    for param_name, param_values in params.items():
                        # Convert to string to handle non-serializable objects
                        mlflow.log_param(f"grid_{algo}_{param_name}", str(param_values))
            else:
                mlflow.log_param("grid_search", self.grid_search)
                
            # Log cross-validation parameters
            if isinstance(self.cross_validation, dict):
                for param_name, param_value in self.cross_validation.items():
                    mlflow.log_param(f"cv_{param_name}", param_value)
            else:
                mlflow.log_param("cross_validation", self.cross_validation)
        except Exception as e:
            print(f"Warning: Could not log parameters to MLflow: {str(e)}")
    
    def _log_model(self, name, model):
        """Log a model to MLflow using sktime's MLflow integration."""
        if not self.mlflow_enabled:
            return
        
        try:
            from sktime.utils.mlflow_sktime import log_model
            import mlflow
            
            # Set up prediction configuration for pyfunc flavor
            # This allows the model to be used for predictions with the pyfunc flavor
            pyfunc_predict_conf = {
                "predict_method": {
                    "predict": {},
                    "predict_interval": {"coverage": [0.8, 0.9]},
                    "predict_quantiles": {},
                    "predict_var": {},
                }
            }
            
            # Add prediction configuration to the model
            model.pyfunc_predict_conf = pyfunc_predict_conf
            
            # Log the model using sktime's MLflow integration
            log_model(
                sktime_model=model,
                artifact_path=f"models/{name}",
                conda_env=None,  # Use default conda environment
                registered_model_name=f"AdaptiveForecaster_{name}"
            )
            
            print(f"Model {name} logged to MLflow")
            
            # If this is a grid search result, log the best parameters
            if hasattr(model, 'best_params_'):
                for param_name, param_value in model.best_params_.items():
                    mlflow.log_param(f"best_{name}_{param_name}", str(param_value))
                
            # If this is a grid search result, log the best score
            if hasattr(model, 'best_score_'):
                mlflow.log_metric(f"best_{name}_score", model.best_score_)
        except Exception as e:
            print(f"Warning: Could not log model {name} to MLflow: {str(e)}")
    
    def _log_metrics(self, name, metrics):
        """Log metrics to MLflow."""
        if not self.mlflow_enabled:
            return
        
        try:
            import mlflow
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{name}_{metric_name}", metric_value)
        except Exception as e:
            print(f"Warning: Could not log metrics for {name} to MLflow: {str(e)}")
    
    def _fit_multiple_individual(self):
        """Fit multiple algorithms individually based on grid search settings."""
        # Track if any models were successfully fitted
        any_fitted = False
        
        # Iterate through all specified algorithms
        for algorithm_name in self.algorithms:
            print(f"Fitting {algorithm_name.upper()} model...")
            
            # Get base forecaster with transformations
            forecaster = self._get_algorithm(algorithm_name)
            
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
                    else:
                        # Fallback to simple fit if grid search fails
                        print(f"  GridSearchCV failed. Falling back to simple fit for {algorithm_name}...")
                        if self._fit_simple(algorithm_name, forecaster):
                            self.forecasters[algorithm_name] = forecaster
                            any_fitted = True
                            print(f"  {algorithm_name.upper()} model fitted successfully with simple fit!")
                except Exception as e:
                    print(f"  Error during grid search: {str(e)}")
                    print(f"  Falling back to simple fit for {algorithm_name}...")
                    if self._fit_simple(algorithm_name, forecaster):
                        self.forecasters[algorithm_name] = forecaster
                        any_fitted = True
                        print(f"  {algorithm_name.upper()} model fitted successfully with simple fit!")
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
        
        if not any_fitted:
            print("Warning: No models were successfully fitted.")
        
        return self
    
    def _fit_simple(self, algorithm_name, forecaster):
        """Fit a forecaster without grid search."""
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
        """Perform grid search for a specific algorithm."""
        print(f"  Performing grid search for {algorithm_name}...")
        
        # Initialize param_grid to avoid UnboundLocalError
        param_grid = {}
        
        try:
            # Get parameter grid
            if isinstance(self.grid_search, dict) and algorithm_name in self.grid_search:
                param_grid = self.grid_search[algorithm_name]
            #else:
            #    param_grid = self._get_param_grid(algorithm_name)
            
            # If no param_grid is provided, just fit without grid search
            if not param_grid:
                print(f"  No parameter grid provided for {algorithm_name}. Fitting with default parameters.")
                return self._fit_simple(algorithm_name, forecaster)
            
            # Configure cross-validation if specified
            if self.cross_validation:
                cv_params = self.cross_validation if isinstance(self.cross_validation, dict) else {}
                cv = self._get_cv_splitter(cv_params)
            else:
                cv = self._get_cv_splitter()
            
            # Get the appropriate metric object
            if self.scoring in self.METRIC_MAP:
                scoring_metric = self.METRIC_MAP[self.scoring]
            else:
                print(f"  Warning: Unknown scoring metric '{self.scoring}'. Defaulting to 'mae'.")
                scoring_metric = self.METRIC_MAP["mae"]
            
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
                return self._try_alternative_metrics(algorithm_name, forecaster, param_grid, cv)
            else:
                # For other errors, fit with default parameters
                return self._fit_simple(algorithm_name, forecaster)
    
    def _try_alternative_metrics(self, algorithm_name, forecaster, param_grid, cv):
        """Try alternative metrics if the original scoring fails."""
        # If param_grid is empty, just fit without grid search
        if not param_grid:
            print(f"  No parameter grid provided for {algorithm_name}. Fitting with default parameters.")
            return self._fit_simple(algorithm_name, forecaster)
            
        for metric_name, metric_obj in self.METRIC_MAP.items():
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
        return self._fit_simple(algorithm_name, forecaster)
        
    def _fit_multiplex(self):
        """Fit models using MultiplexForecaster for efficient model selection."""
        # Create a list of forecasters with transformations applied
        forecaster_list = []
        
        for algorithm_name in self.algorithms:
            if algorithm_name == 'ensemble':
                continue  # Skip ensemble for MultiplexForecaster
                
            print(f"Preparing {algorithm_name.upper()} model...")
            
            # Get base forecaster with transformations
            forecaster = self._get_algorithm(algorithm_name)
            
            if forecaster is not None:
                forecaster_list.append((algorithm_name, forecaster))
        
        if not forecaster_list:
            raise ValueError("No valid forecasters to fit.")
            
        # Create MultiplexForecaster
        multiplex = MultiplexForecaster(
            forecasters=forecaster_list,
            selected_forecaster=None  # Default to first forecaster initially
        )
        
        # Configure cross-validation
        cv_params = self.cross_validation if isinstance(self.cross_validation, dict) else {}
        cv = self._get_cv_splitter(cv_params)
        
        # Create parameter grid for MultiplexForecaster
        # Only include the selected forecasters in the grid search
        param_grid = {"selected_forecaster": [name for name, _ in forecaster_list if name in self.selected_forecasters]}
        
        if not param_grid["selected_forecaster"]:
            print("Warning: No valid selected forecasters for grid search. Using all available forecasters.")
            param_grid["selected_forecaster"] = [name for name, _ in forecaster_list]
        
        print(f"Forecasters to be tuned: {param_grid['selected_forecaster']}")
        
        # Add algorithm-specific parameters
        self._add_algorithm_params_to_grid(param_grid)
        
        print("Performing grid search across selected algorithms...")
        print(f"Parameter grid: {param_grid}")
        print(f"Cross-validation: {cv}")
        
        # Get the appropriate metric object
        if self.scoring in self.METRIC_MAP:
            scoring_metric = self.METRIC_MAP[self.scoring]
        else:
            print(f"Warning: Unknown scoring metric '{self.scoring}'. Defaulting to 'mae'.")
            scoring_metric = self.METRIC_MAP["mae"]
        
        # Create and fit grid search with the proper scoring object
        try:
            grid_search = self._create_and_fit_grid_search(multiplex, param_grid, cv, scoring_metric)
        except Exception as e:
            print(f"Error during grid search: {str(e)}")
            print("Trying with a different scoring metric...")
            
            # Try with different metrics
            for metric_name, metric_obj in self.METRIC_MAP.items():
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
        
        # Also store the best individual forecaster for reference
        if best_algorithm:
            self._store_best_individual_forecaster(best_algorithm, best_params)
            
        print("MultiplexForecaster fitted successfully!")
        return self
    
    def _add_algorithm_params_to_grid(self, param_grid):
        """Add algorithm-specific parameters to the parameter grid."""
        if isinstance(self.grid_search, dict):
            for algo_name in param_grid["selected_forecaster"]:
                if algo_name in self.grid_search:
                    for param, values in self.grid_search[algo_name].items():
                        param_grid[f"{algo_name}__{param}"] = values
        else:
            # Add default parameter grids for each algorithm
            for algo_name in param_grid["selected_forecaster"]:
                default_params = self._get_param_grid(algo_name)
                for param, values in default_params.items():
                    param_grid[f"{algo_name}__{param}"] = values
    
    def _create_and_fit_grid_search(self, forecaster, param_grid, cv, scoring_metric):
        """Create and fit a grid search CV object."""
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
        """Store the best individual forecaster with its optimal parameters."""
        # Get the best individual forecaster with its optimal parameters
        best_forecaster = self._get_algorithm(best_algorithm)
        
        # Apply the best parameters found by grid search
        best_algo_params = {
            k.replace(f"{best_algorithm}__", ""): v 
            for k, v in best_params.items() 
            if k.startswith(f"{best_algorithm}__")
        }
        
        if best_algo_params:
            best_forecaster.set_params(**best_algo_params)
            
        # Fit the best individual forecaster
        self._fit_simple(best_algorithm, best_forecaster)
        
        # Store the forecaster
        self.forecasters[best_algorithm] = best_forecaster
    
    def predict(self):
        """Generate forecasts for all fitted models."""
        # Check if any forecasters have been fitted
        if not hasattr(self, 'forecasters') or not self.forecasters:
            print("No forecasters have been fitted. Run fit() first.")
            return self
        
        # Initialize forecasts and metrics dictionaries if they don't exist
        if not hasattr(self, 'forecasts'):
            self.forecasts = {}
        if not hasattr(self, 'metrics'):
            self.metrics = {}
        
        # Start MLflow run for predictions if not already in a run
        if self.mlflow_enabled:
            try:
                import mlflow
                from datetime import datetime
                
                # Check if already in a run
                try:
                    mlflow.active_run()
                except:
                    # Start a new run if not already in one
                    mlflow.start_run(run_name=f"AdaptiveForecaster_Predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            except Exception as e:
                print(f"Warning: Could not start MLflow run for predictions: {str(e)}")
        
        # If using MultiplexForecaster, predict with it first
        if "multiplex" in self.forecasters:
            print("Generating forecast with MultiplexForecaster...")
            self._predict_with_forecaster("multiplex")
            
            # Get the best algorithm name
            best_algo = self.get_best_algorithm()
            if best_algo in self.forecasters:
                print(f"Best algorithm selected by MultiplexForecaster: {best_algo.upper()}")
                
                # Log to MLflow
                if self.mlflow_enabled:
                    try:
                        import mlflow
                        mlflow.log_param("best_algorithm", best_algo)
                    except Exception as e:
                        print(f"Warning: Could not log best algorithm to MLflow: {str(e)}")
        
        # Generate forecasts for all other fitted models
        for name in self.forecasters:
            if name == "multiplex":
                continue  # Already predicted
            
            print(f"Generating forecast for {name.upper()}...")
            self._predict_with_forecaster(name)
        
        # Determine best model based on RMSE if metrics are available
        if self.metrics:
            best_model_name = min(self.metrics, key=lambda x: self.metrics[x]['rmse'])
            self.best_model = best_model_name
            self.best_forecast = self.forecasts[best_model_name]
            
            # Log to MLflow
            if self.mlflow_enabled:
                try:
                    import mlflow
                    mlflow.log_param("best_model_by_rmse", best_model_name)
                    
                    # Log best metrics
                    best_metrics = self.metrics[best_model_name]
                    for metric_name, metric_value in best_metrics.items():
                        mlflow.log_metric(f"best_{metric_name}", metric_value)
                except Exception as e:
                    print(f"Warning: Could not log best model metrics to MLflow: {str(e)}")
        
        # Check if any forecasts were generated
        if not self.forecasts:
            print("No forecasts could be generated. Check for errors during prediction.")
            return self
        
        print("All forecasts generated successfully!")
        
        # Print the forecasts
        self.print_forecasts()
        
        # End MLflow run if we started one
        if self.mlflow_enabled:
            try:
                import mlflow
                if mlflow.active_run():
                    mlflow.end_run()
            except Exception as e:
                print(f"Warning: Could not end MLflow run: {str(e)}")
        
        return self
    
    def _predict_with_forecaster(self, name):
        """Generate forecast and calculate metrics for a specific forecaster."""
        try:
            forecaster = self.forecasters[name]
            
            # Generate forecast
            if self.X_test is None:
                forecast = forecaster.predict(fh=self.fh)
            else:
                forecast = forecaster.predict(fh=self.fh, X=self.X_test)
                
            self.forecasts[name] = forecast
            
            # Calculate metrics if test data is available
            if len(self.y_test) >= self.forecast_horizon:
                try:
                    y_test_pred = self.y_test.iloc[:self.forecast_horizon]
                    forecast_values = forecast[:len(y_test_pred)]
                    
                    # Calculate metrics properly by calling the metric objects
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
                    
                    # Log metrics to MLflow
                    if self.mlflow_enabled:
                        self._log_metrics(name, self.metrics[name])
                except Exception as e:
                    print(f"  Warning: Could not calculate metrics for {name}: {str(e)}")
                    # Still keep the forecast even if metrics calculation fails
            
            return True
        except Exception as e:
            print(f"  Error predicting with {name}: {str(e)}")
            return False
    
    def get_best_algorithm(self):
        """Get the best algorithm selected by MultiplexForecaster."""
        if "multiplex" in self.forecasters:
            multiplex = self.forecasters["multiplex"]
            if hasattr(multiplex, "best_params_"):
                return multiplex.best_params_.get("selected_forecaster", "unknown")
        return self.best_model
    
    def get_best_params(self):
        """Get the best parameters found during grid search."""
        if "multiplex" in self.forecasters:
            multiplex = self.forecasters["multiplex"]
            if hasattr(multiplex, "best_params_"):
                return multiplex.best_params_
        return None
    
    def get_cv_results(self):
        """Get cross-validation results from grid search."""
        if self.cv_results_ is not None:
            # Convert to DataFrame for better visualization
            results_df = pd.DataFrame(self.cv_results_)
            
            # Find the appropriate score column
            score_cols = [col for col in results_df.columns if col.startswith('mean_test_')]
            
            if score_cols:
                # Use the first score column for sorting
                score_col = score_cols[0]
                print(f"Sorting by {score_col}")
                
                # Sort by the score column
                results_df = results_df.sort_values(score_col, ascending=False)
            else:
                print("Warning: No 'mean_test_' columns found in CV results. Results will not be sorted.")
            
            return results_df
        else:
            return "No cross-validation results available. Run fit() with grid_search=True first."
    
    def print_cv_results(self, n_best=5):
        """Print the best cross-validation results."""
        if self.cv_results_ is None:
            print("No cross-validation results available. Run fit() with grid_search=True first.")
            return
        
        results_df = self.get_cv_results()
        
        # Check if results_df is a string (error message)
        if isinstance(results_df, str):
            print(results_df)
            return
        
        # Select columns to display
        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        score_cols = [col for col in results_df.columns if col.startswith('mean_test_')]
        display_cols = param_cols + score_cols
        
        if not display_cols:
            print("No parameter or score columns found in CV results.")
            print("Available columns:", results_df.columns.tolist())
            return
        
        # Print the best n results
        print("\n" + "="*50)
        print("CROSS-VALIDATION RESULTS (TOP {})".format(n_best))
        print("="*50)
        
        top_results = results_df[display_cols].head(n_best)
        
        # Format the scores
        for col in score_cols:
            if col in top_results.columns:
                top_results[col] = top_results[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        
        print(top_results)
        print("="*50)
    
    def print_forecasts(self):
        """Print forecasts for all models."""
        if not self.forecasts:
            print("No forecasts available. Run predict() first.")
            return
            
        # Create a DataFrame for better display
        forecast_df = pd.DataFrame()
        
        # Add actual values if available
        if len(self.y_test) > 0:
            # Check if y_test is already a DataFrame or Series
            if isinstance(self.y_test, pd.DataFrame):
                actual_values = self.y_test.iloc[:self.forecast_horizon]
            else:  # It's a Series
                actual_values = self.y_test.iloc[:self.forecast_horizon]
                
            forecast_df['Actual'] = actual_values
            
        # Add forecasts
        for name, forecast in self.forecasts.items():
            # Check if forecast is a DataFrame or Series
            if isinstance(forecast, pd.DataFrame):
                forecast_values = forecast.iloc[:self.forecast_horizon, 0]  # Take first column
            else:  # It's a Series
                forecast_values = forecast.iloc[:self.forecast_horizon]
                
            forecast_df[f'{name.capitalize()} Forecast'] = forecast_values
            
        # Add errors if actual values are available
        if 'Actual' in forecast_df.columns:
            for name in self.forecasters.keys():
                forecast_col = f'{name.capitalize()} Forecast'
                forecast_df[f'{name.capitalize()} Error'] = forecast_df['Actual'] - forecast_df[forecast_col]
                
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
            metrics_df = pd.DataFrame(self.metrics).T
            
            # Format metrics for better display
            for col in metrics_df.columns:
                metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.4f}")
                
            print(metrics_df)
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
        """
        if not self.forecasts:
            print("No forecasts available. Run predict() first.")
            return
            
        # Select models to plot
        if models is None:
            models_to_plot = list(self.forecasts.keys())
        else:
            models_to_plot = [m for m in models if m in self.forecasts]
            if not models_to_plot:
                print("None of the specified models found in forecasts.")
                return
                
        # Create figure
        plt.figure(figsize=figsize)
        
        # Convert Period index to datetime if needed
        def convert_index_to_datetime(idx):
            if hasattr(idx, 'dtype') and str(idx.dtype).startswith('period'):
                return idx.to_timestamp()
            return idx
        
        # Plot historical data if requested
        if include_history:
            y_train_idx = convert_index_to_datetime(self.y_train.index)
            
            if isinstance(self.y_train, pd.DataFrame):
                plt.plot(y_train_idx, self.y_train.iloc[:, 0], 'k-', label='Historical Data')
            else:
                plt.plot(y_train_idx, self.y_train, 'k-', label='Historical Data')
                
        # Plot actual test data if available
        if len(self.y_test) > 0:
            y_test_idx = convert_index_to_datetime(self.y_test.index)
            
            if isinstance(self.y_test, pd.DataFrame):
                plt.plot(y_test_idx[:self.forecast_horizon], 
                         self.y_test.iloc[:self.forecast_horizon, 0], 
                         'b-', label='Actual')
            else:
                plt.plot(y_test_idx[:self.forecast_horizon], 
                         self.y_test.iloc[:self.forecast_horizon], 
                         'b-', label='Actual')
                
        # Plot forecasts
        colors = ['r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, name in enumerate(models_to_plot):
            forecast = self.forecasts[name]
            color = colors[i % len(colors)]
            
            # Handle 'best' model
            if name == 'best' and hasattr(self, 'best_model') and self.best_model:
                forecast = self.forecasts[self.best_model]
                name = f"Best ({self.best_model})"
            
            forecast_idx = convert_index_to_datetime(forecast.index)
            
            if isinstance(forecast, pd.DataFrame):
                plt.plot(forecast_idx[:self.forecast_horizon], 
                         forecast.iloc[:self.forecast_horizon, 0], 
                         f'{color}-', label=f'{name.capitalize()} Forecast')
            else:
                plt.plot(forecast_idx[:self.forecast_horizon], 
                         forecast.iloc[:self.forecast_horizon], 
                         f'{color}-', label=f'{name.capitalize()} Forecast')
                
        # Add labels and legend
        plt.title('Time Series Forecast')
        plt.xlabel('Time')
        plt.ylabel(self.target)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Show plot
        plt.show()

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

    def _store_cv_results(self, grid_search):
        """Store cross-validation results from grid search."""
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