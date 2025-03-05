"""
MLflow integration utilities for the AdaptiveForecaster.
"""
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import warnings
import mlflow
from mlflow.tracking import MlflowClient
import os
from typing import Optional, Dict, Any


def setup_mlflow(experiment_name: Optional[str] = None) -> bool:
    """
    Set up MLflow tracking.
    
    Parameters
    ----------
    experiment_name : str, optional
        Name of the MLflow experiment. If None, uses 'Default'
    
    Returns
    -------
    bool
        True if MLflow was successfully set up, False otherwise
    """
    try:
        # Set tracking URI to localhost:5000
        tracking_uri = "http://localhost:5000"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Set tracking URI to: {tracking_uri}")
        
        # Create MLflow client
        client = MlflowClient(tracking_uri=tracking_uri)
        
        experiment_id = None
        
        # Set or create experiment
        if experiment_name:
            # First try to get existing experiment
            existing_experiment = client.get_experiment_by_name(experiment_name)
            
            if existing_experiment is not None:
                experiment_id = existing_experiment.experiment_id
                print(f"Found existing experiment: {experiment_name} with ID: {experiment_id}")
            else:
                # Create new experiment if it doesn't exist
                try:
                    experiment_id = client.create_experiment(experiment_name)
                    print(f"Created new experiment: {experiment_name} with ID: {experiment_id}")
                except Exception as e:
                    print(f"Error creating experiment: {str(e)}")
                    # One more try to get the experiment in case of race condition
                    existing_experiment = client.get_experiment_by_name(experiment_name)
                    if existing_experiment is not None:
                        experiment_id = existing_experiment.experiment_id
                        print(f"Found existing experiment: {experiment_name} with ID: {experiment_id}")
                    else:
                        raise Exception(f"Could not create or find experiment: {experiment_name}")
        else:
            # Handle default experiment
            default_exp = client.get_experiment_by_name("Default")
            if default_exp is None:
                experiment_id = client.create_experiment("Default")
                print(f"Created default experiment with ID: {experiment_id}")
            else:
                experiment_id = default_exp.experiment_id
                print(f"Using default experiment with ID: {experiment_id}")
            experiment_name = "Default"
        
        # Set the active experiment
        if experiment_id is not None:
            mlflow.set_experiment(experiment_name)
            # Verify the experiment was set correctly
            current_experiment = client.get_experiment_by_name(experiment_name)
            if current_experiment is not None:
                print(f"Successfully set active experiment to: {experiment_name} (ID: {current_experiment.experiment_id})")
            else:
                raise Exception(f"Failed to set active experiment to: {experiment_name}")
            
        return True
    except Exception as e:
        print(f"Error setting up MLflow: {str(e)}")
        return False


def log_forecaster(forecaster, experiment_name: str, model_name: str) -> None:
    """
    Log a fitted AdaptiveForecaster object to MLflow, with separate runs for each model.
    
    Parameters
    ----------
    forecaster : AdaptiveForecaster
        Fitted AdaptiveForecaster object
    experiment_name : str
        Name of the MLflow experiment
    model_name : str
        Base name for the models
    """
    try:
        # Set up MLflow
        if not setup_mlflow(experiment_name):
            raise Exception("Failed to set up MLflow")
            
        # Set tracking URI and experiment
        tracking_uri = "http://localhost:5000"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        # Common parameters for all models
        common_params = {
            "target": forecaster.target,
            "forecast_horizon": forecaster.forecast_horizon,
            "test_size": forecaster.test_size,
            "algorithms": str(forecaster.algorithms),
            "transformations": str(forecaster.transformations),
            "seasonal_period": forecaster.seasonal_period,
            "grid_search": str(forecaster.grid_search),
            "cross_validation": str(forecaster.cross_validation),
            "scoring": forecaster.scoring
        }
        
        # Log each model in a separate run
        for name, model in forecaster.forecasters.items():
            run_name = f"{model_name}_{name}"
            with mlflow.start_run(run_name=run_name) as run:
                print(f"Started MLflow run for model {name}: {run_name}")
                
                # Log common parameters
                mlflow.log_params(common_params)
                
                # Set up prediction configuration
                pyfunc_predict_conf = {
                    "predict_method": {
                        "predict": {},
                        "predict_interval": {"coverage": [0.8, 0.9]},
                        "predict_quantiles": {},
                        "predict_var": {},
                    }
                }
                model.pyfunc_predict_conf = pyfunc_predict_conf
                
                # Log the model
                from sktime.utils.mlflow_sktime import log_model as sktime_log_model
                sktime_log_model(
                    sktime_model=model,
                    artifact_path="model",
                    conda_env=None,
                    registered_model_name=None
                )
                
                # Log model-specific parameters
                if hasattr(model, 'best_params_'):
                    mlflow.log_params({"best_params": str(model.best_params_)})
                
                # Log model-specific metrics
                if name in forecaster.metrics:
                    mlflow.log_metrics(forecaster.metrics[name])
                
                # Log CV results if available
                if hasattr(forecaster, 'cv_results') and name in forecaster.cv_results:
                    mlflow.log_dict(forecaster.cv_results[name], "cv_results.json")
                
                # Log if this is the best model
                if forecaster.best_model == name:
                    mlflow.log_param("is_best_model", True)
                
                print(f"Successfully logged model {name} to MLflow run: {run.info.run_id}")
            
    except Exception as e:
        warnings.warn(f"Error logging to MLflow: {str(e)}")


def log_parameters(mlflow_enabled, params_dict):
    """Deprecated: Use log_forecaster instead."""
    warnings.warn("This function is deprecated. Use log_forecaster instead.", DeprecationWarning)
    pass


def log_model(mlflow_enabled, name, model, register=False):
    """Deprecated: Use log_forecaster instead."""
    warnings.warn("This function is deprecated. Use log_forecaster instead.", DeprecationWarning)
    pass


def log_metrics(mlflow_enabled, name, metrics):
    """Deprecated: Use log_forecaster instead."""
    warnings.warn("This function is deprecated. Use log_forecaster instead.", DeprecationWarning)
    pass


def start_run(mlflow_enabled, run_name=None):
    """Deprecated: Use log_forecaster instead."""
    warnings.warn("This function is deprecated. Use log_forecaster instead.", DeprecationWarning)
    pass


def end_run(mlflow_enabled):
    """Deprecated: Use log_forecaster instead."""
    warnings.warn("This function is deprecated. Use log_forecaster instead.", DeprecationWarning)
    pass