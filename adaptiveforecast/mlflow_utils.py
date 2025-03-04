"""
MLflow integration utilities for the AdaptiveForecaster.
"""

import warnings


def setup_mlflow():
    """
    Set up MLflow tracking.
    
    Returns
    -------
    bool
        Whether MLflow is enabled.
    """
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
            print(f"MLflow tracking enabled. Experiment: {experiment_name}, ID: {experiment_id}")
            return True
        except Exception as e:
            warnings.warn(f"Could not set up MLflow experiment: {str(e)}")
            return False
    except ImportError:
        warnings.warn("MLflow not installed. Model tracking disabled.")
        return False


def log_parameters(mlflow_enabled, params_dict):
    """
    Log parameters to MLflow.
    
    Parameters
    ----------
    mlflow_enabled : bool
        Whether MLflow is enabled.
    params_dict : dict
        Dictionary of parameters to log.
    """
    if not mlflow_enabled:
        return
    
    try:
        import mlflow
        
        # Log basic parameters
        for key, value in params_dict.items():
            # Convert to string to handle non-serializable objects
            if not isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                value = str(value)
            mlflow.log_param(key, value)
            
    except Exception as e:
        warnings.warn(f"Could not log parameters to MLflow: {str(e)}")


def log_model(mlflow_enabled, name, model):
    """
    Log a model to MLflow.
    
    Parameters
    ----------
    mlflow_enabled : bool
        Whether MLflow is enabled.
    name : str
        Model name.
    model : object
        Model object.
    """
    if not mlflow_enabled:
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
        warnings.warn(f"Could not log model {name} to MLflow: {str(e)}")


def log_metrics(mlflow_enabled, name, metrics):
    """
    Log metrics to MLflow.
    
    Parameters
    ----------
    mlflow_enabled : bool
        Whether MLflow is enabled.
    name : str
        Model name.
    metrics : dict
        Dictionary of metrics.
    """
    if not mlflow_enabled or metrics is None:
        return
    
    try:
        import mlflow
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"{name}_{metric_name}", metric_value)
    except Exception as e:
        warnings.warn(f"Could not log metrics for {name} to MLflow: {str(e)}")


def start_run(mlflow_enabled, run_name=None):
    """
    Start an MLflow run.
    
    Parameters
    ----------
    mlflow_enabled : bool
        Whether MLflow is enabled.
    run_name : str, optional
        Name for the run.
    """
    if not mlflow_enabled:
        return
    
    try:
        import mlflow
        from datetime import datetime
        
        # Check if already in a run
        try:
            current_run = mlflow.active_run()
            if current_run is not None:
                return  # Already in a run
        except:
            pass
            
        # Start a new run
        if run_name is None:
            run_name = f"AdaptiveForecaster_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        mlflow.start_run(run_name=run_name)
    except Exception as e:
        warnings.warn(f"Could not start MLflow run: {str(e)}")


def end_run(mlflow_enabled):
    """
    End the current MLflow run.
    
    Parameters
    ----------
    mlflow_enabled : bool
        Whether MLflow is enabled.
    """
    if not mlflow_enabled:
        return
    
    try:
        import mlflow
        
        # Check if in a run
        try:
            current_run = mlflow.active_run()
            if current_run is not None:
                mlflow.end_run()
        except:
            pass
    except Exception as e:
        warnings.warn(f"Could not end MLflow run: {str(e)}")