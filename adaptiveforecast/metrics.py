"""
Metrics calculation and evaluation utilities.
"""

import pandas as pd
import numpy as np
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError, 
    MeanAbsoluteError, 
    MeanSquaredError
)


# Define metric mapping
METRIC_MAP = {
    "mae": MeanAbsoluteError(),
    "mse": MeanSquaredError(),
    "mape": MeanAbsolutePercentageError(),
    "rmse": MeanSquaredError(square_root=True)
}


def calculate_metrics(y_test, forecast, forecast_horizon):
    """
    Calculate performance metrics for a forecast.
    
    Parameters
    ----------
    y_test : pandas.Series
        Actual test values.
    forecast : pandas.Series
        Forecast values.
    forecast_horizon : int
        Forecast horizon.
    
    Returns
    -------
    dict
        Dictionary of metrics.
    """
    try:
        if len(y_test) < forecast_horizon:
            return None
            
        y_test_pred = y_test.iloc[:forecast_horizon]
        forecast_values = forecast[:len(y_test_pred)]
        
        # Calculate metrics properly by calling the metric objects
        mae = MeanAbsoluteError().evaluate(y_test_pred, forecast_values)
        mse = MeanSquaredError().evaluate(y_test_pred, forecast_values)
        rmse = np.sqrt(mse)
        mape = MeanAbsolutePercentageError().evaluate(y_test_pred, forecast_values)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
    except Exception as e:
        print(f"Warning: Could not calculate metrics: {str(e)}")
        return None


def format_metrics_dataframe(metrics):
    """
    Format metrics for display.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics by model.
    
    Returns
    -------
    pandas.DataFrame
        Formatted metrics DataFrame.
    """
    if not metrics:
        return None
        
    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame(metrics).T
    
    # Format metrics for better display
    for col in metrics_df.columns:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.4f}")
        
    return metrics_df


def get_best_model(metrics):
    """
    Determine the best model based on RMSE.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics by model.
    
    Returns
    -------
    str
        Name of the best model.
    """
    if not metrics:
        return None
        
    return min(metrics, key=lambda x: metrics[x]['rmse'])


def get_cv_results_dataframe(cv_results):
    """
    Convert CV results to a DataFrame.
    
    Parameters
    ----------
    cv_results : dict
        Cross-validation results dictionary.
    
    Returns
    -------
    pandas.DataFrame
        Formatted CV results DataFrame.
    """
    if cv_results is None:
        return "No cross-validation results available."
    
    # Convert to DataFrame for better visualization
    results_df = pd.DataFrame(cv_results)
    
    # Find the appropriate score column
    score_cols = [col for col in results_df.columns if col.startswith('mean_test_')]
    
    if score_cols:
        # Use the first score column for sorting
        score_col = score_cols[0]
        
        # Sort by the score column
        results_df = results_df.sort_values(score_col, ascending=False)
    
    return results_df


def select_cv_display_columns(results_df, n_best=5):
    """
    Select columns for displaying CV results.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        CV results DataFrame.
    n_best : int, optional
        Number of best results to display.
    
    Returns
    -------
    pandas.DataFrame
        Selected columns and rows.
    """
    if isinstance(results_df, str):
        return results_df
    
    # Select columns to display
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    score_cols = [col for col in results_df.columns if col.startswith('mean_test_')]
    display_cols = param_cols + score_cols
    
    if not display_cols:
        return "No parameter or score columns found in CV results."
    
    # Format the scores for readable display
    result = results_df[display_cols].head(n_best).copy()
    
    for col in score_cols:
        if col in result.columns:
            result[col] = result[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
    return result