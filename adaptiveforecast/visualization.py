"""
Visualization utilities for the AdaptiveForecaster.
"""
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_forecasts(y_train, y_test, forecasts, forecast_horizon, 
                   best_model=None, figsize=(12, 6), include_history=True, 
                   models=None, target=''):
    """
    Plot forecasts for all or selected models.
    
    Parameters
    ----------
    y_train : pandas.Series
        Training data.
    y_test : pandas.Series
        Test data.
    forecasts : dict
        Dictionary of forecasts by model.
    forecast_horizon : int
        Forecast horizon.
    best_model : str, optional
        Name of the best model.
    figsize : tuple, default=(12, 6)
        Figure size.
    include_history : bool, default=True
        Whether to include historical data in the plot.
    models : list, default=None
        List of model names to plot. If None, all models are plotted.
    target : str, default=''
        Name of the target variable.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    if not forecasts:
        print("No forecasts available.")
        return
        
    # Select models to plot
    if models is None:
        models_to_plot = list(forecasts.keys())
    else:
        models_to_plot = [m for m in models if m in forecasts]
        if not models_to_plot:
            print("None of the specified models found in forecasts.")
            return
            
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Convert Period index to datetime if needed
    def convert_index_to_datetime(idx):
        if hasattr(idx, 'dtype') and str(idx.dtype).startswith('period'):
            return idx.to_timestamp()
        return idx
    
    # Plot historical data if requested
    if include_history:
        y_train_idx = convert_index_to_datetime(y_train.index)
        
        if isinstance(y_train, pd.DataFrame):
            plt.plot(y_train_idx, y_train.iloc[:, 0], 'k-', label='Historical Data')
        else:
            plt.plot(y_train_idx, y_train, 'k-', label='Historical Data')
            
    # Plot actual test data if available
    if y_test is not None and len(y_test) > 0:
        y_test_idx = convert_index_to_datetime(y_test.index)
        
        if isinstance(y_test, pd.DataFrame):
            plt.plot(y_test_idx[:forecast_horizon], 
                     y_test.iloc[:forecast_horizon, 0], 
                     'b-', label='Actual')
        else:
            plt.plot(y_test_idx[:forecast_horizon], 
                     y_test.iloc[:forecast_horizon], 
                     'b-', label='Actual')
            
    # Plot forecasts
    colors = ['r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, name in enumerate(models_to_plot):
        forecast = forecasts[name]
        color = colors[i % len(colors)]
        
        # Highlight best model
        if name == best_model:
            linestyle = '--'
            linewidth = 2.5
            name = f"Best ({name})"
        else:
            linestyle = '-'
            linewidth = 1.5
        
        forecast_idx = convert_index_to_datetime(forecast.index)
        
        if isinstance(forecast, pd.DataFrame):
            plt.plot(forecast_idx[:forecast_horizon], 
                     forecast.iloc[:forecast_horizon, 0], 
                     f'{color}{linestyle}', 
                     linewidth=linewidth,
                     label=f'{name.capitalize()} Forecast')
        else:
            plt.plot(forecast_idx[:forecast_horizon], 
                     forecast.iloc[:forecast_horizon], 
                     f'{color}{linestyle}', 
                     linewidth=linewidth,
                     label=f'{name.capitalize()} Forecast')
            
    # Add labels and legend
    plt.title('Time Series Forecast')
    plt.xlabel('Time')
    plt.ylabel(target if target else 'Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return fig


def create_forecast_dataframe(y_test, forecasts, forecast_horizon):
    """
    Create a DataFrame for displaying forecasts.
    
    Parameters
    ----------
    y_test : pandas.Series
        Test data.
    forecasts : dict
        Dictionary of forecasts by model.
    forecast_horizon : int
        Forecast horizon.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with forecasts and errors.
    """
    if not forecasts:
        return None
        
    # Create a DataFrame for better display
    forecast_df = pd.DataFrame()
    
    # Add actual values if available
    if y_test is not None and len(y_test) > 0:
        # Check if y_test is already a DataFrame or Series
        if isinstance(y_test, pd.DataFrame):
            actual_values = y_test.iloc[:forecast_horizon]
        else:  # It's a Series
            actual_values = y_test.iloc[:forecast_horizon]
            
        forecast_df['Actual'] = actual_values
        
    # Add forecasts
    for name, forecast in forecasts.items():
        # Check if forecast is a DataFrame or Series
        if isinstance(forecast, pd.DataFrame):
            forecast_values = forecast.iloc[:forecast_horizon, 0]  # Take first column
        else:  # It's a Series
            forecast_values = forecast.iloc[:forecast_horizon]
            
        forecast_df[f'{name.capitalize()} Forecast'] = forecast_values
        
    # Add errors if actual values are available
    if 'Actual' in forecast_df.columns:
        for name in forecasts.keys():
            forecast_col = f'{name.capitalize()} Forecast'
            if forecast_col in forecast_df.columns:
                forecast_df[f'{name.capitalize()} Error'] = forecast_df['Actual'] - forecast_df[forecast_col]
                
    return forecast_df


def plot_cv_results(cv_results, n_best=5, figsize=(10, 6)):
    """
    Plot cross-validation results.
    
    Parameters
    ----------
    cv_results : pandas.DataFrame
        Cross-validation results DataFrame.
    n_best : int, optional
        Number of best results to plot.
    figsize : tuple, optional
        Figure size.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    if cv_results is None or isinstance(cv_results, str):
        print("No cross-validation results to plot.")
        return None
    
    # Find the score columns
    score_cols = [col for col in cv_results.columns if col.startswith('mean_test_')]
    if not score_cols:
        print("No score columns found in CV results.")
        return None
    
    # Use the first score column
    score_col = score_cols[0]
    
    # Get top n results
    top_results = cv_results.sort_values(score_col, ascending=False).head(n_best)
    
    # Extract relevant parameter for plotting
    param_cols = [col for col in top_results.columns if col.startswith('param_')]
    if not param_cols:
        print("No parameter columns found in CV results.")
        return None
    
    # Select most informative parameter column (e.g., selected_forecaster)
    param_col = next((col for col in param_cols if 'selected_forecaster' in col), param_cols[0])
    
    # Get param values and scores
    param_values = top_results[param_col].apply(str)
    scores = top_results[score_col]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.bar(param_values, scores)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    # Add labels and title
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Score')
    ax.set_title(f'Top {n_best} Cross-Validation Results')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig