import pandas as pd
import matplotlib.pyplot as plt
from adaptiveforecast.forecaster import AdaptiveForecaster
from sktime.datasets import load_airline

# Load the airline passengers dataset
data = load_airline()

# Basic usage with algorithm-specific transformations
forecaster = AdaptiveForecaster(
    df=data,
    forecast_horizon=12,  # Forecast 12 steps ahead
    algorithms=['naive', 'arima', 'ets'],  # Use these algorithms
    transformations={
        'naive': None,  # No transformations for naive
        'arima': ['deseasonalize', 'detrend'],  # Apply both to ARIMA
        'ets': ['deseasonalize']  # Only deseasonalize for ETS
    },
    seasonal_period=12,  # Monthly seasonality
    grid_search=True,  # Perform grid search for hyperparameter tuning
    cross_validation=True  # Use cross-validation for model selection
)

# Fit the models
forecaster.fit()

# Generate forecasts
forecaster.predict()

# Plot the forecasts
forecaster.plot_forecasts()

# Get the forecast DataFrame
forecast_df = forecaster.get_forecast_dataframe()
print(forecast_df)

# Get the best model and parameters
best_model = forecaster.best_model
best_params = forecaster.get_best_params()
print(f"Best model: {best_model}")
print(f"Best parameters: {best_params}")

# Save the forecast summary
forecaster.save_summary('forecast_results.json') 