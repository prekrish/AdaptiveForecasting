import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adaptiveforecast.interface import AdaptiveForecaster
from sktime.datasets import load_airline

# Load the airline passengers dataset
df = load_airline()

# Basic usage example
print("\n=== BASIC USAGE EXAMPLE ===")
forecaster = AdaptiveForecaster(
    df=df,
    forecast_horizon=12,
    test_size=24,
    algorithms=['naive', 'arima', 'ets'],
    transformations=['impute', 'deseasonalize'],
    impute_method='mean',
    seasonal_period=12
)

# Check missing values
missing_summary = forecaster.check_missing_values()
print("Missing Values Summary:")
print(missing_summary)

# Fit models
forecaster.fit()

# Generate forecasts
forecaster.predict()

# Get summary of results
summary = forecaster.get_summary()
print("\nForecast Summary:")
print(summary)

# Plot forecasts
plt.figure(figsize=(12, 6))
forecaster.plot_forecasts(models=['best', 'arima'])
plt.title("Basic Forecast Example")
plt.tight_layout()
plt.savefig("basic_forecast.png")
plt.close()