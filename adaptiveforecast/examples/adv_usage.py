# First, suppress warnings before any imports
import warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.")
warnings.filterwarnings("ignore", message="The user-specified parameters provided alongside auto=True in AutoETS may not be respected")
# Also suppress all FutureWarnings from sklearn
warnings.filterwarnings("ignore", category=FutureWarning)

# Then import other modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adaptiveforecast import AdaptiveForecaster
from sktime.datasets import load_airline

# Load the airline passengers dataset
df = load_airline()

# Advanced usage with algorithm-specific transformations and grid search
print("\n\n=== ADVANCED USAGE WITH ALGORITHM-SPECIFIC TRANSFORMATIONS AND GRID SEARCH ===")

# Define custom parameter grids for each algorithm and transformers
param_grid = {
    'naive': {'forecaster__strategy': ['last', 'mean', 'drift'], 'forecaster__window_length': [3, 6, 12]},
    'arima': {'forecaster__d': [0, 1], 'forecaster__max_p': [2, 3], 'forecaster__max_q': [2, 3]},
    'ets': {'forecaster__error': ['add', 'mul'], 'forecaster__trend': ['add', 'mul', None]},
    'impute': {'impute__method': ['mean', 'median', 'ffill', 'bfill', 'drift']},
    'deseasonalize': {'deseasonalize__model': ['additive', 'multiplicative']},
    'detrend': {'detrend__model': ['linear']}
}

# Define cross-validation parameters
cv_params = {
    'method': 'expanding',
    'step': 12
}

# Create and use the forecaster with algorithm-specific transformations
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    advanced_forecaster = AdaptiveForecaster(
    df=df,
    forecast_horizon=12,
    test_size=24,
    algorithms=['naive', 'arima', 'ets'],
    transformations={
        'naive': [{'name': 'impute', 'method': 'mean'}],  # Using dictionary with parameters
        'arima': ['deseasonalize', 'detrend'],
        'ets': ['deseasonalize', 'impute']  # Using string (default parameters)
    },
    seasonal_period=12,
    grid_search=param_grid,
    cross_validation=cv_params,
    scoring="mape"
    )

    advanced_forecaster.fit()
    advanced_forecaster.predict()

from adaptiveforecast.mlflow_utils import log_forecaster

# Log the fitted forecaster to MLflow
log_forecaster(
    forecaster=advanced_forecaster,
    experiment_name='AirlinePassengerForecast',
    model_name='AirlinePassenger'
)
# Get summary of results
print("\nAdvanced Forecast Summary:")
best_model = advanced_forecaster.best_model
best_params = advanced_forecaster.get_best_params()
print(f"Best model: {best_model}")
print(f"Best parameters: {best_params}")

# Get metrics for all models
for model, metrics in advanced_forecaster.metrics.items():
    print(f"\n{model.upper()} metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper()}: {value:.4f}")

# Plot forecasts
plt.figure(figsize=(12, 6))
advanced_forecaster.plot_forecasts(models=[advanced_forecaster.best_model])
plt.title("Advanced Forecast with Algorithm-Specific Transformations")
plt.tight_layout()
plt.savefig("advanced_forecast.png")
plt.close()

# Get forecast DataFrame
forecast_df = advanced_forecaster.get_forecast_dataframe()
print("\nForecast DataFrame:")
print(forecast_df.head())

# Save summary to file
advanced_forecaster.save_summary("advanced_forecast_summary.json")
print("\nSummary saved to advanced_forecast_summary.json")
