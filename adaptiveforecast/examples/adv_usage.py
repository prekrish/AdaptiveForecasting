import warnings
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", message="The user-specified parameters provided alongside auto=True in AutoETS may not be respected")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adaptiveforecast.core import AdaptiveForecaster
from sktime.datasets import load_airline

# Load the airline passengers dataset
df = load_airline()
# Advanced usage with grid search and MultiplexForecaster
print("\n\n=== ADVANCED USAGE WITH GRID SEARCH AND MULTIPLEXFORECASTER ===")
# Define custom parameter grids for each algorithm
param_grid = {
    'naive': {'strategy': ['last', 'mean', 'drift']}
    #'arima': {'d': [0, 1], 'max_p': [2, 3], 'max_q': [2, 3]},
    #'ets': {'error': ['add', 'mul'], 'trend': ['add', 'mul', None], 'damped_trend': [True, False]}
}

# Define cross-validation parameters
cv_params = {
    'method': 'expanding',
    'initial': 36,
    'step': 12
}

# Create and use the forecaster with grid search and MultiplexForecaster
advanced_forecaster = AdaptiveForecaster(
    df=df,
    forecast_horizon=12,
    test_size=24,
    algorithms=['naive', 'arima', 'ets'],
    transformations=['deseasonalize'],
    seasonal_period=12,
    grid_search=param_grid,
    cross_validation=cv_params,
    use_multiplex=False,
    scoring="mape",
    selected_forecasters=['ets','naive']
)

# Fit models with grid search
advanced_forecaster.fit()

# Print cross-validation results
advanced_forecaster.print_cv_results(n_best=10)

# Generate forecasts
advanced_forecaster.predict()

# Get summary of results
advanced_summary = advanced_forecaster.get_summary()
print("\nAdvanced Forecast Summary:")
print(advanced_summary)

# Plot forecasts
plt.figure(figsize=(12, 6))
advanced_forecaster.plot_forecasts(models=['multiplex', 'best'])
plt.title("Advanced Forecast with Grid Search")
plt.tight_layout()
plt.savefig("advanced_forecast.png")
plt.close()
