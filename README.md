# AdaptiveForecaster

A user-friendly interface for time series forecasting using sktime.

## Description

AdaptiveForecaster provides a simple interface for users to perform time series forecasting by abstracting away the complexities of the underlying algorithms and processes. It leverages the powerful sktime library to provide a variety of forecasting methods, automated parameter tuning, and ensemble models.

## Features

- Easy-to-use interface for time series forecasting
- Support for multiple forecasting algorithms:
  - Naive forecasting
  - ARIMA models
  - ETS models
  - Theta forecasting
  - Exponential smoothing
- Automatic model selection and hyperparameter tuning
- Time series transformations:
  - Deseasonalization
  - Detrending
  - Box-Cox transformation
  - Lag features
  - Missing value imputation
- Cross-validation with expanding or sliding windows
- Model evaluation with various metrics
- MLflow integration for experiment tracking
- Visualization of forecasts and model performance

## Installation

```bash
# Basic installation
pip install adaptiveforecast

# With MLflow support
pip install adaptiveforecast[mlflow]

# For developers
pip install adaptiveforecast[dev]
```

## Quick Start

```python
import pandas as pd
from adaptiveforecast import AdaptiveForecaster

# Load time series data
data = pd.read_csv('your_data.csv', parse_dates=['date_column'], index_col='date_column')

# Initialize the forecaster
forecaster = AdaptiveForecaster(
    df=data,
    target='target_column',
    forecast_horizon=12,  # Forecast 12 steps ahead
    algorithms=['naive', 'arima', 'ets'],  # Use these algorithms
    transformations=['deseasonalize'],  # Apply deseasonalization
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

# Get the best model
best_model = forecaster.best_model
best_params = forecaster.get_best_params()
print(f"Best model: {best_model}")
print(f"Best parameters: {best_params}")

# Save the forecast summary
forecaster.save_summary('forecast_results.json')
```

## Documentation

For detailed documentation, see the [API Reference](docs/api_reference.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.