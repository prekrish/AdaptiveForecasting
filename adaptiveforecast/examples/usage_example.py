import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", message="The user-specified parameters provided alongside auto=True in AutoETS may not be respected")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adaptiveforecast.core import AdaptiveForecaster
from sktime.datasets import load_airline
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.transformations.series.detrend import Detrender,Deseasonalizer
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.lag import Lag
from sktime.transformations.series.impute import Imputer    
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.model_selection import ExpandingWindowSplitter
# Load the airline passengers dataset
df = load_airline()
# Advanced usage with grid search and MultiplexForecaster
print("\n\n=== ADVANCED USAGE WITH GRID SEARCH AND MULTIPLEXFORECASTER ===")

seasonal_period = 12
algorithms=['naive','arima','exp_smoothing']
transformations={
    'naive':None,
    'arima':['deseasonalize','detrend'],
    'exp_smoothing':['impute']
}
# Define custom parameter grids for each algorithm
parameters = {
    'naive': {'strategy': ['last', 'mean', 'drift'],'window_length':[3,6,12],'sp':[seasonal_period]},
    'arima': {'d': [0, 1], 'max_p': [2, 3], 'max_q': [2, 3],'sp':[seasonal_period]},
    'exp_smoothing': {'seasonal': ['add', 'mul'], 'trend': ['add', 'mul', None], 'damped_trend': [True, False]},
    'impute':{'method':['mean','median','ffill','bfill','drift']}
}

# Define cross-validation parameters
cv_params = {
    'method': 'expanding',
    'initial': 5,
    'step': 3
}

models = {
        'naive':NaiveForecaster(),
        'arima':AutoARIMA(suppress_warnings=True),
        'ets':AutoETS(),
        'exp_smoothing':ExponentialSmoothing()
}

transformers = {
    'deseasonalize':Deseasonalizer(sp=seasonal_period),
    'detrend':Detrender(),
    'boxcox':BoxCoxTransformer(),
    'lag':Lag(lags=[1,2,3,4,5,6]),
    'impute':Imputer()
}

cv = ExpandingWindowSplitter(
    initial_window=24,
    step_length=12,
    fh=[1,2,3])

fittedmodels=[]
for algoname in algorithms:
    param_grid = {}
    transformerslist = []
    if transformations[algoname]:
        
        for transform in transformations[algoname]:
            transformerslist.append(
                (transform, transformers[transform])
            )
            if transform in parameters.keys():
                for key,value in parameters[transform].items():
                    param_grid[transform+'__'+key]=value
    #print(param_grid)
    print("--------------{}------------------".format(algoname))
    param_grid['forecaster']=[models[algoname]]
    for key,value in parameters[algoname].items():
        param_grid['forecaster__'+key]=value
    transformerslist.append(('forecaster',models[algoname]))
    pipe = TransformedTargetForecaster(steps=transformerslist)
    print(transformerslist,param_grid)
    if not bool(param_grid):
        pipe.fit(df)
        y_pred = pipe.predict(fh=[1,2,3])
    else:
        gscv = ForecastingGridSearchCV(
            forecaster=pipe,param_grid=param_grid,cv=cv)
        gscv.fit(df)
        fittedmodels.append(gscv.best_forecaster_)
        y_pred = gscv.predict(fh=[1,2,3])
    