#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import autokeras as ak
import datetime as dt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import time
import warnings

from autokeras.preprocessors.common import AddOneDimension
from data import get_data
from keras.saving import register_keras_serializable
from math import sqrt
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

register_keras_serializable()(AddOneDimension)

warnings.filterwarnings("ignore")


def create_dataset_modified(data, lookback, forecast):

    data1 = data[data.columns[0]].values

    X = []
    
    y = []

    n_total = len(data1)  # total number of days

    n_days = n_total-lookback-forecast

    for i in range(n_days):

        aux = data1[i:i+lookback]

        aux = aux.ravel()

        X.append(np.array(aux))

        y.append(data1[i+lookback+forecast-1])

    return np.array(X), np.array(y)


def akeras(df, forecast, lag,time_scale, run):
    
    print("Run:",run,", Estação:", df.columns[0], ", Horizonte:", forecast)
    
    _start = time.perf_counter()

    train, test = train_test_split(df, test_size = 0.2, shuffle = False)

    Xtrain, Ytrain = create_dataset_modified(train, lag, forecast)
    
    Xtest, Ytest   = create_dataset_modified(test,  lag, forecast)

    resize = lag + forecast - 1

    date = test.iloc[resize:-1, :].index.values

    # Creating the AutoKeras model for regression
    model = ak.AutoModel(inputs=ak.Input(),
                         outputs=ak.RegressionHead(),
                         max_trials=1000,
                         overwrite=True,
                         #project_name=f"{df.columns[0]}_{TEMPO}_{forecast}_{lag}_{run}",
                         project_name="autoKerasTrial",
                         metrics=['root_mean_squared_error']
                         )

    model.fit(Xtrain, 
              Ytrain,
              epochs=500,
              batch_size=128,
              verbose=0)
    
    best_model = model.export_model()
    
    best_model_config = best_model.get_config()

    train_predictions = model.predict(Xtrain,
                                      verbose=0)
    
    test_predictions  = model.predict(Xtest,
                                      verbose=0)

    _end = time.perf_counter()
    
    print("Time:", dt.timedelta(seconds=_end - _start))
    
    results = {'date'                   : date,
               'decomposition_level'    : None,
               'forecast'               : forecast,
               "framework"              : "AutoKeras",
               'heuristic'              : None,
               'heuristic_evolution'    : None,
               'iteration'              : None,
               'Lj'                     : None,
               'look_back'              : lag,
               'mape'                   : metrics.mean_absolute_percentage_error(Ytest, test_predictions) * 100,
               'mape_mean_he'           : None,
               'mape_mean_le'           : None,
               'model'                  : best_model_config['name'],
               'model_params'           : best_model_config,
               'observed'               : Ytest,
               'population'             : None,
               'predicted'              : test_predictions,
               'rmse'                   : sqrt(metrics.mean_absolute_error(Ytest, test_predictions)),
               'run'                    : run,
               'run_time'               : dt.timedelta(seconds=_end - _start),
               'r2'                     : metrics.r2_score(Ytest, test_predictions),
               'station'                : df.columns[0],
               'time_scale'             : time_scale,
               'test_time'              : dt.timedelta(seconds=_end - _start),
               'train_observed'         : Ytrain,
               'train_predicted'        : train_predictions,
               'train_time'             : dt.timedelta(seconds=_end - _start),
               'wavelet'                : False,
               'wavelet_filter'         : None,
               }

    results_t = {key: list([value]) for key, value in results.items()}

    df_results = pd.DataFrame(results_t)

    df_results.to_pickle('./pkl/' +
                         str(run) +
                         '_autoKeras'  +
                         "_FORECAST_"  +
                         str(forecast) +
                         "_"+
                         str(time_scale) +
                         "_DATABASE_" +
                         df.columns[0] +
                         "_" +
                         time.strftime("%Y_%m_%d_%Hh_%Mm_%S") +
                         '.pkl')

  
# Function that runs each configuration
def job_runner(cfg):
    database, file, time_scale, forecast, lag ,run= cfg
    df_list = get_data(database, file, time_scale)  # search the data
    akeras(df_list, forecast, lag,time_scale, run)  # performs processing
   

if __name__ == "__main__":
    
    RUNS = 20
    
    CONFIG = [("stations", "44200000", "daily", 1, 2),    # config found in wann
              ("stations", "44200000", "daily", 7, 8),
              ("stations", "44200000", "daily", 21, 47),

              ("stations", "44200000", "daily", 1, 7),    # max lags possible
              ("stations", "44200000", "daily", 7, 49),
              ("stations", "44200000", "daily", 21, 147),

              ("stations", "44200000", "monthly", 1, 4),  # config found in wann
              ("stations", "44200000", "monthly", 6, 7),
              ("stations", "44200000", "monthly", 12, 14),

              ("stations", "44200000", "monthly", 1, 6),  # max lags possible
              ("stations", "44200000", "monthly", 6, 36),
              ("stations", "44200000", "monthly", 12, 72),

              ("basins", "PARAIBA_DO_SUL_ANTA", "hourly", 3, 4), # config found in wann
              ("basins", "PARAIBA_DO_SUL_ANTA", "hourly", 12, 13),
              
              ("basins", "PARAIBA_DO_SUL_ANTA", "hourly", 3, 72),  # max lags possible
              ("basins", "PARAIBA_DO_SUL_ANTA", "hourly", 12, 288),]
    
    CONFIG_RUNS = [config + (run,) for config in CONFIG for run in range(RUNS)]

    workers = 1
    
    if workers == 1 :
        for config in CONFIG_RUNS:

            job_runner(config)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(job_runner, config) for config in CONFIG_RUNS]
       

    