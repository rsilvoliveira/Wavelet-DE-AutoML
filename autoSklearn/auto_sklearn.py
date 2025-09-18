#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import datetime as dt
import numpy as np
import pandas as pd
import sklearn.metrics
import autosklearn.regression
import sklearn.metrics as metrics
from math import sqrt
from data import get_data
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor


LOG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '[%(levelname)s] [%(asctime)s:%(name)s] %(message)s'
        }
    },
    'handlers': {
        'file_handler': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'filename': 'autosklearn.log'
        },
        'distributed_logfile': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'filename': 'distributed.log'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['file_handler']
    },
    'loggers': {
        'autosklearn.metalearning': {
            'level': 'DEBUG',
            'handlers': ['file_handler']
        },
        'autosklearn.automl_common.utils.backend': {
            'level': 'DEBUG',
            'handlers': ['file_handler'],
            'propagate': False
        },
        'smac.intensification.intensification.Intensifier': {
            'level': 'INFO',
            'handlers': ['file_handler']
        },
        'smac.optimizer.local_search.LocalSearch': {
            'level': 'INFO',
            'handlers': ['file_handler']
        },
        'smac.optimizer.smbo.SMBO': {
            'level': 'INFO',
            'handlers': ['file_handler']
        },
        'EnsembleBuilder': {
            'level': 'DEBUG',
            'handlers': ['file_handler']
        },
        'distributed': {
            'level': 'DEBUG',
            'handlers': ['distributed_logfile']
        }
    }
}


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


def main(df,forecast,lag,time_scale,run):

    np.random.seed(run)
    
    print("Run:",run,", Estação:", df.columns[0], ", Horizonte:", forecast)
    
    _start = time.perf_counter()

    train, test = train_test_split(df, test_size=0.2, shuffle=False)

    Xtrain, Ytrain = create_dataset_modified(train, lag, forecast)
    
    Xtest, Ytest = create_dataset_modified(test, lag, forecast)

    resize = lag + forecast - 1

    date = test.iloc[resize:-1, :].index.values

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=86400,  # (24 hrs)
        # per_run_time_limit=30,
        # include={"regressor" : ['mlp',
        #                         'gradient_boosting', 
        #                         'libsvm_svr',
        #                         'adaboost', 
        #                         'ard_regression',
        #                         'decision_tree', 
        #                         'extra_trees',
        #                         'gaussian_process',
        #                         'gradient_boosting',
        #                         'k_nearest_neighbors',
        #                         'liblinear_svr',
        #                         'libsvm_svr',
        #                         'mlp',
        #                         'random_forest',
        #                         'sgd']
        #         "data_preprocessor" : ["feature_type"]},
        # exclude={"data_preprocessor"  : ["no_preprocessing"],
        #          "feature_preprocessor": ["no_preprocessing"]},
        n_jobs=4,
        memory_limit=1024 * 6,
        #logging_config=LOG
    )

    _start_fit = time.perf_counter()
    
    automl.fit(Xtrain, Ytrain, dataset_name=df.columns[0])
    
    _end_fit = time.perf_counter()

    _start_train = time.perf_counter()

    train_predictions = automl.predict(Xtrain)
    
    test_predictions = automl.predict(Xtest)
            
    _end = time.perf_counter()
    
    leaderboard = automl.leaderboard()

    # Separate the model with the highest weight
    best_model_info = automl.show_models()[leaderboard.index[0]]
    
    # Model name and its settings
    model_name = best_model_info["sklearn_regressor"].__class__.__name__
    
    params = best_model_info["sklearn_regressor"].get_params()
    
    results = {"station"                : df.columns[0],
               "time"                   : time_scale,
               "forecast"               : forecast,
               "run"                    : run,
               "wavelet"                : "Sklearn-AutoML",
               "look_back"              : lag,
               "model"                  : model_name,
               "model_params"           : params,
               "mape"                   : metrics.mean_absolute_percentage_error(Ytest, test_predictions) * 100,
               "r2"                     : metrics.r2_score(Ytest, test_predictions),
               "rmse"                   : sqrt(metrics.mean_absolute_error(Ytest, test_predictions)),
               "mape_mean_he"           : None,
               "mape_mean_le"           : None,
               "wavelet_filter"         : None,
               "decomposition_level"    : None,
               "Lj"                     : None,
               "heuristic"              : None,
               "heuristic_evolution"    : None,
               "iteration"              : None,
               "population"             : None,
               "predicted"              : test_predictions,
               "observed"               : Ytest, 
               'date'                   : date,
               'train_time'             : dt.timedelta(seconds=_end_fit - _start_fit), 
               'test_time'              : dt.timedelta(seconds=_end - _start_train), 
               'train_predicted'        : train_predictions,
               'train_observed'         : Ytrain, 
               'run_time'               : dt.timedelta(seconds=_end - _start)
               }

    results_t = {key: list([value]) for key, value in results.items()}

    df_results = pd.DataFrame(results_t)

    df_results.to_pickle('./pkl/' +
                         str(run) +
                         '_autoSklearn'  +
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
    main(df_list, forecast, lag,time_scale, run)    # performs processing
    

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
            futures = [executor.submit(job_runner, config) for config in CONFIG_RUNS[:2]]
       
           
    

