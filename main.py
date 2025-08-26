#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:05:07 2024

@author: rodrigo
"""
import concurrent.futures
import itertools
import logging
import os
import pickle
import pywt
import sys 
import time  
import warnings

import datetime as dt
import numpy as np
import pandas as pd

from math import sqrt
from pyextremes import EVA
from sklearn.metrics import (r2_score,
                             mean_squared_error,
                             mean_absolute_percentage_error as cmape)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from termcolor import colored
from typing import List, Tuple

from data import get_data
from heuristics import WANN, get_heuristica
from regressors.regressors import get_estimator
from wavelet_ml_config import config, get_positions,get_filters,get_models,get_params

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["GRPC_VERBOSITY"] = "NONE"

warnings.filterwarnings('ignore')

# Logger configuration
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# =============================================================================
# 
# =============================================================================
def MAPE(original,forecasts):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between forecasted and actual values.

    :param forecasts: Array-like, forecasted values.
    :param original: Array-like, actual observed values.
    :return: Mean Absolute Percentage Error (MAPE) between the forecasts and original values.
    :rtype: float
    """
   
    return cmape(original, forecasts) * 100


def MAPE_all(forecasts, original):
    """
    Computes the Mean Absolute Percentage Error (MAPE) for each forecast value.

    The MAPE is calculated as:
        MAPE_i = 100 * |original[i] - forecast[i]| / original[i]

    Args:
        forecasts (array-like): Sequence of forecasted values.
        original (array-like): Sequence of true/original values. Must be the same length as forecasts.

    Returns:
        numpy.ndarray: Array containing the MAPE for each element.
    """
    mape_all = []
    
    for i in range(len(forecasts)):
    
        mape_all.append(100*(abs(original[i]-forecasts[i])/original[i]))
    
    mape_all = np.array(mape_all)

    return mape_all

# =============================================================================
# 
# =============================================================================
def create_dataset_modified(data: pd.DataFrame, lookback: int, forecast: int):
    """
    Creates a dataset for supervised learning from a time series.

    Given a time series, it generates input sequences of length `lookback`
    and corresponding target values at a `forecast` step ahead.

    Args:
        data (pandas.DataFrame): Input time series with at least one column.
        lookback (int): Number of past time steps to use as features.
        forecast (int): Number of steps ahead to forecast.

    Returns:
        tuple:
            X (numpy.ndarray): 2D array of shape (n_samples, lookback),
                where each row contains a sequence of past values.
            y (numpy.ndarray): 1D array of shape (n_samples,),
                where each element is the target value at `forecast` steps ahead.
    """
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


# =============================================================================
# 
# =============================================================================
def create_dataset_modified_wann(data: pd.DataFrame, coeffs: List[Tuple[np.ndarray, np.ndarray]], lookback, forecast) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a dataset for training/testing a Wavelet Neural Network (WANN).

    The function builds input sequences that include:
    - A sliding window of the original time series (`data`)
    - Corresponding wavelet coefficients (approximation and detail) from multiple decomposition levels

    Each sample is constructed using a `lookback` window, and the target (`y`)
    corresponds to the value of the original series `forecast` steps ahead.

    Args:
        data (pandas.DataFrame): Input time series with at least one column.
            Only the first column will be used.
        coeffs (list): List of tuples `(approximation, detail)` where each element
            is a NumPy array representing wavelet decomposition coefficients.
        lookback (int): Number of past time steps to use as input features.
        forecast (int): Number of steps ahead to forecast.

    Returns:
        tuple:
            X (numpy.ndarray): 2D array of shape (n_samples, n_features),
                where each row contains concatenated features
                (original values + wavelet coefficients).
            y (numpy.ndarray): 1D array of shape (n_samples,),
                containing the target values `forecast` steps ahead.

    Raises:
        TypeError: If inputs are not of the expected type.
        ValueError: If lookback/forecast are not positive, or if `data` has no rows.
    """
    X = []
    y = []

    n_total = len(coeffs[0][0])  # total number of days
    
    n_days = n_total-lookback-forecast

    for i in range(n_days):

        aux = list(data[i:i+lookback])

        for c in coeffs:  # levels

            for k in range(2):  # approximation and detail

                aux.extend(c[k][i:i+lookback])

        X.append(np.array(aux))

        y.append(data[i+lookback+forecast-1])
        
    return np.array(X), np.array(y)


# =============================================================================
# 
# =============================================================================
def wavelets(df: pd.DataFrame, dec_level: int, wavelet_filter: str, look_back: int, forecast: int,save: bool):
    """
    Performs undecimated wavelet transform (UWT) on a time series DataFrame and prepares
    training and testing datasets for a Wavelet Neural Network (WANN).

    The function:
    - Splits the input DataFrame into train and test sets.
    - Decomposes each series using the specified wavelet filter up to `dec_level`.
    - Removes boundary-affected coefficients.
    - Constructs supervised learning datasets (`X`, `Y`) using a sliding window
    of `look_back` steps and forecasting horizon `forecast`.
    - Optionally returns the preprocessed test set and boundary offset (`Lj`) if `save` is True.

    Args:
        df (pandas.DataFrame): Input time series with one or more columns.
        dec_level (int): Level of wavelet decomposition.
        wavelet_filter (str): Name of the wavelet filter to use.
        look_back (int): Number of past time steps used as input features.
        forecast (int): Number of steps ahead to forecast.
        save (bool): If True, also return the adjusted test DataFrame and Lj offset.

    Returns:
        tuple:
            If save is False:
                trainX (numpy.ndarray): 2D array of training inputs.
                trainY (numpy.ndarray): 1D array of training targets.
            If save is True:
                testX (numpy.ndarray): 2D array of testing inputs.
                testY (numpy.ndarray): 1D array of testing targets.
                test_lj (pandas.DataFrame): Adjusted test DataFrame after removing Lj values.
                Lj (int): Number of boundary-affected coefficients removed.

    Raises:
        Returns (np.inf, np.inf) in cases of invalid input sizes or if the look_back
        window is too large relative to train/test sets.
    """
    
    train, test = train_test_split(df, test_size=0.2,shuffle=False)
    
    # new train and test size because of IWT
    train_size = (2**dec_level)*int(len(train)/(2**dec_level))
    
    test_size = (2**dec_level)*int(len(test)/(2**dec_level))
    
    train, test = df[0:train_size], df[train_size:train_size+test_size]
    
    if len(train)==0 or len(test) == 0:
        
        return np.inf , np.inf 
    
    coefs_train = []
    
    for column in train:
        # Decompose training data using undecimated wavelet transform
        aux_train = pywt.swt(np.ravel(train[column]), wavelet_filter, level=dec_level)
    
        coefs_train.append(aux_train)

    coefs_test = []
    
    for column in test:
       
        aux_test = pywt.swt(np.ravel(test[column]), wavelet_filter, level=dec_level)
        
        coefs_test.append(aux_test)
        
    w = pywt.Wavelet(wavelet_filter)
   
    filter_len = w.dec_len
    
    Lj = ((2**dec_level)-1)*(filter_len-1)+1
    
    new_coefs_train = []
    
    for item in coefs_train:
       
        aux = []
       
        for i in range(dec_level):
            
            new_cA_train = item[i][0][Lj:]
       
            new_cD_train = item[i][1][Lj:]
           
            aux.append([new_cA_train, new_cD_train])

        new_coefs_train.append(aux)

    new_coefs_test = []
    
    # remove boundary affected coef-s from the test
    for item in coefs_test:
        
        aux = []
       
        for i in range(dec_level):
            
            new_cA_test = item[i][0][Lj:]
        
            new_cD_test = item[i][1][Lj:]
            
            aux.append([new_cA_test, new_cD_test])
        
        new_coefs_test.append(aux)
        
    # removing the first Lj values from the begining of the TEST and TRAIN series
    test_lj = test.iloc[Lj:, 0:1]
   
    test_lj_raw = test.iloc[Lj:, 0:1].values
    
    train_lj = train.iloc[Lj:, 0:].values

    trainX, trainY = create_dataset_modified_wann(data     = train_lj[:, 0],
                                                  coeffs   = new_coefs_train[0],
                                                  lookback = look_back,
                                                  forecast = forecast)
    
    testX, testY = create_dataset_modified_wann(data     = test_lj_raw[:, 0],
                                                coeffs   = new_coefs_test[0],
                                                lookback = look_back,
                                                forecast = forecast)
    
    if save:
        
        return testX, testY, test_lj, Lj
    
    else:
                
        if look_back >= test_size or look_back >= train_size:
            return np.inf , np.inf 
        
        if len(trainX) == 0 :
            
            return np.inf , np.inf 
        
        if len(trainY) == 0 :
            
            return np.inf , np.inf 
        
        if len(testX) == 0 :
            
            return np.inf , np.inf 
        
        if len(testY) == 0 :
            
            return np.inf , np.inf 
        
        elif len(testX)-Lj<(forecast+1)*2+(forecast+1):
        # elif len(testX) - (2 ** look_back - 1) * (filter_len - 1) - 3 * forecast - 4 <= 0:
            
            return np.inf , np.inf 
        
        else:
        
            return trainX, trainY
    
    
# =============================================================================
#     
# =============================================================================
def get_extremes(model, extremes_type, mean):
    """
    Extracts extreme values from a time series using the Peaks Over Threshold (POT) method.

    This function applies the POT method on a fitted EVA model to retrieve either
    high or low extremes, using a specified threshold. The extremes are returned
    as stored in the model's `extremes` attribute.

    Args:
        model (EVA): A pyextremes EVA model object fitted to the time series.
        extremes_type (str): Type of extremes to retrieve; typically "high" or "low".
        mean (float): Threshold value for determining extremes, usually the mean of the series.

    Returns:
        pandas.DataFrame: DataFrame containing the extracted extreme values along with their timestamps.
    
    Example:
        >>> from pyextremes import EVA
        >>> model = EVA(data=df['vaz_pred'])
        >>> threshold = df['vaz_pred'].mean()
        >>> high_ex = get_extremes(model, "high", threshold)
        >>> low_ex = get_extremes(model, "low", threshold)
    """    
    model.get_extremes(method        = "POT",
                       extremes_type = extremes_type,
                       threshold     = mean,
                       r             = "24H")

    return model.extremes


# =============================================================================
# 
# =============================================================================
def calculate_mape(extreme):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) for extreme events.

    This function computes the MAPE between predicted and observed values 
    in the input DataFrame `extreme`, merges the results as a new column, 
    sorts by index, and returns the mean MAPE.

    Args:
        extreme (pandas.DataFrame): DataFrame containing at least the columns:
            - 'vaz_pred': predicted values
            - 'vaz_test': observed/true values
            - 'date-time': timestamps for merging and sorting

    Returns:
        float: Mean MAPE over all entries in the DataFrame.

    Example:
        >>> mean_mape = calculate_mape(extreme_df)
    """
    mape = MAPE_all(extreme['vaz_pred'],
                    extreme['vaz_test']).tolist()
    
    ts = pd.merge(extreme,
                  pd.Series(mape,
                            index = extreme.index,
                            name  = 'mape'),
                  on=["date-time"])

    mape_extreme = ts.sort_index()

    #max_mape = mape_extreme['mape'].max()
    mean_mape = mape_extreme['mape'].mean()
    
    return mean_mape

    
# =============================================================================
# 
# =============================================================================
def analysis(look_backs,forecast,data,testY,testPredict):
    """
    Performs evaluation of a prediction model, including general and extreme-event metrics.

    This function:
    - Aligns predicted and observed data based on the look-back and forecast horizon.
    - Calculates standard error metrics: MAPE, R², RMSE.
    - Identifies high and low extremes using Peaks Over Threshold (POT) via pyextremes EVA.
    - Computes the mean MAPE for high and low extreme events.

    Args:
        look_backs (int): Number of past time steps used as input features.
        forecast (int): Forecast horizon (number of steps ahead to predict).
        data (pandas.DataFrame): Original time series DataFrame.
        testY (array-like): Observed/true values for the test set.
        testPredict (array-like): Predicted values corresponding to `testY`.

    Returns:
        tuple:
            mean_mape_he (float): Mean MAPE for high extremes.
            mean_mape_le (float): Mean MAPE for low extremes.
            mape1 (float): Overall MAPE between `testY` and `testPredict`.
            r2 (float): R² score of the predictions.
            rmse (float): Root Mean Squared Error of the predictions.
            date (numpy.ndarray): Array of datetime indices corresponding to aligned test data.

    Example:
        >>> mean_mape_he, mean_mape_le, mape1, r2, rmse, date = analysis(
        >>>     look_backs=10,
        >>>     forecast=1,
        >>>     data=df,
        >>>     testY=testY,
        >>>     testPredict=testPredict
        >>> )
    """
    resize = look_backs + forecast - 1
    
    data_aux = data.iloc[resize:-1, :]
    
    date = data_aux.index.values
    
    mape1 = MAPE(testY,testPredict)
    
    r2 = r2_score(testY, testPredict)
    
    mse = mean_squared_error(testY, testPredict)
    
    rmse = sqrt(mse)

    df = pd.DataFrame()
    
    df['vaz_pred'] = testPredict
    
    df['vaz_test'] = testY
    
    df.index = pd.DatetimeIndex(date, name='date-time')

    # calculating low and high extremes

    model = EVA(data=df['vaz_pred'])
   
    mean = df['vaz_pred'].mean()
    
    high_ex = get_extremes(model, "high", mean)

    low_ex = get_extremes(model, "low", mean)
    
    vaz_test = df['vaz_test']

    join_vaz_he = pd.merge(vaz_test, high_ex, on=["date-time"], )
    
    join_vaz_le = pd.merge(vaz_test, low_ex, on=["date-time"], )

    mean_mape_he = calculate_mape(join_vaz_he)
    
    mean_mape_le = calculate_mape(join_vaz_le)
    
    return mean_mape_he,mean_mape_le, mape1,r2, rmse, date

    
# =============================================================================
# 
# =============================================================================
def wann(cfg, *args):
    """
    Trains and evaluates a Wavelet-Augmented Neural Network (WANN) or a standard model
    on a given time series dataset, optionally using wavelet decomposition.

    This function:
    - Parses the configuration array `cfg` to determine preprocessing, model type,
      wavelet usage, decomposition level, and estimator parameters.
    - Prepares input and output datasets using either wavelet-transformed features
      or standard sliding windows.
    - Initializes the estimator/model with the specified parameters.
    - Performs training and prediction on training and test sets.
    - Measures training and testing time.
    - Computes evaluation metrics including overall MAPE, extreme-event MAPE,
      R² score, and RMSE.
    - Returns a dictionary summarizing results, predictions, metrics, and parameters.

    Args:
        cfg (array-like): Configuration vector specifying:
            - cfg[0]: use_wavelet (0 or 1)
            - cfg[1]: wavelet filter index
            - cfg[2]: look-back window
            - cfg[3]: decomposition level
            - cfg[4]: estimator/model index
            - remaining entries: estimator hyperparameters
        *args: Variable-length argument list containing:
            - df (pandas.DataFrame): Input time series data.
            - forecast (int): Forecast horizon.
            - save (bool): If True, include additional outputs (e.g., Lj, test dataset).

    Returns:
        dict: Dictionary containing:
            - 'wavelet': bool, whether wavelet decomposition was used
            - 'mape_mean_he': Mean MAPE for high extremes
            - 'mape_mean_le': Mean MAPE for low extremes
            - 'modelo': Estimator/model name
            - 'model_params': Model hyperparameters
            - 'wav_filter': Wavelet filter used
            - 'level': Decomposition level
            - 'lb': Look-back window
            - 'Lj': Number of boundary-affected coefficients removed
            - 'mape': Overall MAPE
            - 'r2': R² score
            - 'rmse': Root Mean Squared Error
            - 'pred': Test predictions (list)
            - 'obs': Test observations (list)
            - 'date': Corresponding dates (list)
            - 'train_time': Training time in seconds
            - 'test_time': Testing time in seconds
            - 'train_pred': Training predictions (list)
            - 'train_obs': Training observations (list)

    Example:
        >>> cfg = [1, 2, 10, 3, 0, 0.01, 100]
        >>> results = wann(cfg, df, forecast=1, save=False)
        >>> print(results['mape'], results['r2'])
    """

    df, forecast, save = args
    
    wavelet = round(cfg[0])
    
    wavelet_filter = round(cfg[1])
    
    look_backs = round(cfg[2])
    
    dec_level = round(cfg[3])
    
    estimator_name = round(cfg[4])
    
    wavelet_filter = get_filters(wavelet_filter)
    
    estimator_name = get_models(estimator_name)
    
    estimator_params_idx = get_positions(estimator_name)
    
    estimator_params = cfg[estimator_params_idx]
    
    estimator_params_names = get_params(estimator_name)
    
    # Wavelet
    if wavelet:
        
        trainX, trainY = wavelets(df, dec_level, wavelet_filter, look_backs, forecast, False)
    
    else:
        train, test = train_test_split(df, test_size=0.2,shuffle=False)
    
        trainX, trainY = create_dataset_modified(train, look_backs, forecast)
    
    if np.isscalar(trainX): return np.inf
    
    estimator = get_estimator(estimator_params, estimator_params_names, estimator_name)
    
    if not save:
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        try:
            score = cross_val_score(estimator, trainX, trainY, cv=tscv,verbose=0)
        except:
            return np.inf
        
        score = np.mean(score)
        
        return score
    
    if not wavelet:
        
        testX, testY = create_dataset_modified(test, look_backs, forecast)
        
    else:        
        testX, testY , test_lj, Lj= wavelets(df, dec_level, wavelet_filter, look_backs, forecast, True)
    
    estimator.fit(trainX, trainY)
    
    pred = estimator.predict(trainX)
    
    _start = time.perf_counter()
    
    estimator.fit(trainX, trainY)
    
    train_time = time.perf_counter() - _start
    
    _start = time.perf_counter()
    
    testPredict = estimator.predict(testX)
    
    if estimator_name == "LSTM" or estimator_name == "CONV_LSTM":
       
        testPredict = testPredict.flatten()
        
    test_time = time.perf_counter() - _start
    
    testPredict = np.maximum(testPredict, 0)
            
    if not wavelet:
        
        Lj = 0
       
        data = test
    
    else:
    
        data = test_lj
    
    mean_mape_he,mean_mape_le, mape1,\
    r2, rmse, date = analysis(look_backs, forecast, data, testY, testPredict)
    
    params = estimator._convert_params()
    
    results = {'wavelet' :      bool(round(wavelet)),
               'mape_mean_he' : mean_mape_he,
               'mape_mean_le' : mean_mape_le,
               'modelo' :       estimator_name, 
               'model_params' : params,
               'wav_filter' :   wavelet_filter,
               'level' :        dec_level,
               'lb' :           look_backs,
               'Lj' :           Lj,
               'mape' :         mape1, 
               'r2' :           r2, 
               'rmse' :         rmse, 
               'pred' :         testPredict.tolist(), 
               'obs' :          testY.tolist(), 
               'date' :         date.tolist(),
               'train_time' :   train_time,
               'test_time' :    test_time,
               'train_pred':    pred.tolist(),
               'train_obs':     trainY.tolist()}
    
    return results


def evolution(evo):
    """
    Converts a list of configuration vectors into a summarized pandas DataFrame.

    Each configuration vector in `evo` contains settings for a model or experiment,
    including wavelet usage, wavelet filter, look-back window, estimator type,
    hyperparameters, and a performance metric. This function extracts these values,
    resolves the filter and estimator names, and formats the results into a DataFrame.

    Args:
        evo (list of array-like): List of configuration vectors. Each vector should include:
            - cfg[0]: wavelet usage flag (0 or 1)
            - cfg[1]: wavelet filter index
            - cfg[2]: look-back window
            - cfg[3]: decomposition level
            - cfg[4]: estimator/model index
            - cfg[estimator_params_idx]: model hyperparameters
            - cfg[-1]: performance metric (e.g., MAPE or score)

    Returns:
        pandas.DataFrame: DataFrame summarizing all configurations with columns:
            - 'wavelet': Wavelet usage (0 or 1)
            - 'wavelet filter': Name of the wavelet filter used
            - 'look backs': Look-back window size
            - 'estimator': Name of the estimator/model
            - 'params': Model hyperparameters
            - 'métrica': Performance metric associated with the configuration

    Example:
        >>> evo = [[1, 2, 10, 3, 0, 0.01, 100, 0.85], [0, 0, 5, 1, 1, 0.1, 50, 0.78]]
        >>> df_summary = evolution(evo)
        >>> print(df_summary.head())
    """
    df = []
    for cfg in evo:
    
        wavelet = round(cfg[0])
        
        wavelet_filter = round(cfg[1])
        
        look_backs = round(cfg[2])
        
        # dec_level = round(cfg[3])
        
        estimator_name = round(cfg[4])
        
        wavelet_filter = get_filters(wavelet_filter)
        
        estimator_name = get_models(estimator_name)
        
        estimator_params_idx = get_positions(estimator_name)
        
        estimator_params = cfg[estimator_params_idx]
        
        estimator_params_names = get_params(estimator_name)
        
        estimator = get_estimator(estimator_params, estimator_params_names, estimator_name)
        
        params = estimator._convert_params()
        
        r = {"wavelet" : wavelet,
             "wavelet filter":wavelet_filter,
             "look backs" :look_backs,
             "estimator":estimator_name,
             "params":params,
             "métrica": cfg[-1]}
        r = {key: list([value]) for key, value in r.items()}
        r = pd.DataFrame(r)
        df.append(r)
    
    df=pd.concat(df)

    return df

# =============================================================================
# 
# =============================================================================
def main(run, df, heuristic,forecast,idx,pop,it,tg,tempo):
    """
    Executes a single run of a Wavelet-Augmented Neural Network (WANN) 
    optimization using a specified heuristic, evaluates performance, 
    and saves the results to a pickle file.

    The function:
    - Sets the random seed for reproducibility.
    - Logs the start of the run.
    - Configures optimization bounds based on forecast horizon and temporal resolution.
    - Initializes the WANN optimization object.
    - Runs the specified heuristic to optimize model parameters.
    - Trains and evaluates the WANN model using the optimized parameters.
    - Stores model results, metrics, and evolution history.
    - Saves results to a structured directory with a timestamped filename.
    - Logs completion or errors encountered during execution.

    Args:
        run (int): Run number, used for reproducibility and logging.
        df (pandas.DataFrame): Input time series dataset (one or more columns).
        heuristic (str): Name of the heuristic optimization method to apply.
        forecast (int): Forecast horizon (number of steps ahead to predict).
        idx (int): Index of the experiment, used in logging.
        pop (int): Population size for the heuristic algorithm.
        it (int): Number of iterations/generations for the heuristic.
        tg (float): Termination threshold or goal for the heuristic.
        tempo (str): Temporal identifier used for file naming and logging.

    Returns:
        None: Results are saved to a pickle file in the directory
        './pkl/{DATABASE}/{heuristic}/' with relevant metrics and model info.

    Logs:
        - Start and end of the run, including MAPE, RMSE, and elapsed time.
        - Errors encountered during execution with traceback formatting.

    Example:
        >>> main(run=1, df=df, heuristic='GA', forecast=1, idx=0, pop=50, it=100, tg=0.01, tempo='daily')
    """
    np.random.seed(run)
    
    logger.info(colored(f'{idx:02d} Iniciando ',
                        'blue',
                        attrs=["bold"]) + 
                heuristic.upper().rjust(4) + 
                " RUN " + 
                str(run) + 
                " Estação: " + 
                str(df.columns[0]) + 
                " Horizonte: " + 
                str(forecast) + 
                " - " + 
                str(dt.datetime.now()))
    try:
            
        _start = time.perf_counter()
        
        lower_bounds, upper_bounds  = config(forecast,tempo)
        
        args = (df, forecast, False)
        
        obj = WANN(args, wann, lower_bounds, upper_bounds)
    
        mh,evo = get_heuristica(heuristic, obj.fitness, lower_bounds,upper_bounds,pop,it,tg)
        evo = evolution(evo)
    
        results=wann(mh, df, forecast, True)
        
        # TODO: Traduzir

        if results == np.inf:
            raise ValueError("Heuristica não encontrou resultado!")
        
        results["est"] = df.columns[0]
        
        results["hor"] = forecast
        
        results['run_time'] = time.perf_counter() - _start
        
        results['run'] = run
        
        results["heuristic"] = heuristic
        
        results["heuristic evo"] = evo
                
        path = f'./pkl/{DATABASE}/{heuristic}/'
        
        if not os.path.exists(path):
            os.makedirs(path) 
                
        file = path + \
                'RUN_' +\
                str(run) +\
                "_FORECAST_" +\
                str(forecast) +\
                "_DATABASE_" + \
                df.columns[0] +\
                "_" + \
                tempo +"_"\
                + heuristic +\
                "_" +time.strftime("%Y_%m_%d_%Hh_%Mm_%S") +'.pkl'
        
        with open(file, 'wb') as f:
            pickle.dump(results,f)
                
        logger.info(colored(f'{idx:02d} Terminando',
                            'green',
                            attrs=["bold"]) + 
                    f'{heuristic.upper().rjust(4)}' + 
                    " Estação: " + 
                    df.columns[0] + 
                    " Horizonte: " + 
                    str(forecast) + 
                    " MAPE: " + 
                    str(round(results["mape"], 4)) + 
                    " RMSE: " + 
                    str(round(results['rmse'], 4)) + 
                    " Time: " + 
                    str(round((time.perf_counter() - _start), 4)))
        
    except Exception as error:
        
        error = str(error)
        error = "\t" + error.replace("\n", "\n\t")
        
        logger.error(colored(f'{idx:02d} ERRO',
                             'red',
                             attrs=["bold"]) + 
                     f'{heuristic.upper().rjust(4)}' + 
                     " Estação: " + 
                     df.columns[0] + 
                     " Horizonte: " + 
                     str(forecast) + 
                     "\n" +
                     colored(error, 'yellow', attrs=["bold"]))
        

# =============================================================================
# 
# =============================================================================
if __name__ == "__main__":
    
    # Number of times to be run - Int
    RUNS = 3
    
    # Time to be forecasted
    FORECAST = [
                1,
                3,
                7,
                # 12,
                # 14,
                # 21,
                # 48,
                ]

    DATABASE = {"10_est" : ["58880001","58235100"],
                # "mg"     : ["44200000","61024000","56610000","61135000","58535000"],
                # "furnas" : ["FURNAS"],
                # "bacias" : ["AMAZONAS DARDANELOS",
                #             "PARANAIBA CACU",
                #             "DOCE AIMORES",  
                #             "URUGUAI CAMPOS NOVOS",
                #             "PARAIBA DO SUL ANTA",],
                # "tests"  : ["tests"]
            }
    
# =============================================================================
#     TEMPO choices:
#     -> "horario"
#     -> "diario"
#     -> "mensal"
# =============================================================================
    
    TEMPO = "diario"
    
    df_list = []
    
    for k in DATABASE.keys():
        
        df_list.extend(get_data(k,DATABASE[k],TEMPO))
    
    heuristics = [
                    # "fda",
                    # "pso",
                    "de",
                    # "igwo",
                    # "cso",
                    # "sa",
                    # "ga"
                  ]
     
    POP = 70
    
    ITER = 150
    
    TARGET_VALUE = 1
    
    N_JOBS = 3
    
    pool = list(itertools.product(range(RUNS),df_list, heuristics, FORECAST))
    
    if N_JOBS == 1:
        
        for i, (run,df, heuristic, forecast) in enumerate(pool):
        
            main(run, df, heuristic,forecast,i,POP,ITER,TARGET_VALUE,TEMPO)
        
    else:
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_JOBS) as executor:
            
            for i, (run,df, heuristic, forecast) in enumerate(pool):
                
                executor.submit(main, run, df, heuristic,forecast,i,POP,ITER,TARGET_VALUE,TEMPO)
                
        # with concurrent.futures.ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        #     futures = []
        #     for i, (run,df, heuristic, forecast) in enumerate(pool):
        #         executor.submit(main, run, df, heuristic,forecast,i,POP,ITER,TARGET_VALUE,TEMPO)
        
            # Aguarda a conclusão de todas as tarefas
            # for future in concurrent.futures.as_completed(futures):
            #     future.result()  # Captura possíveis exceções
            
        