#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:29:05 2024

@author: rodrigo
"""


def get_estimator(bounds, param_names, estimator_name):
    if estimator_name == "MLP_Single" or estimator_name == "MLP_Multi":
        from .mlp_regressor import MLP as estimator

    elif estimator_name == "XGB":
        from .xgb_regressor import XGB as estimator

    elif estimator_name == "LSTM":
        from .lstm_regressor import LSTM_R as estimator

    elif estimator_name == "CONV_LSTM":
        from .conv_lstm_regressor import CONV_LSTM as estimator

    # elif estimator_name == "GMDH":
    #     return gmdh_fit(X_train, y_train, bounds, param_names)

    elif estimator_name == "SVR":
        from .svr_regressor import SVR as estimator

    elif estimator_name == "LSSVR":
        # from .lssvr_regressor import LSSVR as estimator
        from .lssvr_regressor_ver2 import LSSVR as estimator

    elif estimator_name == "ELM":
        from .elm_regressor import ELM as estimator
        
    elif estimator_name == "ELM2":
        from .elm_regressor_ver2 import ELM as estimator

    elif estimator_name == "KNN":
        from .knn_regressor import KNN as estimator
        
# =============================================================================
#     Linear Models
# =============================================================================

    elif estimator_name == "LASSO":
        from .linear.lasso_regressor import LassoModel as estimator
        
    elif estimator_name == "RIDGE":
        from .linear.ridge_regressor import RidgeModel as estimator
        
    elif estimator_name == "LINEAR":
        from .linear.linear_regressor import LinearModel as estimator
        
    elif estimator_name == "ELASTIC_NET":
        from .linear.elastic_net_regressor import ElasticNetModel as estimator
        
    try:
        model = estimator(bounds=bounds, param_names=param_names)
    except:
        print(estimator_name)
        raise ValueError(estimator_name, "not defined")

    return model

# TODO -> Verify if this function is better
'''
def get_estimator(bounds, param_names, estimator_name):
    # name dictionary → import from class
    estimators = {
        "MLP_Single": lambda: __import__(".mlp_regressor", fromlist=["MLP"]).MLP,
        "MLP_Multi":  lambda: __import__(".mlp_regressor", fromlist=["MLP"]).MLP,
        "XGB":        lambda: __import__(".xgb_regressor", fromlist=["XGB"]).XGB,
        "LSTM":       lambda: __import__(".lstm_regressor", fromlist=["LSTM_R"]).LSTM_R,
        "CONV_LSTM":  lambda: __import__(".conv_lstm_regressor", fromlist=["CONV_LSTM"]).CONV_LSTM,
        "SVR":        lambda: __import__(".svr_regressor", fromlist=["SVR"]).SVR,
        "LSSVR":      lambda: __import__(".lssvr_regressor_ver2", fromlist=["LSSVR"]).LSSVR,
        "ELM":        lambda: __import__(".elm_regressor", fromlist=["ELM"]).ELM,
        "ELM2":       lambda: __import__(".elm_regressor_ver2", fromlist=["ELM"]).ELM,
        "KNN":        lambda: __import__(".knn_regressor", fromlist=["KNN"]).KNN,
        "LASSO":      lambda: __import__(".linear.lasso_regressor", fromlist=["LassoModel"]).LassoModel,
        "RIDGE":      lambda: __import__(".linear.ridge_regressor", fromlist=["RidgeModel"]).RidgeModel,
        "LINEAR":     lambda: __import__(".linear.linear_regressor", fromlist=["LinearModel"]).LinearModel,
        "ELASTIC_NET":lambda: __import__(".linear.elastic_net_regressor", fromlist=["ElasticNetModel"]).ElasticNetModel,
    }

    try:
        estimator_class = estimators[estimator_name]()  # get the class
        return estimator_class(bounds=bounds, param_names=param_names)
    except KeyError:
        raise ValueError(f"{estimator_name} not defined")
'''
    
def dict_to_arrays(d):

    keys = list(d.keys())

    values = [v for v in d.values()]

    return keys, values