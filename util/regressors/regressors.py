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


def dict_to_arrays(d):

    keys = list(d.keys())

    values = [v for v in d.values()]

    return keys, values