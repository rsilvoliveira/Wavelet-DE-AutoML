#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:05:10 2024

@author: rodrigo
"""

from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
# from fit_model import device

class XGB(BaseEstimator, RegressorMixin):
    def __init__(self, bounds=None, param_names=None):
        # Inicializa com os parâmetros fornecidos
        self.bounds = bounds
        self.param_names = param_names
        self.model = None

    def _convert_params(self):
        if self.param_names is None or self.bounds is None:
            raise ValueError("param_names and bounds must be set before fitting the model.")
        
        # from .regressors import device

        # Converte os parâmetros para um dicionário
        params_dict = {self.param_names[i]: self.bounds[i]
                       for i in range(len(self.param_names))}
        
        # Arredonda os parâmetros apropriados
        params_dict["max_depth"] = round(params_dict["max_depth"])
        params_dict["n_estimators"] = round(params_dict["n_estimators"])
        params_dict["rounds"] = round(params_dict["rounds"])
        
        # Define parâmetros específicos do XGBoost
        # params_dict["device"] = device()
        # params_dict["device"] = "cpu"  # Assume que o dispositivo é CPU; ajuste se necessário
        params_dict["tree_method"] = "hist"
        params_dict["eval_metric"] = "rmse"
        params_dict["objective"] = "reg:squarederror"

        return params_dict

    def fit(self, X, y):
        # Converte os parâmetros e ajusta o modelo
        params_dict = self._convert_params()
        
        xgb.set_config(verbosity=0)  # Silencia logs
        
        dtrain = xgb.DMatrix(X, label=y)
        num_rounds = params_dict.pop("rounds", 10)  # Use um valor padrão se "rounds" não estiver presente
        
        self.model = xgb.train(params_dict, dtrain, num_rounds)
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("You must fit the model before predicting.")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def score(self, X, y):
        if self.model is None:
            raise RuntimeError("You must fit the model before scoring.")
        # from sklearn.metrics import mean_absolute_percentage_error as cmape
        from sklearn.metrics import mean_squared_error 
        from math import sqrt
        
        y_pred = self.predict(X)
        
        rmse = sqrt(mean_squared_error(y,y_pred))
        
        # return cmape(y, y_pred) * 100
        return rmse

# Exemplo de uso
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression
    
    # Gerar dados de exemplo
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Parâmetros de exemplo
    bounds = [5, 100, 10]  # Exemplo de valores para cada parâmetro
    param_names = ['max_depth', 'n_estimators', 'rounds']
    
    # Instanciar e usar o modelo
    model = XGB(bounds=bounds, param_names=param_names)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
    
    print(f"Model score: {score}")
