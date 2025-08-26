#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:16:09 2024

@author: rodrigo
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error

class RidgeModel(BaseEstimator, RegressorMixin):
    def __init__(self, bounds=None, param_names=None):
        # Inicializa com os parâmetros fornecidos
        self.bounds = bounds
        self.param_names = param_names
        self.model = None

    def _convert_params(self):
        if self.param_names is None or self.bounds is None:
            raise ValueError("param_names and bounds must be set before fitting the model.")
        
        # Converte os parâmetros para um dicionário
        params_dict = {self.param_names[i]: self.bounds[i]
                       for i in range(len(self.param_names))}
        
        integer_par = ['max_iter']
        
        for k in integer_par:
            params_dict[k] = round(params_dict[k])
        
        return params_dict
    
    def fit(self, X, y):
        # Converte os parâmetros e ajusta o modelo
        params_dict = self._convert_params()
        
        # Ajuste do modelo Ridge com parâmetros
        self.model = Ridge(**params_dict)
        self.model.fit(X, y)
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("You must fit the model before predicting.")
        return self.model.predict(X)
    
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
    import numpy as np

    # Simular dados de janelas deslizantes
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Exemplo de características de janelas
    y = np.array([3, 6, 9])  # Exemplo de variável alvo

    # Definir limites e nomes de parâmetros para Ridge
    bounds = [1.0, True,200]  # Exemplo: alpha e fit_intercept
    param_names = ['alpha', 'fit_intercept',"max_iter"]

    # Inicializar e treinar o modelo
    model = RidgeModel(bounds=bounds, param_names=param_names)
    model.fit(X, y)

    # Avaliar o modelo
    mape = model.score(X, y)
    print(f'MAPE: {mape}')
