# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 09:46:33 2024

@author: Rodrigo
"""

from sklearn.base import BaseEstimator, RegressorMixin
from .ELM import ELMRegressor


class ELM(BaseEstimator, RegressorMixin):
    def __init__(self, bounds=None, param_names=None):
        # Inicializa com os parâmetros fornecidos
        self.bounds = bounds
        self.param_names = param_names
        self.model = None

    def _convert_params(self):
        if self.param_names is None or self.bounds is None:
            raise ValueError(
                "param_names and bounds must be set before fitting the model.")

        # Converte os parâmetros para um dicionário
        params_dict = {self.param_names[i]: self.bounds[i]
                       for i in range(len(self.param_names))}

        integer_par = ['n_hidden',
                       'activation_func',
                       ]

        for k in integer_par:
            params_dict[k] = round(params_dict[k])

       
        activation_func = {0: 'tanh',
                           1: 'sine',
                           2: 'tribas',
                           3: 'identity',
                           4: 'relu',
                           5: 'inv_tribas',
                           6: 'sigmoid',
                           7: 'logistic'}

        params_dict['activation_func'] = activation_func[params_dict['activation_func']]
            
        return params_dict

    def fit(self, X, y):
        # Converte os parâmetros e ajusta o modelo
        params_dict = self._convert_params()
        self.model = ELMRegressor(**params_dict)
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
         

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression

    # Gerar dados de exemplo
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Parâmetros de exemplo
    bounds = [5,  30, 2]  # Exemplo de valores para cada parâmetro
    param_names = ['alpha', 'n_hidden', 'activation_func']

    # Instanciar e usar o modelo
    model = ELM(bounds=bounds, param_names=param_names)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)

    print(f"Model score: {score}")
