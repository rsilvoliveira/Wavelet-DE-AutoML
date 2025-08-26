#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:17:41 2024

@author: rodrigo
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR as svr_r

class SVR(BaseEstimator, RegressorMixin):
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
        
        # Arredonda os parâmetros apropriados
        integer_par = ['kernel', 'degree']
        for k in integer_par:
            params_dict[k] = round(params_dict[k])
        
        # Mapeia o valor de kernel para o nome correspondente
        kernel = {0: 'linear',
                  1: 'poly',
                  2: 'rbf',
                  3: 'sigmoid',
                  4: 'precomputed'}
        
        params_dict["kernel"] = kernel.get(params_dict["kernel"], 'rbf')  # Default para 'rbf'
        
        params_dict['max_iter'] = 3000
        
        return params_dict
    
    def fit(self, X, y):
        # Converte os parâmetros e ajusta o modelo
        params_dict = self._convert_params()
        
        self.model = svr_r(**params_dict)
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
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.datasets import make_regression
    
    # Gerar dados de exemplo
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Parâmetros de exemplo
    bounds = [2, 3]  # Exemplo de valores para cada parâmetro
    param_names = ['kernel', 'degree']
    
    # Instanciar e usar o modelo
    model = SVR(bounds=bounds, param_names=param_names)
    # Calcular o score usando cross_val_score
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    # score = model.score(X_test, y_test)
    
    print(f"Model score: {score}")
