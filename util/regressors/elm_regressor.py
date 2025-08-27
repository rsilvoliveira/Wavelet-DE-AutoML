# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 09:46:33 2024

@author: Rodrigo
"""

from sklearn.base import BaseEstimator, RegressorMixin
from skelm import ELMRegressor


class ELM(BaseEstimator, RegressorMixin):
    def __init__(self, bounds=None, param_names=None):
        # Initializes with the provided parameters
        self.bounds = bounds
        self.param_names = param_names
        self.model = None

    def _convert_params(self):
        if self.param_names is None or self.bounds is None:
            raise ValueError(
                "param_names and bounds must be set before fitting the model.")

        # Converts the parameters to a dictionary
        params_dict = {self.param_names[i]: self.bounds[i]
                       for i in range(len(self.param_names))}

        integer_par = ['include_original_features',
                       '1_neurons',
                       '2_neurons',
                       '3_neurons',
                       '1_ufunc',
                       '2_ufunc',
                       '3_ufunc']

        for k in integer_par:
            params_dict[k] = round(params_dict[k])

        params_dict['n_neurons'] = []

        for k in ['1_neurons', '2_neurons', '3_neurons']:
            params_dict['n_neurons'].append(params_dict[k])
            params_dict.pop(k, None)

        #TODO: Alterar a variável valor
        params_dict['n_neurons'] = params_dict['n_neurons'][:next((i for i, valor in enumerate(params_dict['n_neurons']) if valor == 0),  
                                                                  len(params_dict['n_neurons']))]

        ufunc = {0: 'tanh', 1: 'sigm', 2: 'relu', 3: 'lin'}

        params_dict['ufunc'] = []
        for k in ['1_ufunc', '2_ufunc', '3_ufunc']:
            params_dict[k] = ufunc[params_dict[k]]
            params_dict['ufunc'].append(params_dict[k])
            params_dict.pop(k, None)
            
        params_dict['ufunc'] =  params_dict['ufunc'][:len( params_dict['n_neurons'])]
        return params_dict

    def fit(self, X, y):
        # Converts the parameters and fits the model
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

    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Example parameters
    bounds = [5, 1, 30, 10, 0.2,0,1,2]  # Example of values for each parameter
    param_names = ['alpha', 'include_original_features',
                   '1_neurons', '2_neurons', '3_neurons',
                   '1_ufunc', '2_ufunc', '3_ufunc']

    # Instantiate and use the model
    model = ELM(bounds=bounds, param_names=param_names)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)

    print(f"Model score: {score}")
