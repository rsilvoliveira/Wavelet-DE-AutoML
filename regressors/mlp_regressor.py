#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 08:53:45 2024

@author: rodrigo
"""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits


class MLP(BaseEstimator, RegressorMixin):

    def __init__(self, bounds=None, param_names=None):
        # Default parameters for the class
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

        if '2layer' in params_dict:
            integer_par = ['solver', 'activation',
                           'hidden_layer_sizes', '2layer', '3layer']
            for k in integer_par:
                params_dict[k] = round(params_dict[k])

            if params_dict['2layer'] != 0 and params_dict['3layer'] != 0:
                params_dict['hidden_layer_sizes'] = (params_dict['hidden_layer_sizes'],
                                                     params_dict['2layer'],
                                                     params_dict['3layer'])
            elif params_dict['2layer'] != 0 and params_dict['3layer'] == 0:
                params_dict['hidden_layer_sizes'] = (params_dict['hidden_layer_sizes'],
                                                     params_dict['2layer'])
            params_dict.pop('2layer', None)
            params_dict.pop('3layer', None)
        else:
            integer_par = ['solver', 'activation', 'hidden_layer_sizes']
            for k in integer_par:
                params_dict[k] = round(params_dict[k])

        params_dict["solver"] = "adam" if params_dict["solver"] == 0 else "lbfgs"

        activation = {0: 'identity', 1: 'logistic', 2: 'tanh', 3: 'relu'}
        params_dict["activation"] = activation[params_dict["activation"]]

        params_dict["early_stopping"] = True
        params_dict["max_iter"] = 1000
        
        return params_dict

    def fit(self, X, y):
        # Converts the parameters and fits the model
        params_dict = self._convert_params()
        with threadpool_limits(limits=1):  # Limits the CPUs
            self.model = MLPRegressor(**params_dict)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Example parameters
    bounds = [1, 2, 10, 5, 3]  # Example of values for each parameter
    param_names = ['solver', 'activation',
                   'hidden_layer_sizes', '2layer', '3layer']

    # Instantiate and use the model
    model = MLP(bounds=bounds, param_names=param_names)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)

    print(f"Model score: {score}")
