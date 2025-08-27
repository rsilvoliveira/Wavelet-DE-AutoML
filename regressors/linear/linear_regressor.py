#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:10:25 2024

@author: rodrigo
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

class LinearModel(BaseEstimator, RegressorMixin):
    def __init__(self, bounds=None, param_names=None):
        # Initializes with the provided parameters
        self.bounds = bounds
        self.param_names = param_names
        self.model = None

    def _convert_params(self):
        if self.param_names is None or self.bounds is None:
            raise ValueError("param_names and bounds must be set before fitting the model.")

        # Converts the parameters to a dictionary
        params_dict = {self.param_names[i]: self.bounds[i]
                       for i in range(len(self.param_names))}
        
        integer_par = ['fit_intercept', 'copy_X', 'positive']
        
        for k in integer_par:
            params_dict[k] = bool(round(params_dict[k]))
            
        params_dict['n_jobs']=1
        
        # In this example, there are no parameters to round like in SVR, but you can add logic if necessary
        return params_dict
    
    def fit(self, X, y):
        # Converts the parameters and fits the model
        params_dict = self._convert_params()

        # Fits the LinearRegression model with parameters
        self.model = LinearRegression(**params_dict)
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

# Example usage
if __name__ == "__main__":
    import numpy as np

    # Simulate sliding window data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Example sliding window features
    y = np.array([4, 7, 10])  # Example target variable

    # Define bounds and parameter names
    bounds = [True, 0.3, 0.7]  # Just an example, for a parameter that accepts boolean
    param_names = ['fit_intercept', 'copy_X', 'positive']

    # Initialize and train the model
    model = LinearModel(bounds=bounds, param_names=param_names)
    model.fit(X, y)

    # Evaluate the model
    mape = model.score(X, y)
    print(f'MAPE: {mape}')
