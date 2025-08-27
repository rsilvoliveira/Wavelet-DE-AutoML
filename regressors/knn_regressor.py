#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:17:41 2024

@author: rodrigo
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor as knn

class KNN(BaseEstimator, RegressorMixin):
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

        # Rounds the appropriate parameters
        integer_par = ['n_neighbors', 'weights','algorithm','leaf_size']
        for k in integer_par:
            params_dict[k] = round(params_dict[k])

        # Maps the weights value to the corresponding name
        weights = {0: 'uniform',
                  1: 'distance',
                  }
        
        algorithm = {0: 'auto',
                  1: 'ball_tree',
                  2:'kd_tree',
                  3:'brute'
                  }
        
        params_dict["weights"] = weights.get(params_dict["weights"], 'uniform')  
        params_dict["algorithm"] = algorithm.get(params_dict["algorithm"], 'auto')  
        
        return params_dict
    
    def fit(self, X, y):
        # Converts the parameters and fits the model
        params_dict = self._convert_params()
        
        self.model = knn(**params_dict)
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

# Example of use
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.datasets import make_regression

    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Example parameters
    bounds = [5,
              0,
              0,
              30,
              2]  # Example of values for each parameter
    param_names = ['n_neighbors',
                   'weights',
                   'algorithm',
                   'leaf_size',
                   'p',
                   ]

    # Instantiate and use the model
    model = KNN(bounds=bounds, param_names=param_names)
    # Calculate the score using cross_val_score
    score = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
    
    print(f"Model score: {score}")
