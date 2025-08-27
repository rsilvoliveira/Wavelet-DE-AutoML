#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:23:11 2024

@author: rodrigo
"""

# https://github.com/zealberth/lssvr/tree/master

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_X_y, check_array
from sklearn.exceptions import NotFittedError
from scipy.sparse.linalg import lsmr


class LSSVR(BaseEstimator, RegressorMixin):
    """Least Squares Support Vector Regression.

    Parameters
    ----------
    C : float, default=2.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    kernel : {'linear', 'rbf'}, default='linear'
        Specifies the kernel type to be used in the algorithm.
        It must be 'linear', 'rbf' or a callable.

    gamma : float, default = None
        Kernel coefficient for 'rbf'


    Attributes
    ----------
    support_: boolean np.array of shape (n_samples,), default = None
        Array for support vector selection.

    alpha_ : array-like
        Weight matrix

    bias_ : array-like
        Bias vector


    """
    def __init__(self, bounds=[2,1,None], param_names=['C','kernel','gamma']):
        # Initializes with the provided parameters

        self.bounds = bounds
        self.param_names = param_names
        self.model = None
        
    # def __init__(self, C=2.0, kernel='linear', gamma=None):
    #     self.C = C
    #     self.kernel = kernel
    #     self.gamma = gamma
        
    def _convert_params(self):
        if self.param_names is None or self.bounds is None:
            raise ValueError("param_names and bounds must be set before fitting the model.")

        # Converts the parameters to a dictionary
        params_dict = {self.param_names[i]: self.bounds[i]
                       for i in range(len(self.param_names))}

        # Rounds the appropriate parameters
        integer_par = ['kernel']
        for k in integer_par:
            params_dict[k] = round(params_dict[k])

        # Maps the kernel value to the corresponding name
        kernel = {0: 'linear',
                  1: 'poly',
                  2: 'rbf',
                  }

        params_dict["kernel"] = kernel.get(params_dict["kernel"], 'rbf')  # Default to 'rbf'

        self.C = params_dict["C"]
        self.kernel = params_dict["kernel"]
        if 'gamma' in self.param_names:
            self.gamma = params_dict["gamma"]
        else:
            self.gamma = None
            
        return params_dict
    
    def fit(self, X, y, support=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        support : boolean np.array of shape (n_samples,), default = None
            Array for support vector selection.

        Returns
        -------
        self : object
            An instance of the estimator.
        """
        self._convert_params()
        
        X, y = check_X_y(X, y, multi_output=True, dtype='float')

        if not support:
            self.support_ = np.ones(X.shape[0], dtype=bool)
        else:
            self.support_ = check_array(support, ensure_2d=False, dtype='bool')

        self.support_vectors_ = X[self.support_, :]
        support_labels = y[self.support_]

        self.K_ = self.kernel_func(X, self.support_vectors_)
        omega = self.K_.copy()
        np.fill_diagonal(omega, omega.diagonal()+self.support_/self.C)

        D = np.empty(np.array(omega.shape) + 1)

        D[1:, 1:] = omega
        D[0, 0] = 0
        D[0, 1:] = 1
        D[1:, 0] = 1

        shape = np.array(support_labels.shape)
        shape[0] += 1
        t = np.empty(shape)

        t[0] = 0
        t[1:] = support_labels

        # TODO: maybe give access to  lsmr atol and btol ?
        try:
            z = lsmr(D.T, t)[0]
        except:
            z = np.linalg.pinv(D).T @ t

        self.bias_ = z[0]
        self.alpha_ = z[1:]
        self.alpha_ = self.alpha_[self.support_]

        return self

    def predict(self, X):
        """
        Predict using the estimator.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """

        if not hasattr(self, 'support_vectors_'):
            raise NotFittedError

        X = check_array(X, ensure_2d=False)
        K = self.kernel_func(X, self.support_vectors_)
        return (K @ self.alpha_) + self.bias_

    def kernel_func(self, u, v):
        if self.kernel == 'linear':
            return np.dot(u, v.T)

        elif self.kernel == 'rbf':
            return rbf_kernel(u, v, gamma=self.gamma)

        elif callable(self.kernel):
            if hasattr(self.kernel, 'gamma'):
                return self.kernel(u, v, gamma=self.gamma)
            else:
                return self.kernel(u, v)
        else:
            # default to linear
            return np.dot(u, v.T)

    def score(self, X, y):
        # if self.model is None:
            # raise RuntimeError("You must fit the model before scoring.")
        # from sklearn.metrics import mean_absolute_percentage_error as cmape
        from sklearn.metrics import mean_squared_error 
        from math import sqrt
        
        y_pred = self.predict(X)
        
        rmse = sqrt(mean_squared_error(y,y_pred))
        
        # return cmape(y, y_pred) * 100
        return rmse

    def norm_weights(self):
        A = self.alpha_.reshape(-1, 1) @ self.alpha_.reshape(-1, 1).T

        W = A @ self.K_[self.support_, :]
        return np.sqrt(np.sum(np.diag(W)))
    
    
# Example of use
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.datasets import make_regression

    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Example parameters
    bounds = [4, 2]  # Example of values for each parameter
    param_names = ['kernel', 'C']

    # Instantiate and use the model
    model = LSSVR(bounds=bounds, param_names=param_names)
    # Calculate the score using cross_val_score
    score = cross_val_score(model, X_train, y_train, cv=5)
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    # score = model.score(X_test, y_test)
    
    print(f"Model score: {score}")
