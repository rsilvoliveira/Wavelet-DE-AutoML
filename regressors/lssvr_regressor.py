#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:15:33 2024

@author: rodrigo
"""

"""Least Squares Support Vector Regression."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process import kernels
# from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from scipy.sparse import linalg


class LSSVR(BaseEstimator, RegressorMixin):
   
    def __init__(self, bounds=None, param_names=None):
       # Inicializa com os parâmetros fornecidos
       self.bounds = bounds
       self.param_names = param_names
       self.model = None
       self.idxs  = None
       self.K = None
       self.bias = None 
       self.alphas = None
       self.supportVectors      = None
       self.supportVectorLabels = None
              
    def _convert_params(self):
        if self.param_names is None or self.bounds is None:
            raise ValueError("param_names and bounds must be set before fitting the model.")
        
        # Converte os parâmetros para um dicionário
        params_dict = {self.param_names[i]: self.bounds[i]
                       for i in range(len(self.param_names))}
        
        # Arredonda os parâmetros apropriados
        integer_par = ['kernel']
        for k in integer_par:
            params_dict[k] = round(params_dict[k])
        
        # Mapeia o valor de kernel para o nome correspondente
        kernel = {0: 'linear',
                  1: 'rbf',
                  2: 'matern'}
        
        params_dict["kernel"] = kernel.get(params_dict["kernel"], 'rbf')  # Default para 'rbf'
        
        self.C = params_dict['C']
        self.kernel = params_dict['kernel']
        self.gamma = params_dict['gamma']
        
        return params_dict
    
    def fit(self, X, y):
        # Converte os parâmetros e ajusta o modelo
        self._convert_params()
        
        if type(self.idxs) == type(None):
           self.idxs=np.ones(len(X), dtype=bool)
           
        self.supportVectors      = X[self.idxs, :]
        self.supportVectorLabels = y[self.idxs]
        
        K = self.kernel_func(self.kernel, X, self.supportVectors, self.gamma)

        self.K = K
        OMEGA = K
        OMEGA[self.idxs, np.arange(OMEGA.shape[1])] += 1/self.C

        D = np.zeros(np.array(OMEGA.shape) + 1)
        
        D[1:,1:] = OMEGA
        D[0, 1:] += 1
        D[1:,0 ] += 1

        n = len(self.supportVectorLabels) + 1
        t = np.zeros(n)
       
        t[1:n] = self.supportVectorLabels
   
        # sometimes this function breaks
        try:
           z = linalg.lsmr(D.T, t)[0]
        except:
           z = np.linalg.pinv(D).T @ t.ravel()

        self.bias   = z[0]
        self.alphas = z[1:]
        self.alphas = self.alphas[self.idxs]

        return self
    
    def predict(self, x_test):
        K = self.kernel_func(self.kernel, x_test, self.supportVectors, self.gamma)

        return (K @ self.alphas) + self.bias
    
    def kernel_func(self, kernel, u, v, gamma):
        if kernel == 'linear':
            k = np.dot(u, v.T)
        if kernel == 'rbf':
            k = rbf_kernel(u, v, gamma=gamma)
            # temp = kernels.RBF(length_scale=(1/gamma))
            # k = temp(u, v)
        if kernel == 'matern':
            kr = kernels.Matern(nu=self.gamma)
            k = kr(u,v)
            # temp = kernels.RBF(length_scale=(1/gamma))
            # k = temp(u, v)
        
        return k
    
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
        n = len(self.supportVectors)

        A = self.alphas.reshape(-1,1) @ self.alphas.reshape(-1,1).T
        # import pdb; pdb.set_trace()
        W = A @ self.K[self.idxs,:]
        return np.sqrt(np.sum(np.diag(W)))
    
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression

    # Gerar dados de exemplo
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Parâmetros de exemplo
    # Parâmetros de exemplo
    bounds = [4, 2, 0.1]  # Exemplo de valores para cada parâmetro
    param_names = ['kernel', 'C', "gamma"]

    # Instanciar e usar o modelo
    model = LSSVR(bounds=bounds, param_names=param_names)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)

    print(f"Model score: {score}")

    