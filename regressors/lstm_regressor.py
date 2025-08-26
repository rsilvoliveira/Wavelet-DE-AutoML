#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:08:49 2024

@author: rodrigo
"""

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import backend as K

class LSTM_R(BaseEstimator, RegressorMixin):
    def __init__(self, bounds=None, param_names=None):
        # Inicializa os parâmetros do modelo
        self.bounds = bounds
        self.param_names = param_names
        # self.verbose = 0
        self.model = None

    def _convert_params(self):
        if self.param_names is None or self.bounds is None:
            raise ValueError("param_names and bounds must be set before fitting the model.")
        
        # Converte os parâmetros para um dicionário
        params_dict = {self.param_names[i]: self.bounds[i]
                       for i in range(len(self.param_names))}
        
        # Arredonda os parâmetros apropriados
        integer_par = ['first_layer', 'second_layer', 'batch_size']
        for k in integer_par:
            params_dict[k] = round(params_dict[k])
        
        return params_dict
    
    def fit(self, X, y):
        # Converte os parâmetros e prepara os dados
        params_dict = self._convert_params()
        
        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        # Define o modelo
        self.model = Sequential()
        self.model.add(LSTM(units=params_dict["first_layer"],
                            activation="relu",
                            return_sequences=True,
                            input_shape=(1, X.shape[2])))
        self.model.add(LSTM(units=params_dict["second_layer"], 
                            activation="relu"))
        self.model.add(Dense(units=16))
        self.model.add(Dense(units=1))

        # Compila o modelo
        self.model.compile(optimizer="adam", loss="mean_squared_error")

        # Define o callback EarlyStopping
        es = EarlyStopping(monitor="loss", mode="min", verbose=0, patience=50)

        # Define o tamanho do batch
        batch_size = 2 ** params_dict["batch_size"]
        
        # Treina o modelo
        self.model.fit(X,
                       y,
                       epochs=250, 
                       batch_size=batch_size,
                       verbose=0,
                       callbacks=[es])
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("You must fit the model before predicting.")
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        return self.model.predict(X,verbose=0).ravel()
    
    def score(self, X, y):
        if self.model is None:
            raise RuntimeError("You must fit the model before scoring.")
        # from sklearn.metrics import mean_absolute_percentage_error as cmape
        # from sklearn.metrics import root_mean_squared_error
        from sklearn.metrics import mean_squared_error 
        from math import sqrt
        
        y_pred = self.predict(X)
        rmse = sqrt(mean_squared_error(y,y_pred))
        # return cmape(y, y_pred) * 100
        # return root_mean_squared_error(y, y_pred)
        return rmse

# Exemplo de uso
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression
    
    # Gerar dados de exemplo
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Parâmetros de exemplo
    bounds = [50, 30, 4]  # Exemplo de valores para cada parâmetro
    param_names = ['first_layer', 'second_layer', 'batch_size']
    
    # Instanciar e usar o modelo
    model = LSTM_R(bounds=bounds, param_names=param_names)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
    
    print(f"Model score: {score}")
