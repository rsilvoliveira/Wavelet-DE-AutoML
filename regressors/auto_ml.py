#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:22:16 2024

@author: Rodrigo
"""

from sklearn.base import BaseEstimator, RegressorMixin
import autosklearn.regression

class AUTOML(BaseEstimator, RegressorMixin):
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

        integer_par = ['time_left_for_this_task',
                       #'per_run_time_limit',
                       'memory_limit',
                       ]

        for k in integer_par:
            params_dict[k] = round(params_dict[k])
        
        #if params_dict['per_run_time_limit'] == 0:
        #    params_dict['per_run_time_limit'] = None

        # params_dict['logging_config']  = {'version': 1,'disable_existing_loggers': True,
        #                                   'formatters':  
        #                                       {'simple':    
        #                                        {'format': ''}}}
        
        params_dict['logging_config']= {
                                        'version': 1,
                                        'disable_existing_loggers': False,
                                        'formatters': {
                                            'simple': {
                                                'format': '[%(levelname)s] [%(asctime)s:%(name)s] %(message)s'
                                            }
                                        },
                                        'handlers': {
                                            'file_handler': {
                                                'class': 'logging.FileHandler',
                                                'level': 'DEBUG',
                                                'formatter': 'simple',
                                                'filename': 'autosklearn.log'
                                            },
                                            'distributed_logfile': {
                                                'class': 'logging.FileHandler',
                                                'level': 'DEBUG',
                                                'formatter': 'simple',
                                                'filename': 'distributed.log'
                                            }
                                        },
                                        'root': {
                                            'level': 'DEBUG',
                                            'handlers': ['file_handler']
                                        },
                                        'loggers': {
                                            'autosklearn.metalearning': {
                                                'level': 'DEBUG',
                                                'handlers': ['file_handler']
                                            },
                                            'autosklearn.automl_common.utils.backend': {
                                                'level': 'DEBUG',
                                                'handlers': ['file_handler'],
                                                'propagate': False
                                            },
                                            'smac.intensification.intensification.Intensifier': {
                                                'level': 'INFO',
                                                'handlers': ['file_handler']
                                            },
                                            'smac.optimizer.local_search.LocalSearch': {
                                                'level': 'INFO',
                                                'handlers': ['file_handler']
                                            },
                                            'smac.optimizer.smbo.SMBO': {
                                                'level': 'INFO',
                                                'handlers': ['file_handler']
                                            },
                                            'EnsembleBuilder': {
                                                'level': 'DEBUG',
                                                'handlers': ['file_handler']
                                            },
                                            'distributed': {
                                                'level': 'DEBUG',
                                                'handlers': ['distributed_logfile']
                                            }
                                        }
                                    }

        return params_dict

    def fit(self, X, y):
        # Converte os parâmetros e ajusta o modelo
        params_dict = self._convert_params()

        self.model = autosklearn.regression.AutoSklearnRegressor(**params_dict)
        
        self.model.fit(X, y)

        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("You must fit the model before predicting.")
        return self.model.predict(X)

    def score(self, X, y):
        if self.model is None:
            raise RuntimeError("You must fit the model before scoring.")
        from sklearn.metrics import mean_absolute_percentage_error as cmape
        # from sklearn.metrics import mean_squared_error 
        # from math import sqrt
        
        y_pred = self.predict(X)
        
        # rmse = sqrt(mean_squared_error(y,y_pred))
        
        return cmape(y, y_pred) * 100
        # return rmse


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression

    # Gerar dados de exemplo
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Parâmetros de exemplo
    bounds = [30, 
              #0,
              1024*5]  # Exemplo de valores para cada parâmetro
    param_names = ['time_left_for_this_task', 
                   #'per_run_time_limit',
                   'memory_limit']

    # Instanciar e usar o modelo
    model = AUTOML(bounds=bounds, param_names=param_names)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)

    print(f"Model score: {score}")
