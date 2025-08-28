
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:15:13 2024

@author: rodrigo
"""

USE_WAVELETS = (0,1)

FILTERS = ('coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6',
           'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8',
           'db9', 'db10', 
           # 'db11', 'db12', 'db13', 'db14', 'db15',
           # 'db16', 'db17', 'db18', 'db19', 'db20',
           'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4',
           'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7',
           'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8',
           'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4',
           'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5',
           'rbio3.7', 'rbio3.9', 'rbio4.4', 
           # 'rbio5.5', 'rbio6.8',
           'dmey',
           'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9',
           'sym10',
           # 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16',
           # 'sym17', 'sym18', 'sym19', 'sym20'
           )

DEC_LEVEL = (1, 8)

MODELS = {
# =============================================================================
#           'MLP_Single'  : {'solver':             (0, 1),
#                            'activation':         (0, 3),
#                            'hidden_layer_sizes': (5, 200),
#                            'learning_rate_init': (0.01, 0.1)},
# =============================================================================
          
          'MLP_Multi'   :  {'solver':             (0, 1),
                            'activation':         (0, 3),
                            'hidden_layer_sizes': (5, 200),
                            '2layer':             (0, 100),
                            '3layer':             (0, 50),
                            'learning_rate_init': (0.01, 0.1)}, # ValueError: Expected 2D array, got 1D array instead:
          
          'XGB'         : {'eta':              (0, 1),
                           'gamma':            (0, 10),
                           'max_depth':        (0, 10),
                           'min_child_weight': (0, 10),
                           'max_delta_step':   (0, 10),
                           # 'supper_boundssample': (0.1, 1),
                           'lambda':           (0, 10),
                           'rounds':           (50, 150),
                           'n_estimators':     (10, 3000)},
          
#          'LSTM'        : {'first_layer':  (10, 15),
#                           'second_layer': (10, 20),
#                           'batch_size':   (3, 5)}, 
          
#          'CONV_LSTM'   : {'first_layer':  (10, 50),
#                           'second_layer': (10, 50),
#                           'batch_size':   (4, 9),
#                           'filters':      (2, 6),
#                           'dense_units':  (10, 30)},
          
#          'GMDH'        :  {'ref_functions':           (0, 3),
#                            'criterion_type':          (0, 3),
#                            'criterion_minimum_width': (1, 6),
#                            'max_layer_count':         (10, 50),
#                            'alpha':                   (0.1, 1)},
          
          'SVR'         : {'kernel':  (0, 3),
                           'degree':  (2, 4),
                           'C':       (0.5, 3.0),
                           'epsilon': (0.1, 0.2)},
      
          'LSSVR'       : {'kernel':  (0, 2),
                           'C':       (0.5, 3.0),
                           'gamma':   (0.01, 0.1)},
      
# =============================================================================
#           'ELM'         : {'alpha':                     (0, 10000),
#                            'include_original_features': (0, 1),  # bool
#                            '1_neurons':                 (1, 500),  # int or [int],
#                            '2_neurons':                 (0, 500),
#                            '3_neurons':                 (0, 500),
#                            '1_ufunc':                   (0, 3),
#                            '2_ufunc':                   (0, 3),
#                            '3_ufunc':                   (0, 3)},
# =============================================================================
#          
#          'ELM2'       : {'n_hidden' :          (1, 500),
#                          'alpha' :             (0, 10),
#                          'activation_func' :   (0, 7),
#                          'rbf_width' :         (0, 2)},
          
#          'LASSO'      : {'alpha' :    (0,10000),     # float [0, inf)
#                          'max_iter' : (500, 10000),  # int
#                          'tol' :      (1e-10,1e-2)}, # float
          
#          'RIDGE'      : {'alpha' :    (0,10000),     # float [0, inf)
#                          'max_iter' : (500, 10000),  # int
#                          'tol' :      (1e-10,1e-2)}, # float
          
#          'LINEAR'     : {'fit_intercept' : (0,1),  # bool
#                          'copy_X' :        (0,1),  # bool
#                          'positive' :      (0,1)}, # bool
           
          'ELASTIC_NET': {'alpha' :    (0,10000),    # float [0, inf)
                          'l1_ratio' : (0,1),        # float [0,1]
                          'max_iter' : (500, 10000), # int
                          'tol' :      (1e-10,1e-2), # float
                          'selection' : (0,1)},      # {‘cyclic’, ‘random’}
                    
            'KNN'      : {'n_neighbors' : (2,100),
                          'weights'     : (0,1),
                          'algorithm'   : (0,3),
                          'leaf_size'   : (10,100),
                          'p'           : (1,10)} 
          }


def config(forecast,forecast_type="daily"):
    
    model_list = tuple(MODELS.keys())
    
    models_params_list = []
    for v in MODELS.values():
        models_params_list.extend(v.keys())
        
    params_values_list = []
    for v in MODELS.values():
        params_values_list.extend(v.values())
    
    h = {"hourly":24,
         "daily":7,
         "monthly":6}
    
    # look_backs = (forecast + h[forecast_type],
                  # (forecast + h[forecast_type]) * 2)
    
    look_backs = (forecast + 1,
                   (h[forecast_type]) * forecast)


    # look_backs = (1, forecast) 
    
    cfg_list = [USE_WAVELETS,
                (0,len(FILTERS)-1),
                look_backs,
                DEC_LEVEL,
                (0,len(model_list)-1),
                ]
    
    cfg_list.extend(params_values_list)
    
    lower_bounds, upper_bounds = zip(*cfg_list)
    lower_bounds, upper_bounds = tuple(lower_bounds), tuple(upper_bounds)
    
    return lower_bounds, upper_bounds, #FILTERS, MODELS
    

def get_positions(estimator):
    # Checks if key exists
    if estimator in MODELS:
        # Gets the list of keys from the external dictionary
        external_keys = list(MODELS.keys())
        
        # Calculates the starting position considering the keys of the previous groups
        initial_position = 0
        for key in external_keys[:external_keys.index(estimator)]:
            initial_position += len(MODELS[key])  # Sums the number of keys from each previous dictionary
        
        # Generates a list of positions
        positions = [initial_position + i for i in range(len(MODELS[estimator]))]
        
        positions = [i + 5 for i in positions]
        
        return positions
    else:
        return f"The key '{estimator}' does not exist."


def get_models(k):
    
    models = list(MODELS.keys())
    
    return models[k]


def get_params(estimator):
    
    p = list(MODELS[estimator].keys())
    
    return p


def get_filters(k):
    
    return FILTERS[k]


if __name__ == "__main__":
    lower_bounds, upper_bounds  = config(7)
    
    idx = get_positions(estimator="ELM")
    # idx = [i + 4 for i in idx]
    
    model = get_models(0)
    filter = get_filters(0)
    params = get_params(model)
