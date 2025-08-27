#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:32:37 2024

@author: rodrigo
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

VERBOSE = False


class WANN:
    def __init__(self, args, fun, lb, ub):
        self.args = args
        self.obj = fun
        self.lb = lb
        self.ub = ub

    def fitness(self, x):
        self.result = self.obj(x, *self.args)
        return self.result
    

def get_heuristica(name, fun, lb, ub,pop,it,TARGET_VALUE = None):    #TODO: Alterar aqui o nome

    if name == "pso":
        from pyMetaheuristic.algorithm import particle_swarm_optimization as meta
        # from util.heuristicas import particle_swarm_optimization as meta      #TODO: Alterar ou remover isso aqui 
        parameters = {
            'swarm_size': pop,
            'min_values': lb,
            'max_values': ub,
            'iterations': it,
            'decay': 0,
            'w': 0.9,
            'c1': 2,
            'c2': 2,
            'verbose': VERBOSE,
            'start_init': None,
            'target_value': TARGET_VALUE
        }
        
    elif name == "de":
        from pyMetaheuristic.algorithm import differential_evolution as meta
        # from util.heuristicas import differential_evolution as meta       #TODO: Alterar ou remover isso aqui
        # from util.heuristicas import diff_parallel as meta                #TODO: Alterar ou remover isso aqui
        parameters = {
            'n': pop,
            'min_values': lb,
            'max_values': ub,
            'iterations': it,
            'F': 0.9,
            'Cr': 0.2,
            'verbose': VERBOSE,
            'start_init': None,
            'target_value': TARGET_VALUE
        }

    elif name == "gwo":
        from pyMetaheuristic.algorithm import grey_wolf_optimizer as meta
        parameters = {
            'pack_size': pop,
            'min_values': lb,
            'max_values': ub,
            'verbose': VERBOSE,
            'iterations': it,
            'start_init': None,
            'target_value': TARGET_VALUE
        }

    elif name == "cso":
        from pyMetaheuristic.algorithm import chicken_swarm_optimization as meta
        # from util.heuristicas import chicken_swarm_optimization as meta              #TODO: Alterar ou remover isso aqui
        parameters = {
            'size': pop,
            'min_values': lb,
            'max_values': ub,
            'generations': it,
            'g': 5,
            'rooster_ratio': 0.2,
            'hen_ratio': 0.6,
            'verbose': VERBOSE,
            'start_init': None,
            'target_value': TARGET_VALUE
        }

    elif name == "sa":
        from pyMetaheuristic.algorithm import simulated_annealing as meta
        # from util.heuristicas import simulated_annealing as meta                #TODO: Alterar ou remover isso aqui
        parameters = {
            'min_values': lb,
            'max_values': ub,
            'alpha': 0.9,
            'mu': 0,
            'sigma': 1,
            'temperature_iterations': 5,
            'initial_temperature': 15.0,
            'final_temperature': 0.5,
            'verbose': VERBOSE,
            'start_init': None,
            'target_value': TARGET_VALUE
        }

    elif name == "ga":
        from pyMetaheuristic.algorithm import genetic_algorithm as meta
        # from util.heuristicas import genetic_algorithm as meta           #TODO: Alterar ou remover isso aqui
        parameters = {
            'population_size': pop,
            'min_values': lb,
            'max_values': ub,
            'generations': it,
            'mutation_rate': 0.1,
            'elite': 1,
            'eta': 1,
            'mu': 1,
            'verbose': VERBOSE,
            'start_init': None,
            'target_value': TARGET_VALUE
        }

    elif name == "igwo":
        from pyMetaheuristic.algorithm import improved_grey_wolf_optimizer as meta 
        # from util.heuristicas import improved_grey_wolf_optimizer as meta      #TODO: Alterar ou remover isso aqui
        parameters = {
            'pack_size': pop,
            'min_values': lb,
            'max_values': ub,
            'iterations': it,
            'verbose': VERBOSE,
            'start_init': None,
            'target_value': TARGET_VALUE
        }

    elif name == "fda":
        from pyMetaheuristic.algorithm import flow_direction_algorithm as meta
        # from util.heuristicas import flow_direction_algorithm as meta            #TODO: Alterar ou remover isso aqui
        parameters = {
            'size': pop,
            'min_values': lb,
            'max_values': ub,
            'iterations': it,
            'beta': 8,
            'verbose': VERBOSE,
            'start_init': None,
            'target_value': TARGET_VALUE
        }
    
    else:
        raise NameError(f"Heuristic -{name}- is not defined!")

    return meta(target_function=fun, **parameters)
