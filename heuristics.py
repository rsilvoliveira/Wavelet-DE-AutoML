#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:32:37 2024

@author: rodrigo
"""

import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent))


class WANN:
    def __init__(self, args: List, function, lower_bounds: List, upper_bounds: List):
        self.args = args
        self.obj = function
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def fitness(self, x: List) -> float:
        self.result = self.obj(x, *self.args)
        return self.result
    

def get_heuristic(name: str, function, lower_bounds: List, upper_bounds: List,population: int,iteractions: int,target_value: float = None,verbose: bool = False):

    if name == "pso":
        from util.heuristics import particle_swarm_optimization as meta
        parameters = {
            'swarm_size': population,
            'min_values': lower_bounds,
            'max_values': upper_bounds,
            'iterations': iteractions,
            'decay': 0,
            'w': 0.9,
            'c1': 2,
            'c2': 2,
            'verbose': verbose,
            'start_init': None,
            'target_value': target_value
        }
        
    elif name == "de":
        from util.heuristics import differential_evolution as meta
        parameters = {
            'n': population,
            'min_values': lower_bounds,
            'max_values': upper_bounds,
            'iterations': iteractions,
            'F': 0.9,
            'Cr': 0.2,
            'verbose': verbose,
            'start_init': None,
            'target_value': target_value
        }

    elif name == "gwo":
        from util.heuristics import grey_wolf_optimizer as meta
        parameters = {
            'pack_size': population,
            'min_values': lower_bounds,
            'max_values': upper_bounds,
            'verbose': verbose,
            'iterations': iteractions,
            'start_init': None,
            'target_value': target_value
        }

    elif name == "cso":
        from util.heuristics import chicken_swarm_optimization as meta
        parameters = {
            'size': population,
            'min_values': lower_bounds,
            'max_values': upper_bounds,
            'generations': iteractions,
            'g': 5,
            'rooster_ratio': 0.2,
            'hen_ratio': 0.6,
            'verbose': verbose,
            'start_init': None,
            'target_value': target_value
        }

    elif name == "sa":
        from util.heuristics import simulated_annealing as meta
        parameters = {
            'min_values': lower_bounds,
            'max_values': upper_bounds,
            'alpha': 0.9,
            'mu': 0,
            'sigma': 1,
            'temperature_iterations': 5,
            'initial_temperature': 15.0,
            'final_temperature': 0.5,
            'verbose': verbose,
            'start_init': None,
            'target_value': target_value
        }

    elif name == "ga":
        from util.heuristics import genetic_algorithm as meta
        parameters = {
            'population_size': population,
            'min_values': lower_bounds,
            'max_values': upper_bounds,
            'generations': iteractions,
            'mutation_rate': 0.1,
            'elite': 1,
            'eta': 1,
            'mu': 1,
            'verbose': verbose,
            'start_init': None,
            'target_value': target_value
        }

    elif name == "igwo":
        from util.heuristics import improved_grey_wolf_optimizer as meta
        parameters = {
            'pack_size': population,
            'min_values': lower_bounds,
            'max_values': upper_bounds,
            'iterations': iteractions,
            'verbose': verbose,
            'start_init': None,
            'target_value': target_value
        }

    elif name == "fda":
        from util.heuristics import flow_direction_algorithm as meta
        parameters = {
            'size': population,
            'min_values': lower_bounds,
            'max_values': upper_bounds,
            'iterations': iteractions,
            'beta': 8,
            'verbose': verbose,
            'start_init': None,
            'target_value': target_value
        }
    
    else:
        raise NameError(f"Heuristic -{name}- is not defined!")

    return meta(target_function=function, **parameters)
