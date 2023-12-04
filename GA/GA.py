'''
author: Xu Junzhe
date: 2023/12/1
unicoding: utf-8
'''
import numpy as np
import random
from stA_star import *

class GA():
    def __init__(self, raw_map, start, end, ST_Table, population_size=100, max_iter=100, cross_rate=0.8, mutation_rate=0.01):
        self.raw_map = raw_map
        self.start = start
        self.end = end
        self.ST_Table = ST_Table
        self.population_size = population_size
        self.max_iter = max_iter
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.population = self.init_population()
        self.fitness = self.get_fitness()
        self.best_path = []
        self.best_fitness = 0
        self.iter = 0