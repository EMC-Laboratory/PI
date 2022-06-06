# Module to generate initial population

import torch
import random
import numpy as np
import copy

class Population():
    def __init__(self, opt):
        self.generations = 50
        self.current_gen = 1
        self.population = 50 # initial population solution set
        self.numDecaps = opt.num_decaps  # total different number of possible decaps at a location
        self.chromosomes = None
        self.mutationRate = .1
        self.crossoverRate = .6  # for
        self.stagnateCounter = 0
        self.startGen = 1
        #self.set_initial_pop(opt)

    def return_chromosomes(self):
        return(copy.deepcopy(self.chromosomes))

    def increment_stagnate_counter(self):
        self.stagnateCounter = self.stagnateCounter + 1

    def reset_stagnate_counter(self):
        self.stagnateCounter = 0

    def create_population(self,opt):
        self.chromosomes = np.random.randint(self.numDecaps + 1, size=(self.population, opt.decap_ports))
        #self.chromosomes = [[random.randrange(self.numDecaps+1) for i in range(opt.decap_ports)] for j in range(self.population)]

    def create_fixed_population(self,opt, default_chromosome = 0):
        self.chromosomes = [[default_chromosome for i in range(opt.decap_ports)] for y in range(self.population)]

    def create_null_population(self,opt):
        self.chromosomes = np.zeros((self.population,opt.decap_ports), dtype= int)

    def create_custom_population(self, opt, min_cap, bulk_cap):
        #self.chromosomes = np.random.randint(min_cap , high= bulk_cap + 1, size=(self.population, opt.decap_ports))
        self.chromosomes = np.random.randint(min(min_cap,bulk_cap), high=(max(min_cap,bulk_cap))+1, size=(self.population, opt.decap_ports))

    def generate_chromosome(self, opt):
        return np.random.randint(opt.num_decaps + 1, size=(1, opt.decap_ports))

    def change_mutation_rate(self,new_rate):
        self.mutationRate = new_rate

    def increment_generation(self):
        self.current_gen = self.current_gen + 1

    def set_generation(self, gen_num):
        self.current_gen = gen_num




    #def set_initial_pop(self, opt):
     #   for i in range(self.population):  # Number of chromosomes
      #      for j in range(opt.total_ports-1):
       #         self.chromosomes[i][j] = random.randrange(0, self.numDecaps)  # generates 0 to 10
        #        # opt.total_ports = total number of ports on board. Only use 15-1 ports for decaps




