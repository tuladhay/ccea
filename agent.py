import numpy as np
import networks
from networks import MLPNetwork


class NeuralNetworkAgent:
    ''' Agent is not a single agent. It is a population of policies '''
    def __init__(self, name, nn_input_size, nn_output_size, nn_hidden_size,
                 population_size,
                 ):
        self.name = name
        self.state = []     # Being used somewhere?
        self.comm = []      # Being used somewhere?
        self.action = []    # Being used somewhere?
        self.population = [MLPNetwork(nn_input_size, nn_output_size,
                           nn_hidden_size)
                           for _ in range(population_size)]
        self.best_policy = None
        self.initialize_fitness()

    def initialize_fitness(self):
        # Adds "fitness" to all the genes in the agent's population
        for policy in self.population:
            policy.fitness = 0.0
            policy.train(False)

    def get_params(self):
        return {'best_policy': self.best_policy.state_dict()}
