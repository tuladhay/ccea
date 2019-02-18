import numpy as np
from agent import NeuralNetworkAgent as Agent
import random
import torch
import fastrand, math
from torch.autograd import Variable
import copy
# if fastrand is giving problems about attribute, make sure /home/lib/python' is in your source path.


class CCEA:

    def __init__(self, params):
        '''
        :param params: parameters that contain details for CCEA
        population_size     : number of policies for each agent to evolve
        n_agents            : total number of agents in the environment (each has a population of policies)
        mut_prob            : probability of mutation
        nn_input_size, nn_output_size, nn_hidden_size, lr: neural network initialization parameters
        agents              : agent with its own population of policies
        '''
        self.population_size = params.population_size  # Note. After mutation, this isn't the same
        self.n_agents = params.num_agents
        self.mut_prob = params.mutation_rate

        # Communication
        self.comm = params.communication
        self.n_comm_bits = params.n_comm_bits

        # For Neural Network Policies
        self.nn_input_size = params.nn_input_size
        self.nn_output_size = params.nn_output_size
        self.nn_hidden_size = params.nn_hidden_size

        self.agents = [Agent(i,
                             self.nn_input_size,
                             self.nn_output_size,
                             self.nn_hidden_size,
                             self.population_size
                             ) for i in range(self.n_agents)]
        self.team = []
        self.best_team = []
        self.random_team_list = []
        self.joint_action = []
        self.leniency_evals = 5     # each policy evaluated n times with other policies for leniency

        self.init_dict = {'n_agents' : self.n_agents,
                         'hidden_dim': self.nn_hidden_size,
                         'input_dim': self.nn_input_size,
                         'output_dim': self.nn_output_size,
                         'population_size': self.population_size,
                         'mut_prob': self.mut_prob}

    def reset_teambuilder(self):
        agent_lineup = []
        for n in range(self.n_agents):
            agent_lineup.append(np.arange(len(self.agents[0].population)))
            agent_lineup[n] = np.repeat(agent_lineup[n], self.leniency_evals)
            np.random.shuffle(agent_lineup[n])
        self.random_team_list = agent_lineup
        # this is good. arrays are reshuffled within themselves

    def make_team(self):
        '''
        Randomly choose a policy from the population of each of the agent, and return a team fitness
        '''
        # pop from each agent list
        team = []
        for i, a in enumerate(self.random_team_list):
            index = a[0]    # grab the index for policy to use
            self.random_team_list[i] = np.delete(a, 0)      # update the list by popping
            team.append(self.agents[i].population[index])
        self.team = team
        # this is good

    def get_team_action(self, joint_observation):
        joint_action = []
        # rearrange observations to be per agent, and convert to torch Variable
        torch_obs = [Variable(torch.Tensor(joint_observation[i]).view(1, -1),
                            requires_grad=False)
                            for i in range(self.n_agents)]
        for pol, obs in zip(self.team, torch_obs):
            joint_action.append(pol.forward(obs))
        self.joint_action = joint_action

    def assign_fitness(self, team_fitness):
        for policy in self.team:
            # To take max(...) for leniency evaluation
            #if team_fitness > policy.fitness:
            #    policy.fitness = team_fitness
            policy.fitness_list.append(team_fitness)

    def mutate(self):
        ''' For each agent, create 2*population by mutation.
        NOTE: self.population_size doesn't currently reflect this increase
        '''

        def regularize_weight(weight, mag):
            if weight > mag: weight = mag
            if weight < -mag: weight = -mag
            return weight

        def mutate_inplace(gene):
            '''
            Mutates a neural network policy (in place)
            todo: what are the parameters
            '''

            mut_strength = 0.1
            num_mutation_frac = 0.1
            super_mut_strength = 10
            super_mut_prob = self.mut_prob  # was 0.05
            reset_prob = super_mut_prob + 0.05

            num_params = len(list(gene.parameters()))
            # ssne_probabilities = np.random.uniform(0, 1, num_params) * 2
            model_params = gene.state_dict()

            for i, key in enumerate(model_params):  # Mutate each param

                if key == 'lnorm1.gamma' or key == 'lnorm1.beta' or key == 'lnorm2.gamma' or key == 'lnorm2.beta' or key == 'lnorm3.gamma' or key == 'lnorm3.beta': continue

                # References to the variable keys
                W = model_params[key]
                if len(W.shape) == 2:  # Weights, no bias

                    num_weights = W.shape[0] * W.shape[1]
                    ssne_prob = 1  # ssne_probabilities[i]

                    if random.random() < ssne_prob:
                        num_mutations = fastrand.pcg32bounded(
                            int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
                        for _ in range(num_mutations):
                            ind_dim1 = fastrand.pcg32bounded(W.shape[0])
                            ind_dim2 = fastrand.pcg32bounded(W.shape[-1])
                            random_num = random.random()

                            if random_num < super_mut_prob:  # Super Mutation probability
                                W[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[ind_dim1, ind_dim2])
                            elif random_num < reset_prob:  # Reset probability
                                W[ind_dim1, ind_dim2] = random.gauss(0, 1)
                            else:  # mutauion even normal
                                W[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * W[ind_dim1, ind_dim2])

                            # Regularization hard limit
                            W[ind_dim1, ind_dim2] = regularize_weight(W[ind_dim1, ind_dim2], 1000000)

        for agent in self.agents:
            mutated_population = []
            for pol in agent.population:
                offspring = copy.deepcopy(pol)
                mutate_inplace(offspring)
                mutated_population.append(offspring)
            # add the mutated population to make 2*K population of M agents

            for pol in mutated_population:
                pol.fitness = 0.0

            agent.population += mutated_population
            #Todo: should fitness = 0 for mutated?

    def selection(self):
        # Select the best K policies in the team. Saves best policies in a team.
        self.best_team = []
        for agent in self.agents:
            agent.population.sort(key=lambda x: x.fitness, reverse=True)  # FIX THIS
            agent.population = agent.population[:self.population_size]  # select K best
            agent.best_policy = agent.population[0]
            self.best_team.append(agent.best_policy)

    def save(self, filename):
        """
        Save trained pgitarameters of all agents into one file
        """
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)
