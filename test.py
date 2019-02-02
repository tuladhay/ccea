from ccea import CCEA
import numpy as np

class Params:

    def __init__(self):
        self.population_size = 10
        self.n_agents = 6
        self.mutation_rate = 0.01

        # For Neural Network Policies
        self.nn_input_size = 8
        self.nn_output_size = 2
        self.nn_hidden_size = 16


if __name__=="__main__":
    params = Params()
    ccea = CCEA(params)

    # Makes population of 2*K policies for M agents
    ccea.mutate()

    for _ in range(100):
        ccea.make_team()

        # dummy observation.
        # Must be a list of observations for list of agents
        joint_obs = np.random.uniform(-1,1, size=(params.n_agents, params.nn_input_size))

        # Calculate action from policies in team. List of actions for a list of agents
        ccea.get_team_action(joint_obs)

        # Take a step? Or run the whole trajectory? But how?

        # dummy reward after running through environment
        reward = np.random.uniform(0, 10)
        ccea.assign_fitness(reward)

    # selection. Back to K policies for M agents
    ccea.selection()

    print()
