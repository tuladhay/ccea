from ccea import CCEA
import numpy as np
import copy
from xor import XOR


class Params:
    def __init__(self):
        # Environment parameters
        self.dim_x = 20         # world size
        self.dim_y = self.dim_x
        self.obs_radius = 10    # observability
        self.act_dist = 5       # how far the rovers needs to be to activate a POI
        self.angle_res = 90     # angle resolution

        self.num_poi = 1       # number of POIs
        self.num_agents = 1    # number of agents
        self.ep_len = 50       # episode length
        self.poi_rand = True    # initialize POI randomly?
        self.coupling = 1       # Coupling
        self.rover_speed = 1    # default is 1
        self.sensor_model = 'closest'   # 'closest', 'density'

        self.action_dim = 2     # two physical actions

        # CCEA parameters
        self.population_size = 20
        self.mutation_rate = 0.01

        # For Neural Network Policies
        self.nn_input_size = None
        self.nn_output_size = None
        self.nn_hidden_size = 16


if __name__=="__main__":
    params = Params()
    # initialize the environment
    env = XOR()
    obs = env._get_obs()

    # Initialize params for CCEA
    params.nn_input_size = 2
    params.nn_output_size = 1  # set this automatically

    # -----------------------------------------------------------------------------------------------------------------#
    # Logger and save experiment setting
    # -----------------------------------------------------------------------------------------------------------------#

    # -----------------------------------------------------------------------------------------------------------------#
    # Run Algorithm
    # -----------------------------------------------------------------------------------------------------------------#

    # Initialize algorithm
    ccea = CCEA(params)
    test_team_runs = 10      # eval runs for best_team

    episodes = 1000
    for ep_i in range(episodes):
        # Makes population of 2*K policies for M agents
        ccea.mutate()
        # reset team builder. Makes team without replacement for leniency.
        ccea.reset_teambuilder()

        # This evaluation loop is for CCEA (see lecture slides). Leniency evals * mutated population size
        for _ in range(ccea.leniency_evals*len(ccea.agents[0].population)):  # run n number of times for leniency evaluation.
            # List is popped, so will run out if make_team is run more than ccea.leniency eval times.
            env.reset()
            ccea.make_team()

            done = False
            # --- Run entire trajectory using this team policy --- #
            reward = 0.0
            while not done:
                # List of observations for list of agents
                obs = env.current_state
                obs = [np.array(obs)]
                # List of actions for a list of agents. Actions stored in ccea.joint_action
                ccea.get_team_action(obs)
                agent_actions = np.array([np.double(ac.data[0]) for ac in ccea.joint_action])
                agent_actions = int(agent_actions)
                # Step
                _, reward, done = env.step(agent_actions)  # obs, step_rewards, done, self

            # Reward after running entire trajectory
            #fitness = env.rover_rewards[0]      # All agents have same global reward.
            ccea.assign_fitness(reward)        # Appends

        # selection. Back to K policies for M agents. Also makes a best_policy team
        for a in ccea.agents:
            for pol in a.population:
                pol.fitness = np.mean(pol.fitness_list)
                pol.fitness_list = []

        ccea.selection()

        # ------------------------------------------------------------------------------------------------------------#
        # Evaluate the best team. This is the fitness that we record for learning
        # ------------------------------------------------------------------------------------------------------------#
        ccea.team = copy.deepcopy(ccea.best_team)
        fitness = []        # assuming fitness is always positive
        for _ in range(test_team_runs):
            env.reset()
            done = False
            reward = None
            while not done:
                obs = np.array(env.current_state)
                obs = [np.array(obs)]
                ccea.get_team_action(obs)
                agent_actions = np.array([np.double(ac.data[0]) for ac in ccea.joint_action])
                agent_actions = int(agent_actions)
                _, reward, done = env.step(agent_actions)  # obs, step_rewards, done, self

            fitness.append(reward)  # All agents have same global reward.
        fitness = np.mean(fitness)      # averaging over eval runs
        # Logging

        # Print Episode and fitness
        print("Episode:"+str(ep_i) + "  Fitness:" + str(fitness))
