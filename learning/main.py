from envs.rover_domain.rover_domain_python import RoverDomain
from ccea import CCEA
import numpy as np
from tensorboardX import SummaryWriter
import time
from pathlib import Path
import os
import csv
import copy


class Params:
    def __init__(self):
        # Environment parameters
        self.dim_x = 15         # world size
        self.dim_y = self.dim_x
        self.obs_radius = 125   # observability
        self.act_dist = 5       # how far the rovers needs to be to activate a POI
        self.angle_res = 90     # angle resolution

        self.num_poi = 5       # number of POIs
        self.num_agents = 5    # number of agents
        self.ep_len = 50       # episode length
        self.poi_rand = True   # initialize POI randomly
        self.coupling = 3       # Coupling
        self.rover_speed = 1    # default is 1
        self.sensor_model = 'density'   # 'closest', 'density'

        self.communication = False
        self.n_comm_bits = int(360/self.angle_res)*self.communication
        self.action_dim = 2 + self.communication*self.n_comm_bits     # two physical actions + quadrants
        self.comm_one_hot = True  # ONE-HOT VS SOFTMAX

        # CCEA parameters
        self.population_size = 10
        self.mutation_rate = 0.05

        # For Neural Network Policies
        self.nn_input_size = None
        self.nn_output_size = self.action_dim
        self.nn_hidden_size = 16


def get_env_setting():
    setting = {"communication": params.communication,
               "comm_one_hot": params.comm_one_hot,
               "n_comm_bits": params.n_comm_bits,
               "n_agents" : params.num_agents,
               "n_pois": params.num_poi,
               "coupling": params.coupling,
               "n_steps": params.ep_len,
               "setup_size": params.dim_x,
               "act_dist": params.act_dist,
               "obs_radius": params.obs_radius,
               "sensor_model": params.sensor_model,
               "angle_res": params.angle_res,
               "poi_rand": params.poi_rand,
               "episode_length" : params.ep_len,
               "rover_speed": params.rover_speed,
               "action_dim": params.act_dist,
               "population_size": params.population_size,
               "mutation_rate": params.mutation_rate,
                "timestr": timestr,
               "NN_hidden_size": params.nn_hidden_size
                }
    return setting


if __name__=="__main__":
    params = Params()
    # initialize the environment
    env = RoverDomain(params)
    joint_obs = env.reset()

    # Initialize params for CCEA

    params.nn_input_size = len(joint_obs[0])  # todo: fix this
    params.nn_output_size = params.action_dim

    # -----------------------------------------------------------------------------------------------------------------#
    # Logger and save experiment setting
    # -----------------------------------------------------------------------------------------------------------------#
    timestr = time.strftime("__%m%d-%H%M%S")
    file_path = Path('./Experiments') / ("R"+str(params.num_agents) + "-P"+str(params.num_poi) + "-Cp"+str(params.coupling) + timestr)
    os.makedirs(file_path)

    # save setting
    setting = get_env_setting()
    with open(str(file_path) +'/settings.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in setting.items():
            writer.writerow([key, value])

    logger = SummaryWriter(str(file_path))

    # -----------------------------------------------------------------------------------------------------------------#
    # Run Algorithm
    # -----------------------------------------------------------------------------------------------------------------#

    # Initialize algorithm
    ccea = CCEA(params)
    test_team_runs = 5      # eval runs for best_team

    episodes = 10000
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
            while not done:
                # List of observations for list of agents
                joint_obs = np.array(env.get_joint_state())

                # List of actions for a list of agents. Actions stored in ccea.joint_action
                ccea.get_team_action(joint_obs)
                agent_actions = np.array([np.double(ac.data[0]) for ac in ccea.joint_action])

                # Step
                _, _, done, _ = env.step(agent_actions)  # obs, step_rewards, done, self

            # Reward after running entire trajectory
            global_traj_reward = env.get_global_traj_reward()
            ccea.assign_fitness(global_traj_reward)        # Appends to fitness_list

        # Average the fitness_list
        for a in ccea.agents:
            for pol in a.population:
                pol.fitness = np.mean(pol.fitness_list)
                pol.fitness_list = []
        # selection. Back to K policies for M agents. Also makes a best_policy team
        ccea.selection()

        # ------------------------------------------------------------------------------------------------------------#
        # Evaluate the best team. This is the fitness that we record for learning
        # ------------------------------------------------------------------------------------------------------------#
        ccea.team = copy.deepcopy(ccea.best_team)
        fitness = []        # assuming fitness is always positive
        flag_once = True    # For rendering once
        for _ in range(test_team_runs):
            env.reset()
            done = False
            while not done:
                joint_obs = np.array(env.get_joint_state())
                ccea.get_team_action(joint_obs)
                agent_actions = np.array([np.double(ac.data[0]) for ac in ccea.joint_action])
                _, _, done, _ = env.step(agent_actions)  # obs, step_rewards, done, self

            global_traj_reward = env.get_global_traj_reward()
            fitness.append(global_traj_reward)  # All agents have same global reward.
            if flag_once:
                env.render()
                flag_once = False
        fitness = np.mean(fitness)      # averaging over eval runs

        # Logging
        logger.add_scalar('mean_team_fitness', fitness, ep_i)

        # Save model
        if not ep_i % 100:
            os.makedirs(file_path / 'models', exist_ok=True)
            ccea.save(file_path / 'models' / ('model_ep%i.pt' % (ep_i)))
            print("Experiment Filepath = " + str(file_path))

        # Print Episode and fitness
        print("Episode:"+str(ep_i) + "  Fitness:" + str(fitness))
