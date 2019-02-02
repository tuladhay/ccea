# Run file by itself (or import it?) Comment out following code if building 
# from command line. (setup default)
import sys
import numpy
old_sys_argv = sys.argv[:]
sys.argv = ['', 'build_ext', '--inplace']

import rover_domain_setup

sys.argv = old_sys_argv

from rover_domain import *

from ccea import CCEA
import numpy as np
from tensorboardX import SummaryWriter
import time
from pathlib import Path
import os
import csv


class Params:
    def __init__(self):
        self.population_size = 10
        self.n_agents = None
        self.mutation_rate = 0.01

        # For Neural Network Policies
        self.nn_input_size = None
        self.nn_output_size = None
        self.nn_hidden_size = 16

def get_env_setting():
    setting = {"n_agents" : env.n_rovers,
               "n_pois": env.n_pois,
               "n_req": env.n_req,
               "n_steps": env.n_steps,
               "setup_size": env.setup_size,
               "min_dist": env.min_dist,
               "interaction_dist": env.interaction_dist,
               "reorients": env.reorients,
               "discounts_eval": env.discounts_eval,
               "n_obs_sections": env.n_obs_sections,
                "timestr": timestr
                }
    return setting


if __name__=="__main__":

    # initialize the environment
    env = RoverDomain()
    env.reset()

    # Initialize params for CCEA
    params = Params()
    params.n_agents = env.n_rovers
    params.nn_input_size = env.rover_observations.base[0].size
    params.nn_output_size = 2  # set this automatically

    # Logger and save experiment setting
    timestr = time.strftime("__%m%d-%H%M%S")
    file_path = Path('./Experiments') / ("R"+str(env.n_rovers) + "-P"+str(env.n_pois) + "-Cp"+str(env.n_req) + timestr)
    os.makedirs(file_path)

    # save setting
    setting = get_env_setting()
    with open('settings.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in setting.items():
            writer.writerow([key, value])

    logger = SummaryWriter(str(file_path))

    # Initialize algorithm
    ccea = CCEA(params)

    # episode_fitness = []
    episodes = 10000
    for ep_i in range(episodes):
        # Makes population of 2*K policies for M agents
        ccea.mutate()

        # reset team builder. Makes team without replacement for leniency.
        ccea.reset_teambuilder()
        # This evaluation loop is for CCEA (see lecture slides)
        for _ in range(50):  # run n number of times for leniency evaluation.
            # List is popped, so will run out if make_team is run more than ccea.lenienvy eval times.
            env.reset()

            ccea.make_team()

            done = False
            # --- Run entire trajectory using this team policy --- #
            while not done:
                # List of observations for list of agents
                joint_obs = [env.rover_observations.base[i].flatten() for i in range(params.n_agents)]

                # get action from policies in team
                # List of actions for a list of agents. Actions stored in ccea.joint_action
                ccea.get_team_action(joint_obs)
                agent_actions = np.array([np.double(ac.data[0]) for ac in ccea.joint_action])

                # Step
                _, _, done, _ = env.step(agent_actions)  # obs, step_rewards, done, self

            # Reward after running entire trajectory
            env.update_rewards_traj_global_eval()
            fitness = env.rover_rewards[0]      # All agents have same global reward.

            #
            ccea.assign_fitness(fitness)

        # selection. Back to K policies for M agents
        ccea.selection()

        # Logging
        for i, a in enumerate(ccea.agents):
            logger.add_scalar('agent%i/mean_episode_rewards' % i, a.best_policy.fitness, ep_i)

        # Save model
        if not ep_i % 100:
            os.makedirs(file_path / 'models', exist_ok=True)
            ccea.save(file_path / 'models' / ('model_ep%i.pt' % (ep_i)))

        # best fitness
        # best_fitness = [a.fitness for a in ccea.best_team]
        best_fitness = [a.fitness for a in ccea.best_team]
        print("Episode:"+str(ep_i) + "  Fitness:" + str(best_fitness))

    print()
