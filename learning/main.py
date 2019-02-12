# from envs.rover_domain.env_wrapper import RoverDomainPython
from envs.rover_domain.rover_domain_python import RoverDomain
import numpy as np

class Params:
    def __init__(self):
        # Environment parameters
        self.dim_x = 20         # world size
        self.dim_y = self.dim_x
        self.obs_radius = 10    # observability
        self.act_dist = 5       # how far the rovers needs to be to activate a POI
        self.angle_res = 90     # angle resolution

        self.num_poi = 10       # number of POIs
        self.num_agents = 10    # number of agents
        self.ep_len = 100       # episode length
        self.poi_rand = True    # initialize POI randomly?
        self.coupling = 4       # Coupling
        self.rover_speed = 1    # default is 1
        self.sensor_model = 'closest'   # 'closest', 'density'

        self.action_dim = 2     # two physical actions

        # CCEA parameters
        self.population_size = 10
        self.n_agents = None
        self.mutation_rate = 0.01

        # For Neural Network Policies
        self.nn_input_size = None
        self.nn_output_size = None
        self.nn_hidden_size = 16


if __name__=="__main__":
    params = Params()
    # env = RoverDomainPython(params, 1)
    env = RoverDomain(params)
    joint_obs = env.reset()

    for ep in range(10):
        env.reset()
        global_traj_reward = None
        while not env.done:
            joint_action = [np.random.rand(2) for _ in range(env.args.num_agents)]
            _, _, _, global_traj_reward = env.step(joint_action)
            # print(global_traj_reward)
            # print(env.done)
        print("Episode:" +str(ep) + " Global Reward:" + str(global_traj_reward))

    print("Done")
