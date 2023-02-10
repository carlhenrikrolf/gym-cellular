import numpy as np
import gymnasium as gym
from os import system
system('cd ..; pip3 install -e gym-cellular -q')
import gym_cellular
from gym_cellular.envs import PolarisationV2Env as LocalEnv

env = gym.make("gym_cellular/Polarisation-v2", seed=0)
#x = env.tabular_reward_function(0,0)
local_env = LocalEnv(seed=0)
flat_observation = 9
observation = local_env._inverse_tabular_encoding(flat_observation)
print(flat_observation, local_env.tabular_encoding(observation))

flat_action = 0
action = local_env.tabular_decoding(flat_action)
print(flat_action, local_env._inverse_tabular_decoding(action))