import numpy as np
import gymnasium as gym
from os import system
system('cd ../..; pip3 install -e gym-cellular -q')
import gym_cellular
from gym_cellular.envs import PolarisationV2Env as AccessiblePolarisationV2Env

class Test_Polarisation:

    def test_instantiation(self):

        for seed in range(10):
            env = gym.make("gym_cellular/Polarisation-v2", seed=seed)
            assert isinstance(env, gym.Env)

    def test_reset(self):

        env = gym.make("gym_cellular/Polarisation-v2", seed=0)
        for seed in range(10):
            state, info = env.reset(seed=seed)
            assert isinstance(state, dict)
            assert isinstance(info, dict)

    def test_inverse_tabular_encoding(self):

        env = AccessiblePolarisationV2Env(seed=0)
        observation = env._inverse_tabular_encoding(0)
        assert type(observation) is dict
        assert (observation["polarisation"] == 0).all()
        assert (observation["two_way_polarisable"] == 0).all()

    def test_tabular_both_encodings(self):

        env = AccessiblePolarisationV2Env(seed=0)
        for flat_observation in range(10):
            observation = env._inverse_tabular_encoding(0)
            new_flat_observation = env.tabular_encoding(observation)
            assert new_flat_observation == flat_observation