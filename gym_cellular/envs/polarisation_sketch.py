import numpy as np

import gymnasium as gym
from gymnasium import spaces

class PolarisationEnv(gym.Env):
	
	"""
	My own environment. Thanks to Luke Thorburn for idea
	"""
	
	# metadata
	
	def __init__(self, render_mode=None, n_users=2, n_user_states=8, n_moderators=2, recommendations_seed=1):
		
		assert n_moderators <= n_users
		self.n_users = n_users
		self.n_user_states = n_user_states
		self.n_moderators = n_moderators
		self.recomendations_seed = recomendations_seed
		
		self.observation_space = spaces.Box(0, n_user_states - 1, shape=(n_users,), dtype=int)
		self.action_space = spaces.MultiDiscrete(np.zeros((n_cells,), dtype=int) + 2) # two actions per cell
		# leftist content or rightist content
		
		assert render_mode is None
		
	def _get_obs(self)
		return self._polarisation

	def _get_info(self)
		return 'not applicable' # I dont know what to say yet

	def _intracellular_step(self):
		
		self.recommendations_seed
		
		action_mapping = np.zeros(
		for user_state in range(0, n_user_states - 1):
			

	def reset(self, seed=None, options=None):
		
		self._polarisation = self.np_random.integers(0, n_user_states - 1, size=n_users, dtype=int)
		# I still have to ensure that initial state is safe
		
		observation = self._get_obs()
		info = self._get_info()
		
		return observation, info
		
	def step(self, action):
		
		
		terminated = False
		truncated = False # I need to change this to some kind of time limit
		info = self._get_info()
		
		return observation, reward, side_effects, terminated, truncated, info
