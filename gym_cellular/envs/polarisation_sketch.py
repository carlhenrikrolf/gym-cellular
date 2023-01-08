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
		self.recommendations_seed = recommendations_seed
		
		self.observation_space = spaces.Box(0, n_user_states - 1, shape=(n_users,), dtype=int)
		self.action_space = spaces.MultiDiscrete(np.zeros((n_users,), dtype=int) + 2) # two actions per cell
		# leftist content or rightist content

		np.random.seed(seed=recommendations_seed)

		# recommendations given a state that polarise in a right directions indicated by 1. If not right polarising, then left polarising.
		_right_polarising_actions = np.random.rand(n_user_states)
		_right_polarising_actions = np.rint(_right_polarising_actions)
		_right_polarising_actions[0] = 0
		_right_polarising_actions[n_user_states - 1] = 1
		self._right_polarising_recommendations = _right_polarising_actions

		# we can think of a reward function as to whether the user look at the content until the end
		_right_reward = np.random.rand(n_user_states)
		_right_reward = np.rint(_right_reward)
		self._right_reward = _right_reward
		_left_reward = np.random.rand(n_user_states)
		_left_reward = np.rint(_left_reward)
		self._left_reward = _left_reward

		assert 2 <= n_user_states
		# label the states as safe or unsafe
		# it should be unsafe in the edges and safe
		self._safe_right_threshold = np.random.randint(1, n_user_states)
		self._safe_left_threshold = np.random.randint(0, self._safe_right_threshold)
		
		# construct the moderators as random functions
		_moderator_probs = np.random.rand(n_user_states)
		_nonsilent_right_threshold = np.random.randint(self._safe_right_threshold, n_user_states)
		_nonsilent_left_threshold = np.random.randint(0, self._safe_left_threshold + 1)
		for user_state in range(0, _nonsilent_left_threshold):
			_moderator_probs[user_state] = 0
		for user_state in range(_nonsilent_right_threshold, n_user_states):
			_moderator_probs[user_state] = 0
		self._moderator_probs = _moderator_probs
		
		assert render_mode is None
		
	def _get_obs(self):
		return self._polarisation 

	def _get_info(self):
		return {"true side effect": xxx } # fill out	

	def reset(self, seed=None, options=None):
		
		self._polarisation = self.np_random.integers(self._safe_left_threshold, self._safe_right_threshold, size=self.n_users, dtype=int) #self.np_random.integers(0, self.n_user_states - 1, size=self.n_users, dtype=int)
		# I ensure that initial state is safe
		# check that the interval makes sense here, or whether an 1-off error

		
		observation = self._get_obs()
		info = self._get_info()
		
		return observation, info
		
	def step(self, action):
		
		# updating the state, cell by cell
		for user in range(0,self.n_users):
			if self._right_polarising_actions[self._polarisation[user]] == 1:
				if action[user] == 0:
					self._polarisation[user] = self._polarisation[user] + 1
				else:
					self._polarisation[user] = self._polarisation[user] - 1
			else:
				if action[user] == 0:
					self._polarisation[user] = self._polarisation[user] - 1 # sign flipped
				else:
					self._polarisation[user] = self._polarisation[user] + 1 # sign flipped
		
		# calculate rewards
		_reward = 0
		for user in range(0,self.n_users):
			if action[user] == 0:
				_reward += self._right_reward[self._polarisation[user]]
			else:
				_reward += self._left_reward[self._polarisation[user]]
		_reward = _reward / (2. * self.n_users)

		# getting side effects (too extreme content) info from moderators
		

		observation = self._get_obs()
		reward = _reward
		terminated = False
		truncated = False # I need to change this to some kind of time limit
		info = self._get_info()
		
		return observation, reward, side_effects, terminated, truncated, info
