import math
import numpy as np
from copy import deepcopy
from gym_cellular.envs.utils.space_transformations import cellular2tabular, tabular2cellular

import gymnasium as gym
from gymnasium import spaces

class PolarisationV2Env(gym.Env):
	
	"""
	This is version 1 (as opposed to 0).
	There are two main updates:
	(1) The state space is expanded with a binary which shows
	whether is two-way polarisable or one-way polariseable
	(2) The action space is expanded
	"""
	# probably, I should think of the action being the same video in different user states
	
	def __init__(self, render_mode=None, n_users=2, n_user_states=8, n_moderators=2, seed=1, n_recommendations = 2):
		
		# check inputs to the environment
		assert type(n_users) is int and n_users >= 1
		assert type(n_user_states) is int and n_user_states >= 2
		assert type(n_moderators) is int and 0 <= n_moderators <= n_users
		assert render_mode is None # No rendering for now
		assert type(n_recommendations) is int and n_recommendations >= 2

		# move inputs to self
		self.n_users = n_users
		self.n_user_states = n_user_states
		self.n_moderators = n_moderators
		self.seed = seed
		self.n_recommendations = n_recommendations

		# variables for AlwaysSafe
		self.n_states = (self.n_user_states * 2) ** self.n_users
		self.n_actions = n_recommendations ** n_users 
		
		# set additional attributes
		self.observation_space = spaces.Dict(
			{
				"polarisation": spaces.Box(0, n_user_states - 1, shape=(n_users,), dtype=int),
				"two_way_polarisable": spaces.MultiDiscrete(np.zeros(n_users, dtype=int) + 2),
			}
		)
		self.action_space = spaces.MultiDiscrete(np.zeros((n_users), dtype=int) + n_recommendations) # arbitrary number of intracellular actions
		self.reward_range = (0, 1)
		
		# seed random generator for how the environment is initialised
		np.random.seed(seed=self.seed)

		# for each action, decide a level of polarisation towards which it is attracted
		self._attractor_state = np.random.randint(0, n_user_states, size=n_recommendations, dtype=int)
		
		# we can think of a reward function as to whether the user look at the content until the end
		# the reward gets higher for the edge cases
		_parabola = lambda x: (2. / n_user_states)**2 * (x - n_user_states / 2.)**2
		_intracellular_reward_function = np.random.rand(n_user_states, n_recommendations)
		for user_state in range(0, n_user_states):
			_intracellular_reward_function[user_state, :] = _intracellular_reward_function[user_state, :] + 2 * _parabola(user_state)
		_intracellular_reward_function = _intracellular_reward_function / 3
		self._intracellular_reward_function = _intracellular_reward_function
		
		# label the states as safe or unsafe
		# it should be unsafe in the edges and safe
		self._safe_right_threshold = np.random.randint(1, n_user_states)
		self._safe_left_threshold = np.random.randint(0, self._safe_right_threshold)


		# somewhere, decide a partition where unipolarisable users cannot cross
		self._right_left_split = np.random.randint(self._safe_left_threshold - 1, self._safe_right_threshold + 1)
		
		# moderators
		# initialise the moderator probability function
		_moderator_probs = np.random.rand(n_users, n_users, n_user_states, n_user_states)
		# picking the set of moderators from the set of users
		set_of_moderators = np.random.choice(n_users, n_moderators, replace=False)
		# making the complement such that they are silent
		set_of_nonmoderators = np.setdiff1d(np.array(range(0, n_users)), set_of_moderators)
		for nonmoderator in set_of_nonmoderators:
			for user in range(0, n_users):
				for nonmoderator_state in range(0, n_user_states):
					for user_state in range(0, n_user_states):
						_moderator_probs[nonmoderator, user, nonmoderator_state, user_state] = 0
		# moderators are not uncertain about their own states
		for moderator in set_of_moderators:
			for moderator_state in range(0, n_user_states):
				_moderator_probs[moderator, moderator, moderator_state, moderator_state] = 1
		# strongly polarised moderators start keeping silent about mistakes from "their side"
		for moderator in set_of_moderators:
			_nonsilent_right_threshold = np.random.randint(self._safe_right_threshold, n_user_states)
			_nonsilent_left_threshold = np.random.randint(0, self._safe_left_threshold + 1)
			for user in range(0, n_users):
				for moderator_state in range(0, _nonsilent_left_threshold):
					for user_state in range(self._safe_right_threshold + 1, n_user_states):
						_moderator_probs[moderator, user, moderator_state, user_state] = 0
				for moderator_state in range(_nonsilent_right_threshold + 1, n_user_states):
					for user_state in range(0, self._safe_left_threshold):
						_moderator_probs[moderator, user, moderator_state, user_state] = 0
		# future work: some moderator may be inherently biased for one particular political view
		# and therfore stay silent towards critique of the othe side
		# also including such moderators may be more convincing for some people
		self._moderator_probs = _moderator_probs

	def get_initial_policy(self):
		return np.zeros((self.n_users, self.n_states), dtype=int)
		
	def _get_obs(self):
		return {"polarisation": self._polarisation, "two_way_polarisable": self._two_way_polarisable}
	
	def _get_side_effects(self):

		_side_effects = np.zeros((self.n_users, self.n_users), dtype='<U6')
		_side_effects_incidence = 0
		for user in range(0, self.n_users):
			if self._safe_left_threshold <= self._polarisation[user] <= self._safe_right_threshold:
				for moderator in range(0, self.n_users):
					if self._moderator_probs[moderator, user, self._polarisation[moderator], self._polarisation[user]] > np.random.rand(1): # by de>
						_side_effects[moderator, user] = "safe"
					else:
						_side_effects[moderator, user] = "silent"
			else:
				_side_effects_incidence = _side_effects_incidence + 1 # the true safety violation
				for moderator in range(0, self.n_users):
					if self._moderator_probs[moderator, user, self._polarisation[moderator], self._polarisation[user]] > np.random.rand(1):
						_side_effects[moderator, user] = "unsafe"
					else:
						_side_effects[moderator, user] = "silent"
		_side_effects_incidence = _side_effects_incidence / self.n_users
		return _side_effects, _side_effects_incidence
		
	def _get_info(self):
		_side_effects, _ = self._get_side_effects()
		return {"side_effects": _side_effects}
		
	def reset(self, seed=None, options=None):
		
		# ensure that the initial state is safe
		self._polarisation = self.np_random.integers(self._safe_left_threshold, self._safe_right_threshold + 1, size=self.n_users, dtype=int) # if it works like np.random.randint
		self._two_way_polarisable = self.np_random.integers(0, 2, size=self.n_users, dtype=int)
		
		observation = self._get_obs()
		info = self._get_info()
		
		return observation, info
		
	# future work: generate safe policy somehow
		
	def step(self, action):

		self._reward = self.reward_function(self._get_obs(), action)

		# updating the state, cell by cell, updated
		for user in range(0,self.n_users):
			if 0 < self._polarisation[user] < self.n_user_states - 1:
				if self._two_way_polarisable[user] == 0 and self._polarisation[user] == self._right_left_split:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user]
					else:
						self._polarisation[user] = self._polarisation[user] - 1
				elif self._two_way_polarisable[user] == 0 and self._polarisation[user] == self._right_left_split + 1:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user] + 1
					else:
						self._polarisation[user] = self._polarisation[user]
				else:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user] + 1
					else:
						self._polarisation[user] = self._polarisation[user] - 1
			# ensuring that we don't fall off the state space at the edges
			elif self._polarisation[user] == 0:
				if self._two_way_polarisable[user] == 0 and self._polarisation[user] == self._right_left_split:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user]
					else:
						self._polarisation[user] = self._polarisation[user]
				elif self._two_way_polarisable[user] == 0 and self._polarisation[user] == self._right_left_split + 1:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user] + 1
					else:
						self._polarisation[user] = self._polarisation[user]
				else:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user] + 1
					else:
						self._polarisation[user] = self._polarisation[user]
			else:
				if self._two_way_polarisable[user] == 0 and self._polarisation[user] == self._right_left_split:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user]
					else:
						self._polarisation[user] = self._polarisation[user] - 1
				elif self._two_way_polarisable[user] == 0 and self._polarisation[user] == self._right_left_split + 1:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user]
					else:
						self._polarisation[user] = self._polarisation[user]
				else:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user]
					else:
						self._polarisation[user] = self._polarisation[user] - 1

		observation = self._get_obs()
		reward = self._reward
		# side_effects = _side_effects
		terminated = False
		truncated = False # I need to change this to some kind of time limit
		info = self._get_info()
		
		return observation, reward, terminated, truncated, info


	def cellular_encoding(self, observation):

		"""Turns an obseravtion from the environment into a cellular state representation."""

		polarisation = observation['polarisation']
		two_way_polarisable = observation['two_way_polarisable']
		alist = np.zeros(self.n_users, dtype=int)
		alist = polarisation + two_way_polarisable * self.n_user_states
		return alist


	def cellular_decoding(self, action):

		"""Turns a cellular state representation into an action that can be used as input to the environment."""
		
		# trivial in this case
		return action

	
	def tabular_encoding(self, observation):

		"""Turns an observation from the environment into a tabular state representation."""
		
		alist = self.cellular_encoding(observation)
		anint = cellular2tabular(alist, self.n_user_states * 2, self.n_users)
		return anint

	
	def _inverse_tabular_encoding(self, tabular_observation):

		"""The inverse of tabular_encoding. Turns a tabular state representation into an observation that can be used as input to the environment."""

		assert type(tabular_observation) is int
		assert 0 <= tabular_observation < self.n_states

		alist = tabular2cellular(tabular_observation, self.n_user_states * 2, self.n_users)
		polarisation = alist % self.n_user_states 
		two_way_polarisable = alist // self.n_user_states 
		observation = {"polarisation": polarisation, "two_way_polarisable": two_way_polarisable}
		return observation


	def tabular_decoding(self, tabular_action):

		"""
		Turns a tabular action representation into an action that can be used as input to the environment.
		"""
		
		assert type(tabular_action) is int
		assert 0 <= tabular_action < self.n_actions

		alist = tabular2cellular(tabular_action, self.n_recommendations, self.n_users)
		alist = self.cellular_decoding(alist)
		return alist

	
	def _inverse_tabular_decoding(self, action):

		assert 0 <= action < self.n_actions
		tabular_action = cellular2tabular(action, self.n_recommendations, self.n_users)
		return tabular_action

	
	def reward_function(self, observation, action):
		polarisation = deepcopy(observation['polarisation'])
		reward = 0
		for user in range(self.n_users):
			reward += self._intracellular_reward_function[polarisation[user], action[user]]
		reward = reward / (self.n_users * self.n_recommendations)
		if reward <= 0.05:
			reward = 0
		return reward


	def tabular_reward_function(self, tabular_observation, tabular_action):
		
		observation = self._inverse_tabular_encoding(tabular_observation)
		action = self.tabular_decoding(tabular_action)
		reward = self.reward_function(observation, action)
		return reward

	
	def get_side_effects_incidence(self):
		_, side_effects_incidence = self._get_side_effects()
		return side_effects_incidence



