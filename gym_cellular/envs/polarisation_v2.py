import math
import numpy as np
from copy import deepcopy

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

	def _new_bin_enc(self, observation):

		polarisation = observation['polarisation']
		two_way_polarisable = np.reshape(observation['two_way_polarisable'], (self.n_users, 1))
		#intracellular_state_bit_length = polarisation_bit_length + 1
		polarisation_bit_length = math.ceil(np.log2(self.n_user_states))
		bin_polarisation_array = np.zeros((self.n_users, polarisation_bit_length), dtype=int)
		for user in range(self.n_users):
			bin_str = bin(polarisation[user])[2:]
			start_bit = polarisation_bit_length - len(bin_str)
			for bit in range(start_bit, polarisation_bit_length):
				bin_polarisation_array[user, bit] = bin_str[bit - start_bit]
		return np.concatenate((bin_polarisation_array, two_way_polarisable), axis=1)


	def _binary_encoding(self, observation):

		# unpack observation
		polarisation = observation['polarisation']
		two_way_polarisable = observation['two_way_polarisable']

		# form binary array
		user_state_bit_length = math.ceil(np.log2(self.n_user_states))
		bin_array = np.zeros((user_state_bit_length + 1, self.n_users), dtype=int)
		for user in range(self.n_users):
			bin_str = bin(polarisation[user])[2:]
			start_bit = user_state_bit_length - len(bin_str)
			for bit in range(start_bit, user_state_bit_length):
				bin_array[bit, user] = bin_str[bit - start_bit]
			bin_array[user_state_bit_length, user] = two_way_polarisable[user]
		
		return bin_array


	def cellular_encoding(self, observation):

		bin_array = self._new_bin_enc(observation)
		int_list = np.zeros(self.n_users, dtype=int)
		for user in range(self.n_users):
			int_list[user] = int(np.dot(np.flip(bin_array[user,:]), 2 ** np.arange(bin_array[user,:].size)))
		return int_list


	def cellular_decoding(self, action):
		
		return action

	
	def tabular_encoding(self, observation):
		
		bin_array = self._new_bin_enc(observation)
		bin_list = np.reshape(bin_array, np.prod(np.shape(bin_array)))
		integer = int(np.dot(np.flip(bin_list), 2 ** np.arange(bin_list.size)))
		return integer

	
	def _inverse_tabular_encoding(self, tabular_observation):

		assert type(tabular_observation) is int
		assert 0 <= tabular_observation < self.n_states

		binary = bin(tabular_observation)[2:]
		bit_length = (self.n_user_states**self.n_users - 1).bit_length() + self.n_users
		bin_list = np.zeros(bit_length)
		start_bit = bit_length - len(binary)
		for bit in range(start_bit, bit_length):
			bin_list[bit] = int(binary[bit - start_bit])
		bin_array = np.reshape(bin_list, (self.n_users, int(bit_length / self.n_users)))

		two_way_polarisable = np.array(bin_array[:, -1], dtype=int)

		polarisation = np.zeros(self.n_users, dtype=int)
		for user in range(self.n_users):
			polarisation[user] = np.dot(np.flip(bin_array[user, :-1]), 2 ** np.arange(bin_array[user, :-1].size))

		observation = {"polarisation": polarisation, "two_way_polarisable": two_way_polarisable}

		return observation


	def tabular_decoding(self, action):

		"""
		Turns an action in the form of an integer into a vector of integers.
		"""
		
		# form binary array
		recommendation_bit_length = self.n_recommendations - 1
		recommendation_bit_length = recommendation_bit_length.bit_length()
		end_bit = self.n_users * recommendation_bit_length
		bin_list = np.zeros(end_bit, dtype=int)
		bin_str = bin(action)[2:]
		start_bit = end_bit - len(bin_str)
		for bit in range(start_bit, end_bit):
			bin_list[bit] = bin_str[bit - start_bit]
		bin_array = bin_list.reshape(self.n_users, recommendation_bit_length)
		vector = np.zeros(self.n_users, dtype=int)

		# turn binary array into integer list (vector)
		for user in range(self.n_users):
			for bit in range(recommendation_bit_length):
				vector[user] = np.dot(np.flip(bin_array[user, bit]), 2 ** np.arange(recommendation_bit_length))

		return vector

	
	def _inverse_tabular_decoding(self, tabular_action):


		assert 0 <= tabular_action < self.n_actions
		
		binary = bin(tabular_action)[2:]
		bit_length = (self.n_recommendations**self.n_users - 1).bit_length()
		bin_list = np.zeros(bit_length)
		start_bit = bit_length - len(binary)
		for bit in range(start_bit, bit_length):
			bin_list[bit] = int(binary[bit - start_bit])
		bin_array = np.reshape(bin_list, (self.n_users, int(bit_length / self.n_users)))
		int_list = np.zeros(self.n_users, dtype=int)
		for user in range(self.n_users):
			int_list[user] = np.dot(np.flip(bin_array[user]), 2 ** np.arange(bin_array[user].size))
		return int_list

	
	def reward_function(self, observation, action):
		polarisation = deepcopy(observation['polarisation'])
		reward = 0
		for user in range(self.n_users):
			reward += self._intracellular_reward_function[polarisation[user], action[user]]
		reward = reward / (self.n_users * self.n_recommendations)
		if reward <= 0.2:
			reward = 0
		return reward


	def tabular_reward_function(self, tabular_observation, tabular_action):
		
		observation = self._inverse_tabular_encoding(tabular_observation)
		action = self._inverse_tabular_decoding(tabular_action)
		reward = self.reward_function(observation, action)
		return reward

	
	def get_side_effects_incidence(self):
		_, side_effects_incidence = self._get_side_effects()
		return side_effects_incidence



