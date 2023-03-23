import math
import numpy as np
import random
from copy import deepcopy
from time import sleep
from gym_cellular.envs.utils.space_transformations import pruned_cellular2tabular as cellular2tabular, pruned_tabular2cellular as tabular2cellular

import gymnasium as gym
from gymnasium import spaces

class PolarisationV3Env(gym.Env):
	
	"""
	This is version 3.
	There are three main updates from version 2:
	(1) transient states may be pruned. Note that this is not a guarantee that all transient states are pruned---just as in real life ...
	(2) An option of irreversible radicalisation is added
	(3) The silence of the regulators is a parameter that can be set.
	"""
	# probably, I should think of the action being the same video in different user states
	
	def __init__(
		self,
	    render_mode=None,
		n_users=2,
		n_user_states=8,
		n_moderators=2,
		seed=1,
		n_recommendations=2,
		transient_state_pruning=False,
		irreversible_radicalisation=False,
		silence=0,
	):
		
		# check inputs to the environment
		assert type(n_users) is int and n_users >= 1
		assert type(n_user_states) is int and n_user_states >= 2
		assert type(n_moderators) is int and 0 <= n_moderators <= n_users
		assert render_mode is None # No rendering for now
		assert type(n_recommendations) is int and n_recommendations >= 2
		assert 0 <= silence <= 1

		# move inputs to self
		self.n_users = n_users
		self.n_user_states = n_user_states
		self.n_moderators = n_moderators
		self.seed = seed
		self.n_recommendations = n_recommendations
		self.transient_state_pruning = transient_state_pruning
		self.irreversible_radicalisation = irreversible_radicalisation
		self.silence = silence
		
		# seed random generator for how the environment is initialised
		# and how the regulator feedback will be sampled
		np.random.seed(seed=self.seed)
		random.seed(self.seed)

		self.initial_policy = np.zeros((self.n_users, (self.n_user_states * 2) ** self.n_users), dtype=int)
		self._initialise_environment()
		
		# prune the state space
		# doesnt have to be in __init__, could just be a function to be called, for later ...
		self.intracellular_state_space_set = self.get_intracellular_state_space_set()
		self.intracellular_action_space_set = self.get_intracellular_action_space_set()


	def _initialise_environment(self):

		# set additional attributes
		self.observation_space = spaces.Dict(
			{
				"polarisation": spaces.Box(0, self.n_user_states - 1, shape=(self.n_users,), dtype=int),
				"two_way_polarisable": spaces.MultiDiscrete(np.zeros(self.n_users, dtype=int) + 2),
			}
		)
		self.action_space = spaces.MultiDiscrete(np.zeros((self.n_users), dtype=int) + self.n_recommendations)
		self.reward_range = (0, 1)

		
		# for each action, decide a level of polarisation towards which it is attracted
		self._attractor_state = np.random.randint(0, self.n_user_states, size=self.n_recommendations, dtype=int)
		
		# we can think of a reward function as to whether the user look at the content until the end
		# the reward gets higher for the edge cases
		_parabola = lambda x: (2. / self.n_user_states)**2 * (x - self.n_user_states / 2.)**2
		_intracellular_reward_function = np.random.rand(self.n_user_states, self.n_recommendations)
		for user_state in range(0, self.n_user_states):
			_intracellular_reward_function[user_state, :] = _intracellular_reward_function[user_state, :] + 2 * _parabola(user_state)
		_intracellular_reward_function = _intracellular_reward_function / 3
		self._intracellular_reward_function = _intracellular_reward_function
		
		# label the states as safe or unsafe
		# it should be unsafe in the edges and safe
		self._safe_right_threshold = np.random.randint(1, self.n_user_states)
		self._safe_left_threshold = np.random.randint(0, self._safe_right_threshold)


		# somewhere, decide a partition where unipolarisable users cannot cross
		self._right_left_split = np.random.randint(self._safe_left_threshold - 1, self._safe_right_threshold + 1)
		
		# moderators
		# initialise the moderator probability function
		_moderator_probs = np.random.rand(self.n_users, self.n_users, self.n_user_states, self.n_user_states)
		# picking the set of moderators from the set of users
		set_of_moderators = np.random.choice(self.n_users, self.n_moderators, replace=False)
		self._set_of_moderators = set_of_moderators
		# making the complement such that they are silent
		set_of_nonmoderators = np.setdiff1d(np.array(range(0, self.n_users)), set_of_moderators)
		for nonmoderator in set_of_nonmoderators:
			for user in range(0, self.n_users):
				for nonmoderator_state in range(0, self.n_user_states):
					for user_state in range(0, self.n_user_states):
						_moderator_probs[nonmoderator, user, nonmoderator_state, user_state] = 0
		# moderators are not uncertain about their own states
		for moderator in set_of_moderators:
			for moderator_state in range(0, self.n_user_states):
				_moderator_probs[moderator, moderator, moderator_state, moderator_state] = 1
		# strongly polarised moderators start keeping silent about mistakes from "their side"
		for moderator in set_of_moderators:
			_nonsilent_right_threshold = np.random.randint(self._safe_right_threshold, self.n_user_states)
			_nonsilent_left_threshold = np.random.randint(0, self._safe_left_threshold + 1)
			for user in range(0, self.n_users):
				for moderator_state in range(0, _nonsilent_left_threshold):
					for user_state in range(self._safe_right_threshold + 1, self.n_user_states):
						_moderator_probs[moderator, user, moderator_state, user_state] = 0
				for moderator_state in range(_nonsilent_right_threshold + 1, self.n_user_states):
					for user_state in range(0, self._safe_left_threshold):
						_moderator_probs[moderator, user, moderator_state, user_state] = 0
		# future work: some moderator may be inherently biased for one particular political view
		# and therfore stay silent towards critique of the othe side
		# also including such moderators may be more convincing for some people
		self._moderator_probs = _moderator_probs

		# initialise ...
		# ... the environment
		self._initial_polarisation = np.random.randint(self._safe_left_threshold + 1, self._safe_right_threshold + 1, size=self.n_users, dtype=int) #self.np_random.integers(self._safe_left_threshold, self._safe_right_threshold + 1, size=self.n_users, dtype=int) # if it works like np.random.randint
		# the first + 1 is to make sure going left is safe
		self._attractor_state[0] = self._initial_polarisation[0] # to make sure 0 action is safe from initial state
		self._initial_two_way_polarisable = np.random.randint(0, 2, size=self.n_users, dtype=int)

		# cell classification
		self.cell_classes = ["children", "nonconsenting"]
		self.cell_labelling_function = deepcopy(self.cell_classes)
		for cell_class in range(len(self.cell_classes)):
			n = random.randint(0, self.n_users)
			self.cell_labelling_function[cell_class] = random.sample(range(self.n_users), n)


	def get_intracellular_state_space_set(self):

		intracellular_state_space_set = np.zeros(self.n_users, dtype=range)
		if self.transient_state_pruning:
			for user in range(self.n_users):
				if self._initial_two_way_polarisable[user] == 0:
					arange = range(0, self.n_user_states)
				elif self._initial_polarisation[user] <= self._right_left_split:
					arange = range(self.n_user_states, self.n_user_states + self._right_left_split + 1)
				else:
					arange = range(self.n_user_states + self._right_left_split + 1, self.n_user_states + self.n_user_states)
				intracellular_state_space_set[user] = arange
		else:
			for user in range(self.n_users):
				intracellular_state_space_set[user] = range(0, self.n_user_states * 2)
		return intracellular_state_space_set
	

	def get_intracellular_action_space_set(self):

		intracellular_action_space_set = np.zeros(self.n_users, dtype=range)
		for user in range(self.n_users):
			intracellular_action_space_set[user] = range(0, self.n_recommendations)
		return intracellular_action_space_set
	
	
	def get_cell_classes(self):
		return self.cell_classes
	
	
	def get_cell_labelling_function(self):
		return self.cell_labelling_function
	

	def get_initial_policy(self):
		return self.initial_policy 
	
	
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
		
		self._polarisation = self._initial_polarisation
		self._two_way_polarisable = self._initial_two_way_polarisable
			
		observation = self._get_obs()
		info = self._get_info()
		
		return observation, info
		
	def step(self, action):

		self._reward = self.reward_function(self._get_obs(), action)

		if self.irreversible_radicalisation:
			self._irreverible_radicalisation_transition(action)
		else:
			self._vanilla_transition(action)

		observation = self._get_obs()
		reward = self._reward
		# side_effects = _side_effects
		terminated = False
		truncated = False # I need to change this to some kind of time limit
		info = self._get_info()
		
		return observation, reward, terminated, truncated, info
	

	def _vanilla_transition(self, action):

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

	
	def _irreverible_radicalisation_transition(self, action):
		pass # to be filled out later


	def cellular_encoding(self, observation):

		"""Turns an observation from the environment into a cellular state representation."""

		polarisation = observation['polarisation']
		two_way_polarisable = observation['two_way_polarisable']
		alist = np.zeros(self.n_users, dtype=int)
		for user in range(self.n_users):
			alist[user] = polarisation[user] + two_way_polarisable[user] * self.n_user_states
		return alist


	def cellular_decoding(self, action):

		"""Turns a cellular state representation into an action that can be used as input to the environment."""
		
		anaction = deepcopy(action)
		return anaction

	
	def tabular_encoding(self, observation):

		"""Turns an observation from the environment into a tabular state representation."""
		
		alist = self.cellular_encoding(observation)
		anint = cellular2tabular(alist, self.intracellular_state_space_set)
		return anint

	
	def _inverse_tabular_encoding(self, tabular_observation):

		"""The inverse of tabular_encoding. Turns a tabular state representation into an observation that can be used as input to the environment."""

		assert type(tabular_observation) is int
		assert 0 <= tabular_observation < (self.n_user_states * 2) ** self.n_users

		alist = tabular2cellular(tabular_observation, self.intracellular_state_space_set)
		polarisation = alist % self.n_user_states 
		two_way_polarisable = alist // self.n_user_states 
		observation = {"polarisation": polarisation, "two_way_polarisable": two_way_polarisable}
		return observation


	def tabular_decoding(self, tabular_action):

		"""
		Turns a tabular action representation into an action that can be used as input to the environment.
		"""
		
		assert type(tabular_action) is int
		assert 0 <= tabular_action < self.n_recommendations ** self.n_users

		alist = tabular2cellular(tabular_action, self.intracellular_action_space_set)
		alist = self.cellular_decoding(alist)
		return alist

	
	def _inverse_tabular_decoding(self, action):

		assert 0 <= action < self.n_recommendations ** self.n_users
		tabular_action = cellular2tabular(action, self.intracellular_action_space_set)
		return tabular_action

	
	def reward_function(self, observation, action):
		polarisation = deepcopy(observation['polarisation'])
		reward = 0
		for user in range(self.n_users):
			reward += self._intracellular_reward_function[polarisation[user], action[user]]
		reward = reward / (self.n_users * self.n_recommendations)
		#if reward <= 0.05:
		#	reward = 0
		return reward


	def tabular_reward_function(self, tabular_observation, tabular_action):
		
		observation = self._inverse_tabular_encoding(tabular_observation)
		action = self.tabular_decoding(tabular_action)
		reward = self.reward_function(observation, action)
		return reward

	
	def get_side_effects_incidence(self):
		_, side_effects_incidence = self._get_side_effects()
		return side_effects_incidence


	def render(self):

		"""Renders the environment."""

		canvas = ''
		for user in range(self.n_users):
			for user_state in range(self.n_user_states):
				if self._two_way_polarisable[user] == 1:
					if self._safe_left_threshold <= user_state <= self._safe_right_threshold:
						if user_state == self._polarisation[user]:
							if user in self._set_of_moderators:
								canvas += '_M '
							else:
								canvas += '_U '
						else:
							canvas += '_  '
					else:
						if user_state == self._polarisation[user]:
							if user in self._set_of_moderators:
								canvas += 'xM '
							else:
								canvas += 'xU '
						else:
							canvas += 'x  '
				elif self._initial_polarisation[user] <= self._right_left_split:
					if self._safe_left_threshold <= user_state <= self._right_left_split:
						if user_state == self._polarisation[user]:
							if user in self._set_of_moderators:
								canvas += '_M '
							else:
								canvas += '_U '
						else:
							canvas += '_  '
					elif user_state <= self._right_left_split:
						if user_state == self._polarisation[user]:
							if user in self._set_of_moderators:
								canvas += 'xM '
							else:
								canvas += 'xU '
						else:
							canvas += 'x  '
					else:
						canvas += '|| '
				else:
					if self._right_left_split < user_state <= self._safe_right_threshold:
						if user_state == self._polarisation[user]:
							if user in self._set_of_moderators:
								canvas += '_M '
							else:
								canvas += '_U '
						else:
							canvas += '_  '
					elif self._right_left_split < user_state:
						if user_state == self._polarisation[user]:
							if user in self._set_of_moderators:
								canvas += 'xM '
							else:
								canvas += 'xU '
						else:
							canvas += 'x  '
					else:
						canvas += '|| '
			canvas += '\n\n'

		print(canvas)
		sleep(1) # move outside?



