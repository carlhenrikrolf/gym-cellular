from sqlite3 import enable_shared_cache
import numpy as np

import gymnasium as gym
from gymnasium import spaces

class PolarisationV1Env(gym.Env):
	
	"""
	This is version 1 (as opposed to 0).
	There are two main updates:
	(1) The state space is expanded with a binary which shows
	whether is two-way polarisable or one-way polariseable
	(2) The action space is expanded
	"""
	# probably, I should think of the action being the same video in different user states
	
	def __init__(self, render_mode=None, n_users=2, n_user_states=8, n_moderators=2, init_seed=1, n_recommendations = 2):
		
		# check inputs to the environment
		# maybe I could move these to spec?
		assert type(n_users) is int and n_users >= 1
		assert type(n_user_states) is int and n_user_states >= 2
		assert type(n_moderators) is int and n_moderators >= 0
		assert n_moderators <= n_users
		assert render_mode is None # No rendering for now
		assert type(n_recommendations) is int and n_recommendations >= 2

		# move inputs to self
		self.n_users = n_users
		self.n_user_states = n_user_states
		self.n_moderators = n_moderators
		self.init_seed = init_seed
		self.n_recommendations = n_recommendations
		
		# set additional attributes
		self.observation_space = spaces.Dict(
			{
				"polarisation": spaces.Box(0, n_user_states - 1, shape=(n_users,), dtype=int),
				"two_way_polarisable": spaces.Discrete(2),
			}
		)
		#self.observation_space = spaces.Dict(
		#	{
		#		"state": spaces.Box(0, n_user_states - 1, shape=(n_users,), dtype=int),
		#		"reward": spaces.Box(0, 1, shape(1,), dtype=np.float32),
		#		"side_effects": spaces.Box( ... ),
		#	}
		#)
		self.action_space = spaces.MultiDiscrete(np.zeros((n_users), dtype=int) + n_recommendations) # arbitrary number of intracellular actions
		self.reward_range = (0, 1)
		
		# seed random generator for how the environment is initialised
		np.random.seed(seed=init_seed)
		
		# recommendations given a state that polarise in a right directions indicated by 1
		# if not right polarising, then left polarising
		# future work: we might want to be able to change the number of actions as well
		#_right_polarising_actions = np.random.rand(n_user_states)
		#_right_polarising_actions = np.rint(_right_polarising_actions)
		#self._right_polarising_actions = _right_polarising_actions
		#_right_polarising_recommendations = np.random.rand(n_recommendations)
		#_right_polarising_recommendations = np.random.rint(_right_polarising_recommendations)
		#self._right_polarising_recommendations = _right_polarising_recommendations # Maybe I want to choose the attractor state for various degrees of polarisation.

		# for each action, decide a level of polarisation towards which it is attracted
		self._attractor_state = np.random.randint(0, n_user_states, size=n_recommendations, dtype=int)
		
		# we can think of a reward function as to whether the user look at the content until the end
		#_right_reward = np.random.rand(n_user_states) # 2 intracellular actions
		#self._right_reward = _right_reward
		#_left_reward = np.random.rand(n_user_states)
		#self._left_reward = _left_reward
		_parabola = lambda x: (2. / n_user_states)**2 * (x - n_user_states / 2.)**2 
		_intracellular_reward_function = np.random.rand(n_user_states, n_recommendations)
		for user_state in n_user_states:
			_intracellular_reward_function[user_state, :] = _intracellular_reward_function[user_state, :] + 2 * _parabola(user_state)
		_intracellular_reward_function = _intracellular_reward_function / 3
		self._intracellular_reward_function = _intracellular_reward_function
		
		# label the states as safe or unsafe
		# it should be unsafe in the edges and safe
		self._safe_right_threshold = np.random.randint(1, n_user_states)
		self._safe_left_threshold = np.random.randint(0, self._safe_right_threshold)


		# somewhere, decide a partition where unipolarisable users cannot cross
		self._right_left_split = np.random.rand(0, n_user_states)
		
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
		return _side_effects, _side_effects_incidence
		
	def _get_info(self):
		_side_effects, _side_effects_incidence = self._get_side_effects()
		return {"side_effects": _side_effects, "side_effects_incidence": _side_effects_incidence}
		
	def reset(self, seed=None, options=None):
		
		# ensure that the initial state is safe
		self._polarisation = self.np_random.integers(self._safe_left_threshold, self._safe_right_threshold + 1, size=self.n_users, dtype=int) # if it works like np.random.randint
		self._two_way_polarisable = np.random.randint(0, 1, size=self.n_users)
		
		observation = self._get_obs()
		info = self._get_info()
		
		return observation, info
		
	# future work: generate safe policy somehow
		
	def step(self, action):
		
		# calculate rewards
		_reward = 0
		for user in range(0, self.n_users):
			for recommendation in range(0, self.n_recommendations):
				_reward = _reward + self._intracellular_reward[user, recommendation]
		_reward = _reward / (self.n_users * self.n_recommendations)
		if _reward <= 0.2:
			_reward = 0
		self._reward = _reward

		# updating the state, cell by cell, updated
		for user in range(0,self.n_users):
			if 0 < self._polarisation[user] < self.n_user_states - 1:
				if self._two_way_polarisable[user] == 0 and self._polarisation[user] == self._right_left_split[user]:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user]
					else:
						self._polarisation[user] = self._polarisation[user] - 1
				elif self._two_way_polarisable[user] == 0 and self._polarisation[user] == self._right_left_split[user] + 1:
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
				if self._two_way_polarisable[user] == 0 and self._polarisation[user] == self._right_left_split[user]:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user]
					else:
						self._polarisation[user] = self._polarisation[user]
				elif self._two_way_polarisable[user] == 0 and self._polarisation[user] == self._right_left_split[user] + 1:
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
				if self._two_way_polarisable[user] == 0 and self._polarisation[user] == self._right_left_split[user]:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user]
					else:
						self._polarisation[user] = self._polarisation[user] - 1
				elif self._two_way_polarisable[user] == 0 and self._polarisation[user] == self._right_left_split[user] + 1:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user]
					else:
						self._polarisation[user] = self._polarisation[user]
				else:
					if self._polarisation[user] < self._attractor_state[action[user]]:
						self._polarisation[user] = self._polarisation[user]
					else:
						self._polarisation[user] = self._polarisation[user] - 1				
			
		
		
		# # updating the state, cell by cell
		# for user in range(0,self.n_users):
		# 	if 0 < self._polarisation[user] < self.n_user_states - 1:
		# 		if self._right_polarising_actions[self._polarisation[user]] == 1:
		# 			if action[user] == 0: # change to if the action is left of its set point, then go right, o.w. go left
		# 				self._polarisation[user] = self._polarisation[user] + 1
		# 			else:
		# 				self._polarisation[user] = self._polarisation[user] - 1
		# 		else:
		# 			if action[user] == 0:
		# 				self._polarisation[user] = self._polarisation[user] - 1 # sign flipped
		# 			else:
		# 				self._polarisation[user] = self._polarisation[user] + 1 # sign flipped
		# 	# ensuring that we don't fall off the state space at the edges
		# 	elif self._polarisation[user] == 0:
		# 		if self._right_polarising_actions[0] == 1:
		# 			if action[user] == 0:
		# 				self._polarisation[user] = 1
		# 			else:
		# 				self._polarisation[user] = 0
		# 		else:
		# 			if action[user] == 0:
		# 				self._polarisation[user] = 0
		# 			else:
		# 				self._polarisation[user] = 1
		# 	else:
		# 		if self._right_polarising_actions[self.n_user_states - 1] == 1:
		# 			if action[user] == 0:
		# 				self._polarisation[user] = self.n_user_states - 1
		# 			else:
		# 				self._polarisation[user] = self.n_user_states - 2
		# 		else:
		# 			if action[user] == 0:
		# 				self._polarisation[user] = self.n_user_states - 2
		# 			else:
		# 				self._polarisation[user] = self.n_user_states - 1
		# # I also need to implement a place where that partition is

		# getting side effects (too extreme content) reports from moderators
		# self._side_effects = self._get_side_effects()

		observation = self._get_obs()
		reward = _reward
		# side_effects = _side_effects
		terminated = False
		truncated = False # I need to change this to some kind of time limit
		info = self._get_info()
		
		return observation, reward, terminated, truncated, info
