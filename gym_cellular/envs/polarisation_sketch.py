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
		# I could actually add more actions, where some of them have the same effect
		# a noop action makes a safe policy a bit too trivial to be illustrative
		
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
		
		### MODERATORS ###
		
		_moderator_probs = np.random.rand((n_users - 1, n_users - 1, n_user_states - 1, n_user_states - 1))
		# moderators are not uncertain about their own states
		for user in range(0, n_users):
			for user_state in range(0, n_user_states):
				_moderator_probs[user, user, user_state, user_state] = 1
		# strongly polarised moderators start keeping silent about mistakes from "their side"
		for moderator in range(0, n_users):
			_nonsilent_right_threshold = np.random.randint(self._safe_right_threshold, n_user_states)
                        _nonsilent_left_threshold = np.random.randint(0, self._safe_left_threshold + 1)
			for user in range(0, n_users):
				for moderator_state in range(0, _nonsilent_left_threshold):
					for user_state in range(_safe_right_threshold + 1, n_user_states):
						_moderator_probs[moderator, user, moderator_state, user_state] = 0
				for moderator_state in range(_nonsilent_right_threshold + 1, n_user_states):
					for user_state in range(0, _safe_left_threshold):
						_moderator_probs[moderator, user, moderator_state, user_state] = 0
		
		# construct the moderators as random functions
		_moderator_probs = np.random.rand((n_user_states - 1, n_users - 1, n_users - 1))
		#_moderator_probs = np.random.rand(n_user_states)
		# moderators are not uncertain about their own states
		for themself in range(0, n_users):
			for user_state in range(0, n_user_states):
				_moderator_probs[themself, themself, user_state] = 1
		
		for themself in range(0, n_users):
			_nonsilent_right_threshold = np.random.randint(self._safe_right_threshold, n_user_states)
			_nonsilent_left_threshold = np.random.randint(0, self._safe_left_threshold + 1)
			for user in range(0, n_users):
				for user_state in range(0, n_user_states):
					if not _nonsilent_left_threshold <= _user_state <= _nonsilent_right_threshold:
						_moderator_probs[themself, user, user_state] = 0
		# real question: can moderators see users that have become polarised.
		# yes? they can see comments made by certain users and see if that has become the result
		# some users may be of the type that comments a lot and some may be more silent
		# maybe we should have the stochasticity wrt to other users and determinism with respoect to self?
		# if content gets very polarised, then ..
		# when would moderators not work as moderators
		# one case (a) could be that for any content that aligns with their views, they would not report that kind of content
		# another case (b) is if content polarises the moderator themselves such that they will not report inappropriate content on their side
		# (if it's just a little bit, then the moderator is still relatively unbiased and can still report)
		# we're mostly dealing with (b) here. It is most relevant for what I want to model,
		# but maybe I could inlude two types of moderators: biased and biasable.
		# Could perhaps make it more realistic for the average person, but it's a more complex environment
		
		# We also want to make sure that not every user is necessarily a moderator, i.e., they are always silent
		
		### END MODERATORS ###
		
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
		
	# generate safe policy somehow
		
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
		_side_effects = np.zeros((self.n_users, self.n_users), dtype='<U6')
		for themself in range(0, self.n_users):
			for user in range(0, self.n_users):
				if self._safe_left_threshold <= self._polarisation[user] <= self._safe_right_threshold:
					if self._moderator_probs[themself, user, self._polarisation[user]] > np.random.rand((1)):
						_side_effects[themself, user] = "safe"
					else:
						_side_effects[themself, user] = "silent"
				else:
					if self._moderator_probs[themself, user, self._polarisation[user]] > np.random.rand((1)):
						_side_effects[themself, user] = "unsafe"
					else:
						_side_effects[themself, user] = "silent"

		observation = self._get_obs()
		reward = _reward
		side_effects = _side_effects
		terminated = False
		truncated = False # I need to change this to some kind of time limit
		info = self._get_info()
		
		return observation, reward, side_effects, terminated, truncated, info
