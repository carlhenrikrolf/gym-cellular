import numpy as np

import gymnasium as gym
from gymnasium import spaces

class PolarisationEnv(gym.Env):
	
	"""
	My own environment. Thanks to Luke Thorburn for idea
	"""
	
	def __init__(self, render_mode=None, n_users=2, n_user_states=8, n_moderators=2, init_seed=1):
		
		# check inputs to the environment
		# maybe I could move these to spec?
		assert type(n_users) is int and n_users >= 1
		assert type(n_user_states) is int and n_user_states >= 2
		assert type(n_moderators) is int and n_moderators >= 0
		assert n_moderators <= n_users
		assert render_mode is None # No rendering for now

		# move inputs to self
		self.n_users = n_users
		self.n_user_states = n_user_states
		self.n_moderators = n_moderators
		self.init_seed = init_seed
		
		# set additional attributes
		self.observation_space = spaces.Box(0, n_user_states - 1, shape=(n_users,), dtype=int)
		self.action_space = spaces.MultiDiscrete(np.zeros((n_users), dtype=int) + 2) # 2 intracellular actions
		self.reward_range = (0, 1)
		
		# seed random generator for how the environment is initialised
		np.random.seed(seed=init_seed)
		
		# recommendations given a state that polarise in a right directions indicated by 1
		# if not right polarising, then left polarising
		# future work: we might want to be able to change the number of actions as well
		_right_polarising_recommendations = np.random.rand(n_user_states)
		_right_polarising_recommendations = np.rint(_right_polarising_recommendations)
		self._right_polarising_recommendations = _right_polarising_actions
		
		# we can think of a reward function as to whether the user look at the content until the end
		_right_reward = np.random.rand(n_user_states) # 2 intracellular actions
		self._right_reward = _right_reward
		_left_reward = np.random.rand(n_user_states)
		self._left_reward = _left_reward
		
		# label the states as safe or unsafe
		# it should be unsafe in the edges and safe
		self._safe_right_threshold = np.random.randint(1, n_user_states)
		self._safe_left_threshold = np.random.randint(0, self._safe_right_threshold)
		
		# moderators
		# initialise the moderator probability function
		_moderator_probs = np.random.rand(n_users - 1, n_users - 1, n_user_states - 1, n_user_states - 1)
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
					for user_state in range(_safe_right_threshold + 1, n_user_states):
						_moderator_probs[moderator, user, moderator_state, user_state] = 0
				for moderator_state in range(_nonsilent_right_threshold + 1, n_user_states):
					for user_state in range(0, _safe_left_threshold):
						_moderator_probs[moderator, user, moderator_state, user_state] = 0
		# future work: some moderator may be inherently biased for one particular political view
		# and therfore stay silent towards critique of the othe side
		# also including such moderators may be more convincing for some people
		
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
		
		# calculate rewards
                _reward = 0
                for user in range(0,self.n_users):
                        if action[user] == 0:
                                _reward += self._right_reward[self._polarisation[user]]
                        else:
                                _reward += self._left_reward[self._polarisation[user]]
                _reward = _reward / (2. * self.n_users)
		
		# updating the state, cell by cell
		for user in range(0,self.n_users):
			if 0 < self._polarisation[user] < self.n_user_states - 1:
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
			# ensuring that we don't fall off the state space at the edges
			elif self._polarisation[user] == 0:
				if self._right_polarising_actions[0] == 1:
					if action[user] == 0:
						self._polarisation[user] == 1
					else:
						self._polarisation[user] == 0
				else:
					if action[user] == 0:
                                                self._polarisation[user] == 0
                                        else:
                                                self._polarisation[user] == 1
			else:
				if self._right_polarising_actions[self.n_user_states - 1] == 1:
					if action[user] == 0:
						self._polarisation[user] == self.n_user_states - 1
					else:
						self._polarisation[user] == self.n_user_states - 2
				else:
					if action[user] == 0:
                                                self._polarisation[user] == self.n_user_states - 2
                                        else:
                                                self._polarisation[user] == self.n_user_states - 1

		
		
		# getting side effects (too extreme content) info from moderators
		_side_effects = np.zeros((self.n_users, self.n_users), dtype='<U6')
		for moderator in range(0, self.n_users):
			for user in range(0, self.n_users):
				if self._safe_left_threshold <= self._polarisation[user] <= self._safe_right_threshold:
					if self._moderator_probs[moderator, user, self._polarisation[moderator], self._polarisation[user]] > np.random.rand((1)):
						_side_effects[moderator, user] = "safe"
					else:
						_side_effects[moderator, user] = "silent"
				else:
					if self._moderator_probs[moderator, user, self._polarisation[moderator], self._polarisation[user]] > np.random.rand((1)):
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
