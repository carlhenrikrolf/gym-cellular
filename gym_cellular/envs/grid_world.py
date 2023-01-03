import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
	
	def __init__(self, render_mode=None, size=5):
		self.size + size
		self.window_size = 512
		
		self.observation_space = spaces.Dict(
			{
				"agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
				"target": spaces.Box(0, size - 1, shape(2,), dtype=int),
			}
		)
		
		self.action_space = spaces.Discrete(4)
		
		# right: 0, up: 1, left: 2, down: 3
		self._action_to_direction = {
			0: np.array([1,0]),
			1: np.array([0,1]),
			2: np.array([-1,0]),
			3: np.array([0,-1]),
		}
		
		assert render_mode is None or render_mode is in self.metadata["render_modes"]
		self.render_mode = render_mode
		
		self.window = None # reference to the window used for rendering?
		self.clock = None # clock for correct framerate in rendering
		
	def _get_obs(self):
		return {"agent": self._agent_location, "target": self._target_location}
		
	def _get_info(self):
		# Manhattan distance
		return {
			"distance": np.linalg.norm(
				self._agent_location - self._target_location, ord=1
			)
		}
		
	def reset(self, seed=None, options=None):
		super().reset(seed=seed) # super() is a way to inherit the class of the input (seed=seed) in the following function (reset())
		# this should seed np_random
		
		self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int) # output a pair with number between 0 and size of the grid
		
		 
