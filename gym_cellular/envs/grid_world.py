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
		# I guess I don't necessarily want this box thing, what other spaces are there?
		# Maybe it affects rendering what I choose?
		
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
		
		self._target_location = self._agent_location
		while np.array_equal(self._target_location, self._agent_location):
			self._target_location = self.np_random.integers(
				0, self.size, size=2, dtype=int
			)
			
		observation = self._get_obs()
		info = self._get_info()
		
		if self.render_mode == "human":
			self._render_frame()
		
		return observation, info
		
	def step(self, action):
		direction = self._action_to_direction[action]
		
		# don't leave the grid
		# does the environment wrap around space or is it just stop at the edge?
		self._agent_location = np.clip(
			self._agent_location = direction, 0, self.size - 1
		)
		
		# I guess I don't really want a termination criterion?
		# terminated = false is ok?
		terminated = np.array_equal(self._agent_location, self._target_location)
		reward = 1 if terminated else 0 # can I have vectorail rewards to represent constraints?
		observation = self._get_obs()
		info = self._get_info()
		# I guess we can do the following rewrite
		truncated = False
		
		if self.render_mode = "human":
			self._render_frame()
		
		return observation, reward, terminated, truncated, info
		
	def render(self):
		if self.render_mode == "rgb_array":
			return self._render_frame()
			
	def _render_frame(self):
		if self.window is None and self.render_mode == "human":
			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode(
				(self.window_size, self.window_size)
			)
		if self.clock is None and self.render_mode == "human":
			self.clock = pygame.time.Clock()
			
		# make the background on which we paint
		canvas = pygame.Surface((self.window_size, self.window_size))
		canvas.fill((255, 255, 255))
		pix_square_size = (
			self.window_size / self.size
		) # pixel size of grid cells
		
		# draw target
		pygame.draw.rect(
			canvas,
			(255, 0, 0),
			pygame.Rect(
				pix_square_size * self._target_location,
				(pix_square_size, pix_square_size),
			),
		)
		
		# draw agent
		pygame.draw.circle(
			canvas,
			(0, 0, 255),
			(self._agent_location = 0.5) * pix_square_size,
			pix_square_size / 3,
		)
		
		# draw gridlines
		for x in range(self.size + 1):
			pygame.draw.line(
				canvas,
				0,
				(0, pix_square_size * x),
				(self.window_size, pix_square_size * x),
				width = 3,
			)
			pygame.draw.line(
				canvas,
				0,
				(pix_square_size * x, 0),
				(pix_square_size * x, self.window_size),
				width = 3,
			)

		if self.render_mode == "human":
			# copy from canvas to window
			self.window.blit(canvas, canvas.get_rect())
			pygame.event.pump()
			pygame.display.update()
			
			# proper framerate through delays
			self.clock.tick(self.metadata["render_fps"])
		else:
			return np.transpose(
				np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
			)

		# not necessary, but nice to close the rendering window
		def close(self):
			if self.window is not None:
				pygame.display.quit()
				pygame.quit()
