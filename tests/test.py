### comments ###

################

print("START TEST\n")

import gymnasium as gym
import gym_cellular
import numpy as np

# settings
n_time_steps = 19
env = gym.make('gym_cellular/Polarisation-v0',
	n_users = 4,
	n_user_states=8,
	init_seed=7
)

# run
observation, info = env.reset()
for _ in range(0,n_time_steps):
	previous_observation = np.copy(observation)
	action = env.action_space.sample()
	observation, reward, terminated, truncated, info = env.step(action)
	print("previous state:", previous_observation)
	print("action:        ", action)
	print("current state: ", observation)
	print("reward:        ", reward)
	print("side effects:\n", info["side_effects"], "\n")

print("TEST DONE")
