### comments ###

# add more actions

# add users that can only be polarised one way or the other

################
print("\n")
print("%%%%%%%%%%%%%%")
print("% START TEST %")
print("%%%%%%%%%%%%%%")
print("\n")

import gymnasium as gym
import gym_cellular
import numpy as np

# settings
version = 'gym_cellular/Polarisation-v1'
n_time_steps = 5
env = gym.make(version,
	n_users = 4,
	n_user_states=8,
	n_moderators=1,
	init_seed=None
)

print("START RUN")
observation, info = env.reset()
print("initial state:", observation)
print("initial side effects:\n", info["side_effects"])
print( "\n")

for _ in range(0,n_time_steps - 1):
	previous_polarisation = np.copy(observation["polarisation"])
	previous_two_way_polarisable = np.copy(observation["two_way_polarisable"])
	previous_observation = {"polarisation": previous_polarisation, "two_way_polarisable": previous_two_way_polarisable}
	action = env.action_space.sample()
	observation, reward, terminated, truncated, info = env.step(action)
	print("previous state:", previous_observation["polarisation"])
	print("action:        ", action)
	print("current state: ", observation["polarisation"])
	print("reward:        ", reward)
	print("side effects:\n", info["side_effects"])
	if version == 'gym_cellular/Polarisation-v1':
		print("side effects incidence:", info["side_effects_incidence"])
	print("\n")

print("%%%%%%%%%%%%%%")
print("% TEST DONE  %")
print("%%%%%%%%%%%%%%")
print("\n")
