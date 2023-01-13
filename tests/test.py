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
assert version == 'gym_cellular/Polarisation-v1'
_n_user_states = 4
_n_recommendations = 3
n_time_steps = 20
env = gym.make(version,
	n_users = 2,
	n_user_states=_n_user_states,
	n_moderators=1,
	init_seed=None,
	n_recommendations=_n_recommendations
)


observation, info = env.reset()

print("PARAMETERS")
parameters = info["environment_parameters"]
print("number of time steps:", n_time_steps)
print("degrees of polarisation", _n_user_states)
print("number of recommendations:", _n_recommendations)
print("right left split:", parameters["right_left_split"])
print("\n")

print("START RUN")
print("initial state:", observation)
print("initial side effects:\n", info["side_effects"])
print( "\n")

#action = env.action_space.sample()

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
	print("side effects incidence:", info["side_effects_incidence"])
	print("\n")

print("%%%%%%%%%%%%%%")
print("% TEST DONE  %")
print("%%%%%%%%%%%%%%")
print("\n")
