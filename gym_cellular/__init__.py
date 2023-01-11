from gymnasium.envs.registration import register

register(
	id="gym_cellular/GridWorld-v0",
	entry_point="gym_cellular.envs:GridWorldEnv",
	max_episode_steps=300,
)
# id, directory, settings
# for time limit see wrapper documentation

register(
	id="gym_cellular/Polarisation-v0",
	entry_point="gym_cellular.envs:PolarisationEnv",
	max_episode_steps = 300,
)

register(
	id="gym_cellular/Polarisation-v1",
	entry_point="gym_cellular.envs:PolarisationV1Env",
	max_episode_steps=300,
)
