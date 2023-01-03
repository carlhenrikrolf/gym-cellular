from gymnasium.envs.registration import register

register(
	id="gym_cellular/GridWorld-v0",
	entry_point="gym_cellular.envs:Gridworldenv",
	max_episode_steps=300,
)
# id, directory, settings
# for time limit see wrapper documentation

