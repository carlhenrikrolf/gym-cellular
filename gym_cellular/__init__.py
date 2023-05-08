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

register(
	id="gym_cellular/Polarisation-v2",
	entry_point="gym_cellular.envs:PolarisationV2Env",
)

# register(
# 	id="gym_cellular/Polarisation-v3",
# 	entry_point="gym_cellular.envs:PolarisationV3Env",
# )

register(
    id="gym_cellular/Debug-v0",
    entry_point="gym_cellular.envs:DebugEnv",
    max_episode_steps=None,
)

register(
    id="gym_cellular/Cells3States3Actions3-v0",
    entry_point="gym_cellular.envs:Cells3States3Actions3Env",
    max_episode_steps=None,
)