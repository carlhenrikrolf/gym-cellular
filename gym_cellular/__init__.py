from gymnasium.envs.registration import register


register(
    id="gym_cellular/Cells3States3Actions3-v0",
    entry_point="gym_cellular.envs:Cells3States3Actions3Env",
    max_episode_steps=None,
)

register(
    id="gym_cellular/Cells2Rest3-v0",
    entry_point="gym_cellular.envs:Cells2Rest3Env",
    max_episode_steps=None,
)

register(
    id="gym_cellular/Debug-v0",
    entry_point="gym_cellular.envs:DebugEnv",
    max_episode_steps=None,
)


register(
    id="gym_cellular/DeepPlanningDebug-v0",
    entry_point="gym_cellular.envs:DeepPlanningDebugEnv",
    max_episode_steps=None,
)

register(
    id="gym_cellular/DeepExplorationDebug-v0",
    entry_point="gym_cellular.envs:DeepExplorationDebugEnv",
    max_episode_steps=None,
)