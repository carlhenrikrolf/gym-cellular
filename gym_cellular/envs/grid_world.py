import copy as cp
import gymnasium as gym
import numpy as np

n_jurisdictions = 2
grid_shape = (2, 2)
tree_density = 2



class GridWorld(gym.Env):
    def __init__(self, **kwargs):
        self.prior_knowledge = PriorKnowledge(**kwargs)
        self._state_space = gym.spaces.Tuple(
            [
                gym.spaces.Dict(
                    {
                        'agt': gym.spaces.Dict(
                            {
                                'position': gym.spaces.Box(
                                    low=0,
                                    high=1,
                                    shape=grid_shape,
                                    dtype=int,
                                )
                            }
                        ),
                        'living_trees': gym.spaces.MultiBinary(tree_density)
                    }
                )
            ] * n_jurisdictions
        )
        self.state_space = cp.copy(self._state_space)
        self.state_space.sample = self._state_space_sample
        self.observation_space = self.state_space
        self._action_space = gym.spaces.Tuple(
            [
                gym.spaces.Dict(
                    {
                        'go_to': gym.spaces.Dict(
                            {
                                'position': gym.spaces.Box(
                                    low=0,
                                    high=1,
                                    shape=grid_shape,
                                    dtype=int,
                                )
                            }
                        )
                    }
                )
            ] * n_jurisdictions
        )
        self.action_space = cp.copy(self._action_space)
        self.action_space.sample = self._action_space_sample

        self.reward_range = (0, 1)

    def step(self, action):
        return self.state_space.sample(), 0, False, False, {}

    def reset(self):
        state = (
            {'agt': {'position': (0, 0)}, 'living_trees': (1, 1)},
            {'agt': {}, 'living_trees': (1, 0)},
        )
        return state, {}
    
    def _state_space_sample(self):
        state = self._state_space.sample()
        if np.random.rand() < 0.5:
            for jurisdiction in range(n_jurisdictions):
                state[jurisdiction]['agt']['position'] = {}
        return state
    
    def _action_space_sample(self):
        action = self.action_space.sample()
        if np.random.rand() < 0.5:
            for jurisdiction in range(n_jurisdictions):
                action[jurisdiction]['go_to']['position'] = {}
        return action
    
    


class PriorKnowledge():
    def __init__(self, **kwargs):
        self.n_cells = n_jurisdictions

    def available_actions(self, state):
        action_set = set()
        if state is ...:
            action = tuple([] * self.n_cells)
            for cell in range(self.n_cells):
                action[cell] = {'go_to': {'position': ...}}
            action_set.add(action)
        return action_set