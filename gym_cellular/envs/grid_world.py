# modules
import copy as cp
import gymnasium as gym
import numpy as np

# parameters
n_jurisdictions = 2
grid_shape = (2, 2)
tree_positions = np.array(
    [
        [1, 0],
        [1, 0],
    ],
)

# checks
assert n_jurisdictions >= 1
assert len(grid_shape) == 2
assert tree_positions.shape == grid_shape

# derived parameters
n_intracellular_states = (grid_shape[0] * grid_shape[1] + 1 * (n_jurisdictions >= 2)) * 2 ** np.sum(tree_positions)
n_states = grid_shape[0] * grid_shape[1] * n_jurisdictions * (2 ** np.sum(tree_positions)) ** n_jurisdictions # note that n_states < n_intracellular_states ** n_cells
n_intracellular_actions = 3
n_actions = 3

# reward functions
def reward_func(self, state, action, next_state):
    return 0.

# the enviroment class
class GridWorldEnv(gym.Env):
    def __init__(self, **kwargs):
        self.prior_knowledge = PriorKnowledge(**kwargs)
        self._state_space = gym.spaces.Tuple(
            [
                gym.spaces.Dict(
                    {
                        'agt': gym.spaces.Dict(
                            {
                                'position': gym.spaces.Box(
                                    low=np.zeros(len(grid_shape)),
                                    high=np.array(grid_shape) - 1,
                                    dtype=int,
                                )
                            }
                        ),
                        'living_trees': gym.spaces.MultiBinary(grid_shape),
                    }
                ) for _ in range(n_jurisdictions)
            ]
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
                                    low=np.zeros(len(grid_shape)),
                                    high=np.array(grid_shape) - 1,
                                    dtype=int,
                                )
                            }
                        )
                    }
                ) for _ in range(n_jurisdictions)
            ]
        )
        self.action_space = cp.copy(self._action_space)
        self.action_space.sample = self._action_space_sample

        self.reward_range = (0, 1)

        self.reward_func = reward_func

    def step(self, action):
        return self.state_space.sample(), 0, False, False, {}

    def reset(self):
        state = (
            {
                'agt': {
                    'position': (1, 1),
                },
                'living_trees': np.array(
                    [
                        [1, 0],
                        [1, 0],
                    ]
                )
            },
            {
                'agt': {},
                'living_trees': np.array(
                    [
                        [1, 0],
                        [0, 0],
                    ]
                )
            },
        )
        return state, {}
    

    def transition_func(self, state, action):
        # assert state in available actions
        next_state = cp.deepcopy([
            {
            'agt': {},
            'living_trees': cp.deepcopy(tree_positions),
            } for _ in range(n_jurisdictions)
        ])
        for jurisdiction in range(n_jurisdictions):
            # movement of RL agent
            if 'position' in state[jurisdiction]['agt'] and 'position' in action[jurisdiction]['go_to']:
                next_state[jurisdiction]['agt']['position'] = action[jurisdiction]['go_to']['position']
                go_to_jurisdiction = jurisdiction
            elif 'position' in state[jurisdiction]['agt']:
                for go_to_jurisdiction in range(n_jurisdictions):
                    if 'position' in action[go_to_jurisdiction]['go_to']:
                        next_state[go_to_jurisdiction]['agt']['position'] = action[go_to_jurisdiction]['go_to']['position']
                        break
            else:
                continue
            # growth of trees
            was_where = state[jurisdiction]['agt']['position']
            was_at_tree = (state[jurisdiction]['living_trees'][was_where[0], was_where[1]] == 0 and tree_positions[was_where[0], was_where[1]] == 1)
            is_where = next_state[go_to_jurisdiction]['agt']['position']
            is_at_tree = (state[go_to_jurisdiction]['living_trees'][is_where[0], is_where[1]] == 1)
            if was_at_tree and is_at_tree:
                next_state[jurisdiction]['living_trees'][was_where[0], was_where[1]] = 0
                next_state[go_to_jurisdiction]['living_trees'][is_where[0], is_where[1]] = 0
            elif is_at_tree:
                next_state[go_to_jurisdiction]['living_trees'][is_where[0], is_where[1]] = 0
            elif was_at_tree:
                next_state[jurisdiction]['living_trees'][was_where[0], was_where[1]] = 0
            break
        return cp.deepcopy(tuple(next_state))
    
    def side_effects_func(self, state):
        return np.array(
            [['silent'] * grid_shape[0]] * grid_shape[1]
        )
    
    def _state_space_sample(self):
        state = cp.deepcopy(self._state_space.sample())
        other_jurisdiction = np.random.randint(n_jurisdictions)
        state[other_jurisdiction]['agt'] = {}
        for jurisdiction in range(n_jurisdictions):
            state[jurisdiction]['living_trees'] *= tree_positions
        return state
    
    def _action_space_sample(self):
        action = cp.deepcopy(self._action_space.sample())
        other_jurisdiction = np.random.randint(n_jurisdictions)
        action[other_jurisdiction]['go_to'] = {}
        return action

# the prior knowledge class
class PriorKnowledge():
    def __init__(self, **kwargs):
        self.n_cells = n_jurisdictions
        self.reward_func = reward_func

    def available_actions(self, state):
        action_template = [{'go_to': {}} for _ in range(self.n_cells)] 
        for cell in range(self.n_cells):
            if 'position' in state[cell]['agt']:
                break
        other_cell = 0 if cell == 1 else 1
        if (state[cell]['agt']['position'] == np.array([0, 0])).all():
            right = cp.deepcopy(action_template)
            right[cell]['go_to']['position'] = np.array([0, 1])
            down = cp.deepcopy(action_template)
            down[cell]['go_to']['position'] = np.array([1, 0])
            return [tuple(right), tuple(down)]
        elif (state[cell]['agt']['position'] == np.array([1, 0])).all():
            up = cp.deepcopy(action_template)
            up[cell]['go_to']['position'] = np.array([0, 0])
            right = cp.deepcopy(action_template)
            right[cell]['go_to']['position'] = np.array([1, 1])
            return [tuple(up), tuple(right)]
        elif (state[cell]['agt']['position'] == np.array([0, 1])).all():
            left = cp.deepcopy(action_template)
            left[cell]['go_to']['position'] = np.array([0, 0])
            down = cp.deepcopy(action_template)
            down[cell]['go_to']['position'] = np.array([1, 1])
            out = cp.deepcopy(action_template)
            out[other_cell]['go_to']['position'] = np.array([1, 1])
            return [tuple(left), tuple(down), tuple(out)]
        elif (state[cell]['agt']['position'] == np.array([1, 1])).all():
            up = cp.deepcopy(action_template)
            up[cell]['go_to']['position'] = np.array([0, 1])
            left = cp.deepcopy(action_template)
            left[cell]['go_to']['position'] = np.array([1, 0])
            out = cp.deepcopy(action_template)
            out[other_cell]['go_to']['position'] = np.array([0, 1])
            return [tuple(up), tuple(left), tuple(out)]