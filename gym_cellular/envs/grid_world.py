# internal modules
from gym_cellular.envs.utils import generalized_cellular2tabular, generalized_tabular2cellular
# external modules
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
def reward_func(state, action, next_state):
    reward = 0.
    for jurisdiction in range(n_jurisdictions):
        reward += np.maximum(
            0,
            state[jurisdiction]['living_trees'] - next_state[jurisdiction]['living_trees']
        ).sum() #/ tree_positions.sum()
    #reward /= n_jurisdictions
    #reward = np.log2(1 + reward)    
    return reward

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

        if 'reward_func' in kwargs:
            self.reward_func = kwargs['reward_func']
        else:
            self.reward_func = reward_func

        self.data = {}


    def reset(self):
        self.state = cp.deepcopy(self.prior_knowledge.initial_state)
        self.reward = 0.0
        self.data['side_effects_incidence'] = 0.0
        self.side_effects = self.side_effects_func(self.state)
        info = self.get_info()
        self.data['time_step'] = 0
        return self.state, info
    

    def step(self, action):
        next_state = self.transition_func(self.state, action)
        self.reward = self.reward_func(self.state, action, next_state)
        self.side_effects = self.side_effects_func(next_state)
        self.state = next_state
        terminated = False
        truncated = False
        info = self.get_info()
        self.data['time_step'] += 1
        return self.state, self.reward, terminated, truncated, info
    

    def transition_func(self, state, action):
        # assert state in available actions
        # preliminaries
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
            # growth and death of trees
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
        # soil erosion
        n_barren_jurisdictions = 0
        for jurisdiction in range(n_jurisdictions):
            if state[jurisdiction]['living_trees'].sum() == 0:
                next_state[jurisdiction]['living_trees'] = np.zeros(shape=grid_shape, dtype=int)
                n_barren_jurisdictions += 1
        # seed dispersal
        if n_barren_jurisdictions < n_jurisdictions:
            if np.random.rand() < 0.01:
                next_state[np.random.randint(n_jurisdictions)]['living_trees'] = np.random.randint(2, size=grid_shape) * tree_positions
        # return
        self.data['side_effects_incidence'] = n_barren_jurisdictions / n_jurisdictions
        return cp.deepcopy(tuple(next_state))
    
    
    def side_effects_func(self, state):
        side_effects = np.array(
            [['silent'] * grid_shape[0]] * grid_shape[1]
        )
        if state[0]['living_trees'].sum() > 0:
            if state[1]['living_trees'].sum() == 0:
                side_effects[0,1] = 'unsafe'
                side_effects[0,1] = 'safe'
            else:
                side_effects[0,1] = 'safe'
                side_effects[0,0] = 'safe'
        return side_effects
    
    
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
    
    
    def get_info(self):
        info = {'side_effects': self.side_effects}
        return info
    

    def get_state(self):
        return self.state
    


# the prior knowledge class
class PriorKnowledge():

    def __init__(self, **kwargs):
        # defind quantities
        self.n_cells = n_jurisdictions
        self.n_intracellular_actions = grid_shape[0] * grid_shape[1] + 1
        self.action_space = [range(0, self.n_intracellular_actions) for _ in range(self.n_cells)]
        self.n_intracellular_states = (grid_shape[0] * grid_shape[1] + 1) * 2 ** tree_positions.sum()
        self.state_space = [range(0, self.n_intracellular_states) for _ in range(self.n_cells)]
        self.reward_func = reward_func
        self.cell_classes = ['regulator']
        self.cell_labelling = [[] for _ in range(self.n_cells)] # does not know where the regulator is
        if 'confidence_level' in kwargs:
            self.confidence_level = kwargs['confidence_level']
        else:
            self.confidence_level = 0.95
        if 'identical_intracellular_transitions' in kwargs:
            self.identical_intracellular_transitions = kwargs['identical_intracellular_transitions']
        else:
            self.identical_intracellular_transitions = True
        self.initial_safe_states = [(0,0,0)]
        if 'reward_func_is_known' not in kwargs:
            kwargs['reward_func_is_known'] = True
        if kwargs['reward_func_is_known']:
            if 'reward_func' in kwargs:
                self.reward_func = kwargs['reward_func']
            else:
                self.reward_func = reward_func

        self.initial_state = (
            {
                'agt': {
                    'position': np.array([1, 1], dtype=int),
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

        # derived quantities
        self.n_states = self.n_intracellular_states ** self.n_cells # overestimate
        self.n_actions = self.n_intracellular_actions ** self.n_cells #overestimate


    def available_actions(self, state):
        # check validity of state
        n_positions = 0
        cell = None
        for c in range(self.n_cells):
            if 'position' in state[cell]['agt']:
                cell = c
                n_positions += 1
            if (tree_positions - state[cell]['living_trees']).min() <= 1:
                return []
        if n_positions != 1:
            return []
        # check availability of actions
        action_template = [{'go_to': {}} for _ in range(self.n_cells)] 
        other_cell = 0 if cell == 1 else 1
        if (state[cell]['agt']['position'] == np.array([0, 0])).all():
            stay = cp.deepcopy(action_template)
            stay[cell]['go_to']['position'] = np.array([0, 0])
            right = cp.deepcopy(action_template)
            right[cell]['go_to']['position'] = np.array([0, 1])
            down = cp.deepcopy(action_template)
            down[cell]['go_to']['position'] = np.array([1, 0])
            return [tuple(stay), tuple(right), tuple(down)]
        elif (state[cell]['agt']['position'] == np.array([1, 0])).all():
            stay = cp.deepcopy(action_template)
            stay[cell]['go_to']['position'] = np.array([1, 0])
            up = cp.deepcopy(action_template)
            up[cell]['go_to']['position'] = np.array([0, 0])
            right = cp.deepcopy(action_template)
            right[cell]['go_to']['position'] = np.array([1, 1])
            return [tuple(stay), tuple(up), tuple(right)]
        elif (state[cell]['agt']['position'] == np.array([0, 1])).all():
            stay = cp.deepcopy(action_template)
            stay[cell]['go_to']['position'] = np.array([0, 1])
            left = cp.deepcopy(action_template)
            left[cell]['go_to']['position'] = np.array([0, 0])
            down = cp.deepcopy(action_template)
            down[cell]['go_to']['position'] = np.array([1, 1])
            out = cp.deepcopy(action_template)
            out[other_cell]['go_to']['position'] = np.array([1, 1])
            return [tuple(stay), tuple(left), tuple(down), tuple(out)]
        elif (state[cell]['agt']['position'] == np.array([1, 1])).all():
            stay = cp.deepcopy(action_template)
            stay[cell]['go_to']['position'] = np.array([1, 1])
            up = cp.deepcopy(action_template)
            up[cell]['go_to']['position'] = np.array([0, 1])
            left = cp.deepcopy(action_template)
            left[cell]['go_to']['position'] = np.array([1, 0])
            out = cp.deepcopy(action_template)
            out[other_cell]['go_to']['position'] = np.array([0, 1])
            return [tuple(stay), tuple(up), tuple(left), tuple(out)]
        else:
            return []
        
        
    def is_available(self, state, action):
        for available_action in self.available_actions(state):
            output = True
            for cell in range(self.n_cells):
                if 'position' in action[cell]['go_to'] and 'position' in available_action[cell]['go_to']:
                    if not (action[cell]['go_to']['position'] == available_action[cell]['go_to']['position']).all():
                        output = False
                        break
                elif ('position' in action[cell]['go_to']) ^ ('position' in available_action[cell]['go_to']):
                    output = False
                    break
            if output is True:
                break
        return output
    

    def cellularize(self, element, space: str):
        if space == 'action':
            cellular_action = np.zeros(
                shape=self.n_cells,
                dtype=int,
            )
            for cell in range(self.n_cells):
                if 'position' in element[cell]['go_to']:
                    cellular_action[cell] = element[cell]['go_to']['position'][0] * grid_shape[1] + element[cell]['go_to']['position'][1]
                else:
                    cellular_action[cell] = grid_shape[0] * grid_shape[1]
            return cellular_action
        elif space == 'state':
            cellular_state = np.zeros(shape=self.n_cells,dtype=int)
            for cell in range(self.n_cells):
                for i, tp in enumerate(np.nonzero(tree_positions)):
                    if element[cell]['living_trees'][tp[1], tp[0]] == 1:
                        cellular_state[cell] += 2 ** i
                if 'position' in element[cell]['agt']:
                    cellular_state[cell] += (element[cell]['agt']['position'][0] * grid_shape[1] + element[cell]['agt']['position'][1]) * 2 ** tree_positions.sum()
                else:
                    cellular_state[cell] += grid_shape[0] * grid_shape[1] * 2 ** tree_positions.sum()
            return cellular_state
        else:
            raise ValueError('space must be either action or state')
        

    def decellularize(self, cellular_element, space: str):
        if space == 'action':
            action = [{'go_to': {}} for _ in range(self.n_cells)]
            for cell in range(self.n_cells):
                if cellular_element[cell] < grid_shape[0] * grid_shape[1]:
                    action[cell]['go_to']['position'] = np.array(
                        [
                            cellular_element[cell] // grid_shape[1],
                            cellular_element[cell] % grid_shape[1],
                        ]
                    )
            return tuple(action)
        elif space == 'state':
            state = [{'agt': {}, 'living_trees': np.zeros(shape=grid_shape, dtype=int)} for _ in range(self.n_cells)]
            for cell in range(self.n_cells):
                tmp = cp.deepcopy(cellular_element[cell])
                arr = np.zeros(shape=grid_shape, dtype=int)
                for i, tp in enumerate(np.nonzero(tree_positions)):
                    arr[tp[1], tp[0]] = cp.deepcopy(tmp % 2)
                    tmp = tmp // 2
                state[cell]['living_trees'] = cp.deepcopy(arr)
                if tmp < grid_shape[0] * grid_shape[1]:
                    state[cell]['agt']['position'] = np.array(
                        [
                            tmp // grid_shape[1],
                            tmp % grid_shape[1],
                        ]
                    )
            return tuple(state)
        else:
            raise ValueError('space must be either action or state')
        

    def tabularize(self, element, space: str):
        if space == 'action':
            cellular_action = self.cellularize(element, space='action')
            tabular_action = generalized_cellular2tabular(cellular_action, self.action_space)
            return tabular_action
        elif space == 'state':
            cellular_state = self.cellularize(element, space='state')
            tabular_state = generalized_cellular2tabular(cellular_state, self.state_space)
            return tabular_state
        else:
            raise ValueError('space must be either action or state')
        
    
    def detabularize(self, tabular_element, space: str):
        if space == 'action':
            cellular_action = generalized_tabular2cellular(tabular_element, self.action_space)
            action = self.decellularize(cellular_action, space='action')
            return action
        elif space == 'state':
            cellular_state = generalized_tabular2cellular(tabular_element, self.state_space)
            state = self.decellularize(cellular_state, space='state')
            return state
        else:
            raise ValueError('space must be either action or state')
        

    def initial_policy(self, state):
        action = [{'go_to': {}} for _ in range(self.n_cells)]
        for cell in range(self.n_cells):
            another_cell = (cell + 1) % self.n_cells
            if 'position' in state[cell]['agt']:
                if (state[cell]['agt']['position'] == [1, 1]).all():
                    action[another_cell]['go_to']['position'] = np.array([1, 0])
                elif (state[cell]['agt']['position'] == [1, 0]).all():
                    action[another_cell]['go_to']['position'] = np.array([1, 1])
                elif (state[cell]['agt']['position'] == [0, 1]).all():
                    action[cell]['go_to']['position'] = np.array([1, 1])
                elif (state[cell]['agt']['position'] == [0, 0]).all():
                    action[cell]['go_to']['position'] = np.array([1, 0])
                else:
                    raise ValueError('invalid agent position')
        return tuple(action)
            