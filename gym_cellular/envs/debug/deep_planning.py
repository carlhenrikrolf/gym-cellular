from gym_cellular.envs.utils.generalized_space_transformations import generalized_cellular2tabular, generalized_tabular2cellular

import copy as cp
import gymnasium as gym
import numpy as np

n_cells = 2
n_intracellular_states = 4
n_intracellular_actions = 2

def reward_func(state, action, next_state):
    reward = 0.0
    for cell in range(n_cells):
        if action[cell] == 0:
            reward += 0.05
        elif state[cell] == 3 and action[cell] == 1:
            reward += 0.5
    return reward

reward_range = (0.0, 1.0)

n_states = n_intracellular_states**n_cells
n_actions = n_intracellular_actions**n_cells

class DeepPlanningDebugEnv(gym.Env):

    def __init__(self, **kwargs):
        self.reward_func = reward_func
        self.prior_knowledge = PriorKnowledge(**kwargs)
        self.state_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(
                    n=n_intracellular_states,
                    start=0,
                ),
            ] * n_cells
        )
        self.observation_space = self.state_space
        self.state_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(
                    n=n_intracellular_actions,
                    start=0,
                ),
            ] * n_cells
        )
        self.reward_range = reward_range
        self.data = {}


    def reset(self, seed=0):
        self.reward = 0.0
        self.data['side_effects_incidence'] = 0.0
        self.side_effects=np.array(
                [['safe'] * n_cells] * n_cells,
                dtype='<U6',
            )
        self.state = cp.copy(self.prior_knowledge.initial_state)
        info = self.get_info()
        return self.state, info
    

    def step(self, action):
        next_state = self.transition_func(self.state, action)
        self.reward = self.reward_func(self.state, action, next_state)
        self.side_effects = self.side_effects_func(next_state)
        self.state = next_state
        terminated = False
        truncated = False
        info = self.get_info()
        return self.state, self.reward, terminated, truncated, info
    

    def get_data(self):
        self.data['reward'] = self.reward
        return self.data


    def transition_func(self, state, action):
        next_state = tuple([] * n_cells)
        for cell in range(n_cells):
            if action[cell] == 0:
                next_state[cell] = state[cell]
            elif state[cell] == 3 and action[cell] == 1:
                next_state[cell] = 0
            else:
                next_state[cell] = state[cell] + 1
        return next_state
    
    
    def side_effects_func(self, state):
        self.data['side_effects_incidence'] = 0.0
        side_effects=np.array(
                [['safe'] * self.n_cells] * self.n_cells,
                dtype='<U6',
            )
        return side_effects
    

    def get_info(self):
        info = {'side_effects': self.side_effects}
        return info
    

class PriorKnowledge:


    def __init__(self, **kwargs):
        self.state_space = [range(0, n_intracellular_states)] * n_cells
        self.action_space = [range(0, n_intracellular_actions)] * n_cells
        self.n_cells = n_cells
        self.n_intracellular_states = n_intracellular_states
        self.n_intracellular_actions = n_intracellular_actions
        self.n_states = n_states
        self.n_actions = n_actions
        self.reward_range = reward_range
        self.initial_state = tuple([0] * self.n_cells)
        self.cell_classes = ['regulators']
        self.cell_labelling = [[0]] * n_cells
        self.initial_safe_states = [tuple([0] * n_cells)]
        self.reward_func = reward_func
        if 'confidence_level' in kwargs:
            self.confidence_level = kwargs['confidence_level']
        else:
            self.confidence_level = 0.95
        if 'identical_intracellular_transitions' in kwargs:
            self.identical_intracellular_transitions = kwargs['identical_intracellular_transitions']
        else:
            self.identical_intracellular_transitions = True


    def cellularize(self, element, space):
        cellular_element = list(element)
        return cellular_element
    

    def decellularize(self, cellular_element, space):
        element = tuple(cellular_element)
        return element
    

    def tabularize(self, element, space):
        cellular_element = list(element)
        tabular_element = generalized_cellular2tabular(cellular_element, space)
        return tabular_element
    
    
    def detabularize(self, tabular_element, space):
        cellular_element = generalized_tabular2cellular(tabular_element, space)
        element = tuple(cellular_element)
        return element
    

    def initial_policy(self, state):
        action = tuple([0] * n_cells)
        return action