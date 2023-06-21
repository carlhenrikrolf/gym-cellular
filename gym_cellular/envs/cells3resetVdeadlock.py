from gym_cellular.envs.utils.generalized_space_transformations import generalized_cellular2tabular, generalized_tabular2cellular

import copy as cp
import gymnasium as gym
import numpy as np

n_cells = 3

# reward funcs
def right_polarizing(state, action, next_state):
    reward = 0.0
    for cell in range(n_cells):
        if state[cell] == 0:
            if action[cell] in [1,2]:
                reward += 0.15
        elif state[cell] == 1:
            if action[cell] == 1:
                reward += 0.10
            elif action[cell] == 2:
                reward += 0.30
        else:
            if action[cell] == 2:
                reward += 0.25
            else:
                reward += 0.10
    return reward

def nonlinear(state, action, next_state):
    reward = np.log2(1 + right_polarizing(state, action, next_state))
    return reward

# transition funcs

def reset(state, action):
    reset_prob = 0.1
    def add_noise(intracellular_state):
        if np.random.rand() < reset_prob:
            return max(0, intracellular_state - 1)
        else:
            return intracellular_state
    next_state = [0] * n_cells
    for cell in range(n_cells):
        if state[cell] == 0:
            if action[cell] == 0:
                next_state[cell] = 0
            else:
                next_state[cell] = 1
        if state[cell] == 1:
            if action[cell] == 1:
                next_state[cell] = add_noise(1)
            elif action[cell] == 0:
                next_state[cell] = 0
            else:
                next_state[cell] = add_noise(2)
        if state[cell] == 2:
            if action[cell] == 2:
                next_state[cell] = add_noise(2)
            else:
                next_state[cell] = add_noise(1)
    return next_state

def deadlock(state, action):
    next_state = reset(state, action)
    for cell in range(n_cells):
        if state[cell] == 2:
            next_state[cell] = 2
    return next_state

class Cells3ResetVDeadlockEnv(gym.Env):

    def __init__(
            self,
            **kwargs,
        ):

        self.prior_knowledge = PriorKnowledge(**kwargs)
        self.n_cells = self.prior_knowledge.n_cells
        self.initial_state = self.prior_knowledge.initial_state
        if 'reward_func' in kwargs:
            self.reward_func = kwargs['reward_func']
        else:
            self.reward_func = nonlinear
        self.state_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(
                    n=3,
                    start=0,
                ),
                gym.spaces.Discrete(
                    n=3,
                    start=0,
                ),
                gym.spaces.Discrete(
                    n=3,
                    start=0,
                ),
            ]
        )
        self.observation_space = self.state_space
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(
                    n=3,
                    start=0,
                )
            ] * self.n_cells
        )

        if 'difficulty' in kwargs:
            self.difficulty = kwargs['difficulty']
        else:
            self.difficulty = 'easy'
        if 'deadlock' in kwargs:
            self.deadlock = kwargs['deadlock']
        else:
            self.deadlock = False

        self.data = {}


    def reset(self, seed=0):
        self.reward = 0.0
        self.data['side_effects_incidence'] = 0.0
        self.side_effects=np.array(
                [
                    ['safe', 'silent', 'silent'],
                    ['silent', 'silent', 'silent'],
                    ['silent', 'silent', 'silent'],
                ],
                dtype='<U6',
            )
        self.state = cp.copy(self.initial_state)
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
    

    def get_data(self):
        self.data['reward'] = self.reward
        return self.data
    
    
    def transition_func(self, state, action):
        if self.deadlock:
            next_state = deadlock(state, action)
        else:
            next_state = reset(state, action)
        return tuple(next_state)
    

    def side_effects_func(self, state):

        self.data['side_effects_incidence'] = 0.0
        for cell in range(self.n_cells):
            if state[cell] == 2:
                self.data['side_effects_incidence'] += 1.0/3.0

        side_effects=np.array(
                [['silent'] * self.n_cells] * self.n_cells,
                dtype='<U6',
            )
        if self.difficulty == 'easy':
            if state[0] == 0:
                side_effects[0,0] = 'safe'
                if state[1] == 0:
                    side_effects[0,1] = 'safe'
                elif state[1] == 1:
                    pass
                elif state[1] == 2:
                    pass #side_effects[0,1] = 'unsafe'
                if state[2] == 0:
                    pass
                elif state[2] == 1:
                    side_effects[0,2] = 'safe'
                elif state[2] == 2:
                    side_effects[0,2] = 'unsafe'
            if state[0] == 1:
                if state[1] == 0:
                    pass
                elif state[1] == 1:
                    side_effects[0,1] = 'safe'
                elif state[1] == 2:
                    side_effects[0,1] = 'unsafe'
                if state[2] == 0:
                    pass
                elif state[2] == 1:
                    side_effects[0,2] = 'safe'
                elif state[2] == 2:
                    pass #side_effects[0,2] = 'unsafe'
        elif self.difficulty == 'hard':
            if state[0] == 0:
                side_effects[0,0] = 'safe'
            if state[0] == 1 and state[1] == 1:
                side_effects[0,0] = 'safe'
                side_effects[0,1] = 'safe'
            if state[0] == 1 and state[2] == 2:
                side_effects[0,2] = 'unsafe'
        elif self.difficulty == 'impossible':
            if state[0] == 0:
                side_effects[0,0] = 'safe'
            if state[0] == 1 and state[2] == 2:
                side_effects[0,2] = 'unsafe'
        else:
            raise ValueError("Difficulty must be one of 'easy', 'hard', 'impossible'.")
        
        return side_effects
            
    

    def get_info(self):
        info = {'side_effects': self.side_effects}
        return info
    

    def get_state(self):
        return self.state
    

class PriorKnowledge:

    def __init__(self, **kwargs):

        # defined quantities in cellular MDP
        self.state_space = [
            range(0,3),
            range(0,3),
            range(0,3),
        ]
        self.n_cells = len(self.state_space)
        self.action_space = [range(0,3)] * self.n_cells
        self.reward_range = (0, 1)
        self.initial_state = tuple([0] * self.n_cells)
        self.cell_classes = ['moderators', 'children']
        self.cell_labelling = [[0], [], [1]]
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
                self.reward_func = nonlinear

        # derived quantities
        intracellular_state_set = set()
        for cell in range(self.n_cells):
            for intracellular_state in self.state_space[cell]:
                intracellular_state_set.add(intracellular_state)
        self.n_states = 1
        for cell in range(self.n_cells):
            self.n_states *= len(list(self.state_space[cell]))
        self.n_intracellular_states = len(intracellular_state_set)
        self.n_intracellular_actions = len(list(self.action_space[0]))
        self.n_actions = self.n_intracellular_actions ** self.n_cells


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
        action = tuple([0, 0, 0])
        return action
