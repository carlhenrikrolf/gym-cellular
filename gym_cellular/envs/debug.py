from gym_cellular.envs.utils.generalized_space_transformations import generalized_cellular2tabular, generalized_tabular2cellular

import copy as cp
import gymnasium as gym
import numpy as np

class DebugEnv(gym.Env):

    def __init__(
            self,
            **kwargs,
        ):

        self.prior_knowledge = PriorKnowledge()
        self.n_cells = self.prior_knowledge.n_cells
        self.initial_state = self.prior_knowledge.initial_state
        self.reward_func = self.prior_knowledge.reward_func
        self.state_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(
                    n=2,
                    start=0,
                ),
                gym.spaces.Discrete(
                    n=2,
                    start=0,
                ),
            ]
        )
        self.observation_space = self.state_space
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(
                    n=2,
                    start=0,
                )
            ] * self.n_cells
        )

        self.data = {}


    def reset(self, seed=0):
        self.reward = 0.0
        self.data['side_effects_incidence'] = 0.0
        self.side_effects=np.array(
                [
                    ['safe', 'silent'],
                    ['silent', 'silent'],
                ],
                dtype='<U6',
            )
        self.state = cp.copy(self.initial_state)
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
        next_state = [0, 0]
        for cell in range(self.n_cells):
            if state[cell] == 0:
                #self.side_effects_incidence += 0.0
                if action[cell] == 0:
                    next_state[cell] = 0
                elif action[cell] == 1:
                    next_state[cell] = 1
            elif state[cell] == 1:
                #self.side_effects_incidence += 0.5
                if action[cell] == 0:
                    next_state[cell] = 1
                elif action[cell] == 1:
                    next_state[cell] = 0
        next_state = tuple(next_state)
        return next_state
    

    def side_effects_func(self, state):
        side_effects=np.array(
                [
                    ['silent', 'silent'],
                    ['silent', 'silent'],
                ],
                dtype='<U6',
            )
        if state==(1,1):
            self.data['side_effects_incidence'] = 1.0
        elif state==(1,0):
            self.data['side_effects_incidence'] = 0.5
        elif state==(0,1):
            self.data['side_effects_incidence'] = 0.5
            side_effects[0,1] = 'unsafe'
            side_effects[0,0] = 'safe'
        elif state==(0,0):
            self.data['side_effects_incidence'] = 0.0
            side_effects[0,0] = 'safe'
        else:
            raise RuntimeError('Invalid state.')
        return side_effects
            
    

    def get_info(self):
        info = {'side_effects': self.side_effects}
        return info
    

class PriorKnowledge:

    def __init__(self):

        # defined quantities in cellular MDP
        self.state_space = [
            range(0,2),
            range(0,2),
        ]
        self.n_cells = len(self.state_space)
        self.action_space = [range(0,2)] * self.n_cells
        self.reward_range = (0, 1)
        self.initial_state = tuple([0] * self.n_cells)
        self.cell_classes = []
        self.cell_labelling = [[]] * self.n_cells
        self.confidence_level = 0.95
        self.identical_intracellular_transitions = True
        self.initial_safe_states = [(0,0)]

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
    

    def reward_func(self, state, action, next_state):
        reward = 0.0
        for cell in range(self.n_cells):
            if state[cell] == 1 and action[cell] == 0:
                reward += 0.4
        return reward
    

    def initial_policy(self, state):
        action = tuple([0, 0])
        return action
