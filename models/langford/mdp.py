#   Two kinds of hard MDP problems defeat commonly used RL algorithms like Q-learning.

#   (1) Antishaping.  If rewards in the vicinity of a start state favor
#   staying near a start state, then reward values far from the start
#   state are irrelevant.  The name comes from "reward shaping" which is
#   typically used to make RL easier.  Here, we use it to make RL harder.

#   (2) Combolock.  When most actions lead towards the start state
#   uniform random exploration is relatively useless.  The name comes
#   from "combination lock" where knowing the right sequence of steps to
#   take is the problem.

#   Here, we create families of learning problems which exhibits these
#   characteristics.  

#   We deliberately simplify in several ways to exhibit the problem.  In
#   particular, all transitions are deterministic and only two actions
#   are available in each state.

#   These problems are straightforwardly solvable by special-case
#   algorithms.  They can be solved by general purpose RL algorithms in
#   the E^3 family.  They are not easily solved by Q-learning style
#   algorithms.

from enum import Enum
import random
import numpy as np
class Action(Enum):
    go_left = 1
    go_right = 2

class State_Reward:
    def __init__(self, state=0, reward=0):
        self.state = state
        self.reward = reward

def compare(sr1, sr2):
    return sr1.reward < sr2.reward

def make_translation(num_states):
    translation = []
    for i in range(num_states):
        translation.append(State_Reward(i, random.random()))
    return translation

class MDP:
    def __init__(self):
        self.state = 0
        self.total_reward = 0
        self.num_steps = 0

        self.start_state = 0
        self.horizon = 0
        self.dynamics = []

    def next_state(self, a):
        actions = self.dynamics[self.state]
        if (a == Action.go_left):
            sr = actions[0]
        else:
            sr = actions[1]
        self.state = sr.state
        self.total_reward = self.total_reward + sr.reward
        self.num_steps = self.num_steps + 1
        tr = -1.
        if (self.num_steps == self.horizon):
            tr = self.total_reward / self.horizon
            self.total_reward = 0
            self.state = self.start_state
            self.num_steps = 0
        return sr, tr
    
    def get_reward(self, state, action):
        actions = self.dynamics[state]
        if (action == Action.go_left):
            sr = actions[0]
        else:
            sr = actions[1]
        return sr
    
    def get_action(self, state):
        return self.dynamics[state]

class AntiShape(MDP):
    def __init(self):
        super.__init__()

    def create(self, num_states):
        self.total_reward = 0
        self.num_steps = 0
        self.horizon = num_states*2

        translation = make_translation(num_states)

        self.start_state = translation[0].state
        self.state = self.start_state
        self.dynamics = [None] * num_states
        for i in range(num_states):
            left_state = 0 if i == 0 else i - 1
            right_state = min(i+1,num_states - 1)
            
            left_reward = 0.25 / (left_state+1)
            right_reward = 0.25 / (right_state+1)
            # left_reward = 0.2
            # right_reward = 0.8
            if (right_state == num_states - 1): 
                right_reward = 1.
            
            sr_left = State_Reward(translation[left_state].state, left_reward)
            sr_right = State_Reward(translation[right_state].state, right_reward)
            
            self.dynamics[translation[i].state] = [sr_left, sr_right]

class ComboLock(MDP):
    def __init(self):
        super.__init__()

    def create(self, num_states):
        self.total_reward = 0
        self.num_steps = 0
        self.horizon = num_states*2
        self.num_states = num_states

        translation = make_translation(num_states)

        self.start_state = translation[0].state
        self.state = self.start_state
        self.dynamics = [None] * num_states
        for i in range(num_states):
            left_state = 0
            right_state = 0

            if (random.random() < 0.5):
                left_state = min(i + 1,num_states - 1)
            else:
                right_state = min(i + 1,num_states - 1)
            
            left_reward = 0
            right_reward = 0 #right_state
            if (right_state == num_states-1):
                right_reward = 1.
            if (left_state == num_states-1):
                left_reward = 1.
            
            sr_left = State_Reward(translation[left_state].state, left_reward)
            sr_right = State_Reward(translation[right_state].state, right_reward)
            
            self.dynamics[translation[i].state] = [sr_left, sr_right]

    def perturb(self):
        x = np.random.choice(self.num_states - 2, 1)
        r = 0
        if (x + 1 == self.num_states):
            r = 1
        self.dynamics[0].append(State_Reward(x + 1, r))
        return x + 1

    def add(self, x):
        r = 0
        if (x + 1 == self.num_states):
            r = 1
        self.dynamics[0].append(State_Reward(x + 1, r))

    def reset(self):
        self.dynamics[0] = [self.dynamics[0][0], self.dynamics[0][1]]

    def get_reward(self, state, action):
        actions = self.dynamics[state]
        if (action == Action.go_left):
            sr = actions[0]
        else:
            sr = actions[1]
        return sr
