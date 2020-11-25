import numpy as np


class Agent(object):
    def __init__(self, lr, gamma, n_actions, n_states, eps_max, eps_min, eps_decay):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.Q = {}

        self.init_Q()

    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0

    def _get_max_action(self, state):
        actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
        max_action = np.argmax(actions)
        return max_action

    def choose_action(self, state):
        r = np.random.random()
        # choose random action
        if (r < self.epsilon):
            action = np.random.choice([i for i in range(self.n_actions)])
        # choose best action otherwise
        else:
            action = self._get_max_action(state)
        return action

    def decrement_epsilon(self):
        # use a linear method, explore other options!
        self.epsilon = self.epsilon*(1-self.eps_decay) if self.epsilon > self.eps_min else self.eps_min

    def learn(self, current_state, current_action, reward, next_state):
        max_action = self._get_max_action(next_state)
        self.Q[(current_state, current_action)] += self.lr*(reward + self.gamma*self.Q[(next_state, max_action)] - self.Q[(current_state, current_action)])
        self.decrement_epsilon()
