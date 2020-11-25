import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from utils import plot_learning_curve


class LinearDQN(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDQN, self).__init__()
        # define layers
        self.fcl1 = nn.Linear(*input_dims, 16)
        self.fcl2 = nn.Linear(16, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print("using: {}".format(self.device))
        self.to(self.device)

    def forward(self, state):
        ly1 = F.relu(self.fcl1(state))
        actions = self.fcl2(ly1)
        return actions


class Agent(object):
    def __init__(self, input_dims, n_actions, lr, gamma=.99, eps = 1.0, eps_dec=1e-5, eps_min=.01):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]

        self.Q = LinearDQN(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        r = np.random.random()
        if r > self.eps:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def decrement_eps(self):
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min

    def learn(self, current_state, current_action, reward, next_state):
        self.Q.optimizer.zero_grad()
        states = T.tensor(current_state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(current_action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[actions]
        q_next = self.Q.forward(next_state).max()
        q_target = reward + self.gamma*q_next
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_eps()


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []

    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, lr=.0001)

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
        scores.append(score)
        eps_history.append(agent.eps)

        if i % 100 == 0:
            avg = np.mean(scores[-100:])
            print("episode: {}\tscore: {:.1f}\tavg score: {:.1f}\teps: {:.2f}".format(i, score, avg, agent.eps))

    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)








