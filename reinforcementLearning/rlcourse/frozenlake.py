import gym
import numpy as np
import matplotlib.pyplot as plt
import sys
from agent import Agent

try:
    args = sys.argv[1:]
    print(args)
    use_best_policy = eval(args[0])
    verbose = eval(args[1])
    use_agent = eval(args[2])
    print(sys.argv[0])
except IndexError:
    use_best_policy = False
    verbose = False

use_agent = True


env = gym.make('FrozenLake-v0')
agent = Agent(lr=.001, gamma=.9, n_actions=4, n_states=16, eps_max=1.0, eps_min=.01, eps_decay=.0000005)
env.reset()
if(verbose):
    env.render()
    print(env.action_space)
    print(env.observation_space)

best_policy = []#{0:actions['>'], 1:actions['>'], 2:actions['v'], 3:actions['<'], 4:actions['v'], 6:actions['v'], 8:actions['>'], 9:actions['>'], 10:actions['v'], 13:actions['>'], 14:actions['>']}


iters = 500000
win_pct = []
scores = []


for i in range(0, iters):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        if(use_best_policy):
            action = best_policy[obs]
        elif(use_agent):
            action = agent.choose_action(obs)
        else:
            action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        agent.learn(obs, action, reward, next_obs)
        score += reward
        obs = next_obs
        #env.render()
    scores.append(score)

    if(i % 100 == 0):
        avg = np.mean(scores[-100:])
        win_pct.append(avg)
        if(i % 1000 == 0):
            print("episode: {}\twin_pct: {:.2f}\tepsilon: {:.2f}".format(i, avg, agent.epsilon))

plt.plot(win_pct)
plt.show()

